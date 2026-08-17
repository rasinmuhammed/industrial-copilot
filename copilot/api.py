"""HTTP surface: ask, stream, replay.

    POST /ask                   one question, verified answer + evidence
    GET  /stream/alerts         SSE feed of alerts with lead time
    GET  /stream/margins        SSE feed of scored cycles
    GET  /envelope              the true failure boundary, for plotting
    GET  /envelope/projection   the boundary as a function of FUTURE wear
    GET  /explorer              the Operating Envelope Explorer
    GET  /fleet                 every machine ranked on one axis of risk
    GET  /fleet/view            the fleet control room
    GET  /health                readiness, versions, gate status
    GET  /                      the ask console, with evidence drill-down
    GET  /reliability           Gates 2 and 3, live and interactive

Thin by design. Every endpoint delegates to the same engine the CLI uses, so
there is exactly one code path from question to verified answer and the API
cannot drift from the terminal.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from copilot.engine import Engine
from copilot.ops import execute
from copilot.ir import parse_plan, PlanError
from copilot.reliability import (
    audit_calibration,
    check_invariants,
    diagnose_drift,
    estimate_threshold,
)
from copilot.session import SessionState
from copilot.ops.registry import TABLE
from copilot.stream import StreamScorer, replay
from copilot.cmms import AlertOutcome, CMMSStore, WorkOrder, generate_from_replay
from copilot.feedback import FeedbackLearner
from copilot.rul import fleet_rul, machine_rul

_HERE = Path(__file__).resolve().parent

app = FastAPI(
    title="Argus",
    description="Computes distance to the failure boundary. No number is authored by a model.",
    version="0.1.0",
)

_engine: Engine | None = None

# Conversational state, keyed by an EXPLICIT session id.
#
# Two defects lived here, both found by probing the HTTP surface rather than
# the engine:
#
#   CROSS-USER BLEED. `session_id` defaulted to the literal "default", so every
#   caller who did not supply one shared a single conversation. Engineer A
#   scoped to L variants; Engineer B then asked for the OVERALL failure rate and
#   received A's L-variant scope — a confident, verified, correctly computed
#   answer to a question nobody asked. Follow-up context is acceptance criterion
#   4 of the brief, and it was the exact mechanism that leaked.
#
#   UNBOUNDED GROWTH. Nothing was ever evicted. 5,000 distinct ids retained
#   5,001 states for the life of the process.
#
# A request with no session id is now STATELESS: it gets a fresh state that is
# never stored, so it cannot inherit or contaminate anything. Continuity is
# opt-in, which is the only safe default for a shared endpoint.
_MAX_SESSIONS = 1000
_sessions: "OrderedDict[str, SessionState]" = OrderedDict()
_sessions_lock = threading.Lock()


def _session_for(session_id: str | None) -> SessionState:
    """Fresh and unshared when no id is given; LRU-bounded when one is."""
    if not session_id:
        return SessionState()
    with _sessions_lock:
        state = _sessions.get(session_id)
        if state is None:
            state = SessionState()
            _sessions[session_id] = state
            while len(_sessions) > _MAX_SESSIONS:
                _sessions.popitem(last=False)
        else:
            _sessions.move_to_end(session_id)
        return state


def engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine.build()
    return _engine


_cmms_store: CMMSStore | None = None
_feedback_learner: FeedbackLearner | None = None


def cmms_store() -> CMMSStore:
    global _cmms_store
    if _cmms_store is None:
        _cmms_store = CMMSStore()
    return _cmms_store


def feedback_learner() -> FeedbackLearner:
    global _feedback_learner
    if _feedback_learner is None:
        _feedback_learner = FeedbackLearner(cmms_store())
    return _feedback_learner


# --------------------------------------------------------------------------
# Ask
# --------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    #: Omit for a one-shot question. Supply a stable id to keep follow-up
    #: context — and note that anyone sharing the id shares the conversation.
    session_id: str | None = None
    include_evidence: bool = False


class AskResponse(BaseModel):
    answer: str
    narration: str
    verified: bool
    refused: bool
    op: str | None
    tier: str
    scope: str
    elapsed_ms: float
    plan_ms: float
    exec_ms: float
    replay_handle: str
    warnings: list[dict[str, str]] = []
    evidence: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    plan_json: dict[str, Any] | None = None


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    eng = engine()
    state = _session_for(request.session_id)
    answer = eng.ask(request.question, state)

    bundle = answer.bundle
    return AskResponse(
        answer=answer.text,
        narration=answer.narration,
        verified=answer.verified,
        refused=answer.refused,
        op=answer.plan.op.value if answer.plan else None,
        tier=answer.tier,
        scope=state.scope_line(),
        elapsed_ms=round(answer.elapsed_ms, 2),
        plan_ms=round(answer.plan_ms, 3),
        exec_ms=round(answer.exec_ms, 2),
        replay_handle=answer.replay_handle,
        warnings=[
            {"code": w.code, "severity": w.severity.value, "message": w.message}
            for w in (bundle.warnings if bundle else [])
        ],
        evidence=(
            {sid: {"value": s.value, "unit": s.unit, "n": s.n, "quality": s.quality.value}
             for sid, s in bundle.slots.items()}
            if request.include_evidence and bundle
            else None
        ),
        rows=bundle.rows if bundle else [],
        plan_json=json.loads(answer.plan.model_dump_json(exclude_none=True)) if answer.plan else None,
    )


@app.delete("/session/{session_id}")
def reset_session(session_id: str) -> dict[str, str]:
    with _sessions_lock:
        _sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/telemetry", response_class=HTMLResponse)
def telemetry_view():
    with open(_HERE / "static" / "telemetry.html") as f:
        return f.read()


@app.get("/api/telemetry")
def api_telemetry(limit: int = 100, offset: int = 0, sort: str = "udi", order: str = "desc"):
    valid_cols = [
        "udi", "product_id", "type", "air_temperature_k", "process_temperature_k",
        "rotational_speed_rpm", "torque_nm", "tool_wear_min", "machine_failure", "timestamp"
    ]
    if sort not in valid_cols:
        sort = "udi"
    order = "DESC" if order.lower() == "desc" else "ASC"
    
    con = engine().ctx.con
    try:
        total = con.sql("SELECT COUNT(*) FROM ai4i").fetchone()[0]
        df = con.sql(f"""
            SELECT 
                TIMESTAMP '2026-01-01 00:00:00' + INTERVAL (udi * 5) MINUTE AS timestamp,
                *
            FROM ai4i
            ORDER BY {sort} {order}
            LIMIT {limit} OFFSET {offset}
        """).df()
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        return {"total": total, "rows": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


def _sse(payload: dict[str, Any], event: str = "message") -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


# A stream must terminate. /stream/alerts defaulted to limit=None, which at the
# default takt multiplier runs 10,000 cycles at 0.2 s each — 33 minutes per
# connection, with no ceiling on `limit` and no ceiling on concurrent
# connections. Each holds a StreamScorer with per-machine observer state, so a
# handful of idle browser tabs is a resource leak and a deliberate handful is a
# denial of service.
#
# Bounds belong on the server. A client asking for more gets the ceiling, not an
# error, because truncating a demo stream is harmless and refusing it is not.
MAX_STREAM_TICKS = 5_000
DEFAULT_STREAM_TICKS = 2_000


async def _tick_stream(
    *, speed: float, limit: int | None, machine: str | None, alerts_only: bool
) -> AsyncIterator[str]:
    scorer = StreamScorer()
    bounded = min(limit or DEFAULT_STREAM_TICKS, MAX_STREAM_TICKS)
    ticks = replay(limit=bounded, speed=0.0, machine=machine, scorer=scorer)
    delay = (120.0 / speed) if speed > 0 else 0.0

    for tick in ticks:
        if alerts_only:
            for alert in tick.alerts:
                yield _sse(alert.as_dict(), "alert")
        else:
            yield _sse(tick.as_dict(), "tick")
        if delay:
            await asyncio.sleep(delay)
        else:
            # Yield control so a fast replay does not block the event loop.
            await asyncio.sleep(0)

    yield _sse(
        {
            "ticks": scorer.ticks,
            "alerts": scorer.alerts_raised,
            "suppressed": scorer.alerts_suppressed,
        },
        "summary",
    )


@app.get("/stream/alerts")
async def stream_alerts(
    speed: float = Query(600.0, ge=0.0, description="takt multiplier; 0 = as fast as possible"),
    limit: int | None = Query(
        DEFAULT_STREAM_TICKS, ge=1, le=MAX_STREAM_TICKS,
        description="cycles to replay; server-capped",
    ),
    machine: str | None = None,
) -> StreamingResponse:
    return StreamingResponse(
        _tick_stream(speed=speed, limit=limit, machine=machine, alerts_only=True),
        media_type="text/event-stream",
    )


@app.get("/stream/margins")
async def stream_margins(
    speed: float = Query(600.0, ge=0.0),
    limit: int | None = Query(500, ge=1, le=MAX_STREAM_TICKS),
    machine: str | None = None,
) -> StreamingResponse:
    return StreamingResponse(
        _tick_stream(speed=speed, limit=limit, machine=machine, alerts_only=False),
        media_type="text/event-stream",
    )


# --------------------------------------------------------------------------
# Envelope — the true boundary, for plotting
# --------------------------------------------------------------------------


@app.get("/envelope")
def envelope(
    tool_wear_min: float = Query(150.0, ge=0.0),
    rotational_speed_rpm: float = Query(1500.0, gt=0.0),
    torque_nm: float = Query(45.0, ge=0.0),
    product_type: str = Query("L", pattern="^[LMH]$"),
) -> dict[str, Any]:
    """The feasible region is computed, not a classifier's decision surface.

    That distinction is the point: no approach based on a learned failure
    probability can draw this boundary correctly.
    """
    plan = parse_plan(
        {
            "op": "envelope",
            "params": {
                "tool_wear_min": tool_wear_min,
                "rotational_speed_rpm": rotational_speed_rpm,
                "torque_nm": torque_nm,
                "product_type": product_type,
            },
        }
    )
    bundle = execute(plan, engine().ctx)
    curve = [
        {
            "rpm": int(key.split(".")[1].removesuffix("rpm")),
            "torque_min": bundle.slots[key].value,
            "torque_max": bundle.slots[key.replace("torque_min", "torque_max")].value,
        }
        for key in bundle.slots
        if key.startswith("curve.") and key.endswith(".torque_min")
        and bundle.slots[key].value is not None
    ]
    return {
        "at": {
            "tool_wear_min": tool_wear_min,
            "rotational_speed_rpm": rotational_speed_rpm,
            "torque_nm": torque_nm,
            "product_type": product_type,
        },
        "current": {
            "safe": bundle.slots["current.safe"].value,
            "fired": bundle.slots["current.fired"].value,
        },
        "safe_torque": {
            "min": bundle.slots.get("safe.torque_min").value if "safe.torque_min" in bundle.slots else None,
            "max": bundle.slots.get("safe.torque_max").value if "safe.torque_max" in bundle.slots else None,
        },
        "fix": {
            k.removeprefix("fix."): bundle.slots[k].value
            for k in bundle.slots
            if k.startswith("fix.")
        },
        "curve": curve,
    }


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


@app.get("/envelope/projection")
def envelope_projection(
    rotational_speed_rpm: float = Query(1400.0, gt=0.0),
    tool_wear_min: float = Query(150.0, ge=0.0),
    torque_nm: float = Query(45.0, ge=0.0),
    product_type: str = Query("L", pattern="^[LMH]$"),
    horizon_cycles: int = Query(120, ge=1, le=2000),
    steps: int = Query(5, ge=2, le=12),
) -> dict[str, Any]:
    """The safe operating window as a function of FUTURE tool wear.

    The novel view. An envelope is usually drawn as a static region — "are we
    inside it?" But the overstrain ceiling is threshold/wear, so the window
    *closes* as the tool wears, and its binding constraint switches from power
    overload to overstrain part-way through the tool's life.

    That turns two separate questions — "how much room do I have?" and "how long
    have I got?" — into one picture: how fast the room is disappearing. It is
    drawable only because the boundary is computed rather than learned.
    """
    from copilot.physics import OSF_THRESHOLD, PWF_HIGH, PWF_LOW, RAD_PER_RPM, WEAR_RATE_PER_CYCLE

    omega = rotational_speed_rpm * RAD_PER_RPM
    threshold = OSF_THRESHOLD[product_type]
    wear_rate = WEAR_RATE_PER_CYCLE[product_type]
    floor = PWF_LOW / omega
    overload_ceiling = PWF_HIGH / omega

    # Wear at which the overstrain ceiling meets the stall floor: beyond this,
    # NO torque satisfies every constraint at this speed.
    closure_wear = threshold * omega / PWF_LOW
    closure_cycles = max(0.0, (closure_wear - tool_wear_min) / wear_rate)

    frames = []
    for i in range(steps):
        cycles = horizon_cycles * i / (steps - 1)
        wear = tool_wear_min + cycles * wear_rate
        osf_ceiling = threshold / wear if wear > 0 else float("inf")
        ceiling = min(overload_ceiling, osf_ceiling)
        frames.append(
            {
                "cycles_ahead": round(cycles, 1),
                "minutes_ahead": round(cycles * 120 / 60, 1),
                "wear_min": round(wear, 1),
                "torque_min": round(floor, 2),
                "torque_max": round(max(floor, ceiling), 2),
                "width": round(max(0.0, ceiling - floor), 2),
                "binding": "overstrain" if osf_ceiling < overload_ceiling else "overload",
                "open": ceiling > floor,
            }
        )

    first, last = frames[0], frames[-1]
    shrink = (
        (first["width"] - last["width"]) / first["width"] * 100.0 if first["width"] > 0 else 0.0
    )
    return {
        "at": {
            "rotational_speed_rpm": rotational_speed_rpm,
            "tool_wear_min": tool_wear_min,
            "torque_nm": torque_nm,
            "product_type": product_type,
        },
        "floor": round(floor, 2),
        "overload_ceiling": round(overload_ceiling, 2),
        "closure_wear_min": round(closure_wear, 1),
        "closure_cycles": round(closure_cycles, 1),
        "shrink_pct": round(shrink, 1),
        "binding_switches": first["binding"] != last["binding"],
        "frames": frames,
    }


def _static(name: str) -> str:
    path = _HERE / "static" / name
    if not path.exists():
        raise HTTPException(500, f"{name} is missing from the package")
    return path.read_text(encoding="utf-8")


@app.get("/static/app.css")
def stylesheet() -> Response:
    return Response(_static("app.css"), media_type="text/css")


@app.get("/explorer", response_class=HTMLResponse)
def explorer() -> str:
    """The Operating Envelope Explorer."""
    return _static("explorer.html")


@app.get("/fleet")
def fleet(
    through_udi: int | None = Query(None, ge=1, description="replay playhead; default = latest"),
    history: int = Query(24, ge=2, le=120, description="cycles of margin history per machine"),
) -> dict[str, Any]:
    """Every machine ranked on ONE axis of risk.

    This is the property that makes a fleet view possible at all. Margins are
    normalised by their own threshold, so a thermal risk and a torque risk land
    on the same scale and can be ordered against each other. Probabilities from
    separate models cannot: a 0.3 from a heat model and a 0.3 from a wear model
    are not the same quantity and ranking them together is meaningless.

    Returns each machine's latest state as of the playhead, its worst
    normalised margin, which rule is binding, and a short history for a
    sparkline.
    """
    con = engine().ctx.con
    ceiling = through_udi or con.execute(
        f"SELECT max(udi) FROM {TABLE}"  # noqa: S608
    ).fetchone()[0]

    rows = con.execute(
        f"""WITH scoped AS (
              SELECT * FROM {TABLE} WHERE udi <= ?
            ),
            ranked AS (
              SELECT *, row_number() OVER (PARTITION BY machine_id ORDER BY udi DESC) AS rn
              FROM scoped
            )
            SELECT machine_id, product_type, udi, ts,
                   rotational_speed_rpm, torque_nm, tool_wear_min, power_w,
                   temp_delta_k, hdf_distance, pwf_distance, osf_distance,
                   worst_normalised_margin, machine_failure
            FROM ranked WHERE rn = 1 ORDER BY worst_normalised_margin""",  # noqa: S608
        [ceiling],
    ).fetchall()

    names = [
        "machine", "variant", "udi", "ts", "rpm", "torque", "wear", "power",
        "temp_delta", "hdf", "pwf", "osf", "worst", "failed",
    ]

    machines = []
    for raw in rows:
        r = dict(zip(names, raw))
        distances = {"HDF": r["hdf"], "PWF": r["pwf"], "OSF": r["osf"]}
        binding = min(distances, key=distances.get)
        spark = con.execute(
            f"""SELECT worst_normalised_margin FROM {TABLE}
                WHERE machine_id = ? AND udi <= ? ORDER BY udi DESC LIMIT ?""",  # noqa: S608
            [r["machine"], ceiling, history],
        ).fetchall()

        worst = float(r["worst"])
        machines.append(
            {
                "machine": r["machine"],
                "variant": r["variant"],
                "udi": int(r["udi"]),
                "worst_margin": round(worst, 4),
                "binding": binding,
                "state": "alert" if worst < 0 else ("watch" if worst < 0.05 else "normal"),
                "rpm": round(float(r["rpm"]), 1),
                "torque": round(float(r["torque"]), 2),
                "wear": round(float(r["wear"]), 1),
                "power": round(float(r["power"])),
                "distances": {k: round(float(v), 4) for k, v in distances.items()},
                "history": [round(float(v[0]), 4) for v in reversed(spark)],
            }
        )

    counts = {"alert": 0, "watch": 0, "normal": 0}
    for m in machines:
        counts[m["state"]] += 1

    return {
        "playhead": int(ceiling),
        "max_udi": int(con.execute(f"SELECT max(udi) FROM {TABLE}").fetchone()[0]),  # noqa: S608
        "machines": machines,
        "counts": counts,
        "worst": machines[0] if machines else None,
    }


@app.get("/fleet/view", response_class=HTMLResponse)
def fleet_view() -> str:
    """The fleet control room."""
    return _static("fleet.html")


@app.post("/reliability/drift")
def drift_probe(
    sensor: str = Query("air_temperature_k", pattern="^(air_temperature_k|rotational_speed_rpm|torque_nm)$"),
    delta: float = Query(0.0),
) -> dict[str, Any]:
    """Inject a fault and watch the system diagnose its own inputs (Gate 2).

    The point is the asymmetry. A drifting thermocouple and a genuinely slowing
    process produce the SAME symptom — the heat-dissipation alert count moves —
    but the process does not control ambient temperature, so the invariants
    separate them. A 0.4 K drift halves HDF alerts, which a conventional copilot
    reports as a good month.
    """
    from copilot.reliability.invariants import _window_stats, _z_of_mean

    con = engine().ctx.con
    con.execute(
        f"""CREATE OR REPLACE TEMP VIEW _drifted AS
            SELECT * EXCLUDE (temp_delta_k, {sensor}),
                   {sensor} + {delta} AS {sensor},
                   process_temperature_k - (CASE WHEN '{sensor}' = 'air_temperature_k'
                       THEN air_temperature_k + {delta} ELSE air_temperature_k END) AS temp_delta_k
            FROM {TABLE}"""  # noqa: S608
    )
    report = diagnose_drift(con, window_where="TRUE", table="_drifted", baseline_table=TABLE)

    fires = con.execute(
        "SELECT count(*) FROM _drifted WHERE temp_delta_k < 8.6 AND rotational_speed_rpm < 1380"
    ).fetchone()[0]
    baseline = con.execute(
        f"SELECT count(*) FROM {TABLE} WHERE hdf_rule"  # noqa: S608
    ).fetchone()[0]

    return {
        "sensor": sensor,
        "delta": delta,
        "verdict": report.verdict.value,
        "explanation": report.explanation,
        "z": {
            "temp_delta": round(report.z_temp_delta, 2),
            "rotational_speed": round(report.z_rotational_speed, 2),
            "torque": round(report.z_torque, 2),
        },
        "hdf_alerts": {"baseline": int(baseline), "observed": int(fires),
                       "change_pct": round((fires - baseline) / baseline * 100, 1) if baseline else 0.0},
        "invariants": [
            {"code": i.code, "description": i.description, "holds": i.holds,
             "observed": round(i.observed, 4), "expected": round(i.expected, 4),
             "detail": i.detail}
            for i in report.invariants
        ],
    }


@app.post("/reliability/kb")
def kb_probe(error_pct: float = Query(0.0, ge=-20.0, le=20.0)) -> dict[str, Any]:
    """Perturb a documented threshold and watch the KB audit itself (Gate 3).

    Model-free and directional: surprise failures mean the rule is too loose,
    false alarms mean too tight. Zero only at the true threshold, monotone in
    the size of the error, and it works with the delayed labels a real CMMS
    produces.
    """
    con = engine().ctx.con
    factor = 1 + error_pct / 100.0
    con.execute(
        f"""CREATE OR REPLACE TEMP VIEW _perturbed AS
            SELECT * EXCLUDE (osf_rule),
                   (overstrain_min_nm > osf_threshold_min_nm * {factor}) AS osf_rule
            FROM {TABLE}"""  # noqa: S608
    )
    report = audit_calibration(con, table="_perturbed")
    return {
        "error_pct": error_pct,
        "healthy": report.healthy,
        "summary": report.summary(),
        "rules": [
            {"mode": r.mode, "surprise_failures": r.surprise_failures,
             "false_alarms": r.false_alarms, "total_signal": r.total_signal,
             "direction": r.direction.value, "advice": r.advice()}
            for r in report.rules
        ],
    }


@app.get("/reliability/thresholds")
def threshold_discovery() -> dict[str, Any]:
    """Re-derive the documented limits from outcomes alone.

    The bracket between the largest non-failing value and the smallest failing
    one is a valid estimate of a deterministic boundary — and its WIDTH is the
    honest uncertainty. L recovers to 0.01% on 87 supporting failures; H to
    3.4% on 2. That is why a knowledge-base entry must carry an interval and
    not a point.
    """
    con = engine().ctx.con
    documented = {"L": 11000.0, "M": 12000.0, "H": 13000.0}
    out = []
    for variant, doc in documented.items():
        lo, hi, mid, support = estimate_threshold(
            con, metric_column="overstrain_min_nm", label_column="osf", product_type=variant
        )
        out.append({
            "variant": variant, "documented": doc,
            "lower": round(lo, 1), "upper": round(hi, 1), "midpoint": round(mid, 1),
            "width": round(hi - lo, 1),
            "error_pct": round(abs(mid - doc) / doc * 100, 3),
            "support": support,
        })
    return {"metric": "overstrain_min_nm", "estimates": out}


@app.get("/reliability", response_class=HTMLResponse)
def reliability_view() -> str:
    """The reliability console."""
    return _static("reliability.html")


@app.get("/health")
def health() -> dict[str, Any]:
    eng = engine()
    con = eng.ctx.con
    calibration = audit_calibration(con)
    invariants = check_invariants(con)
    return {
        "status": "ok" if calibration.healthy and all(i.holds for i in invariants) else "degraded",
        "kb_version": eng.ctx.kb_version,
        "data_version": eng.ctx.data_version,
        "provider": eng.provider_name,
        "kb_calibration": {
            "healthy": calibration.healthy,
            "rules": [
                {
                    "mode": r.mode,
                    "surprise_failures": r.surprise_failures,
                    "false_alarms": r.false_alarms,
                    "direction": r.direction.value,
                }
                for r in calibration.rules
            ],
        },
        "invariants": [
            {"code": i.code, "holds": i.holds, "observed": round(i.observed, 4)}
            for i in invariants
        ],
        "tier_distribution": eng.router.tier_distribution(),
    }


@app.get("/", response_class=HTMLResponse)
def console() -> str:
    return _static("ask.html")


# --------------------------------------------------------------------------
# CMMS — work order lifecycle
# --------------------------------------------------------------------------


class CloseRequest(BaseModel):
    outcome: str  # confirmed | false_alarm | wrong_mode | inconclusive
    confirmed_mode: str | None = None
    technician_id: str | None = None
    notes: str | None = None
    variant: str = "L"  # product type, for KB weight keying


@app.get("/cmms/work_orders")
def list_work_orders(
    open_only: bool = False,
    limit: int = Query(50, le=200),
) -> dict[str, Any]:
    """List work orders, most-recent first."""
    wos = cmms_store().list(limit=limit, open_only=open_only)
    return {
        "work_orders": [w.as_dict() for w in wos],
        "summary": cmms_store().summary(),
    }


@app.post("/cmms/work_orders")
def create_work_order(body: dict[str, Any]) -> dict[str, Any]:
    """Create a work order. Set raised_by='SYNTHETIC' for demo orders."""
    import uuid
    from copilot.cmms import _utcnow
    wo = WorkOrder(
        id         = body.get("id") or f"WO-{uuid.uuid4().hex[:8].upper()}",
        machine_id = body["machine_id"],
        udi        = int(body["udi"]),
        alert_mode = body["alert_mode"],
        raised_at  = _utcnow(),
        raised_by  = body.get("raised_by", "copilot-api"),
    )
    cmms_store().create(wo)
    return wo.as_dict()


@app.post("/cmms/work_orders/{wo_id}/close")
def close_work_order(wo_id: str, body: CloseRequest) -> dict[str, Any]:
    """Record a technician outcome and trigger KB weight update."""
    try:
        outcome = AlertOutcome(body.outcome)
    except ValueError:
        raise HTTPException(400, f"unknown outcome: {body.outcome}")

    wo = cmms_store().close(
        wo_id,
        outcome,
        confirmed_mode=body.confirmed_mode,
        technician_id=body.technician_id,
        notes=body.notes,
    )
    if wo is None:
        raise HTTPException(404, "work order not found or already closed")

    update = feedback_learner().apply(wo, variant=body.variant)
    return {"work_order": wo.as_dict(), "kb_update": update}


@app.post("/cmms/seed")
def seed_cmms(limit: int = Query(20, le=100)) -> dict[str, Any]:
    """Seed the CMMS with SYNTHETIC work orders from the AI4I replay."""
    wos = generate_from_replay(cmms_store(), limit=limit)
    # Replay all closed orders through the learner
    learner = feedback_learner()
    for wo in wos:
        if not wo.is_open():
            learner.apply(wo)
    return {
        "seeded": len(wos),
        "summary": cmms_store().summary(),
    }


@app.get("/cmms/feedback")
def feedback_report() -> dict[str, Any]:
    """KB weight adjustments accumulated from work-order outcomes."""
    return {
        "weights": feedback_learner().report(),
        "summary": cmms_store().summary(),
    }


# --------------------------------------------------------------------------
# RUL — remaining useful life
# --------------------------------------------------------------------------


@app.get("/rul")
def rul_fleet() -> dict[str, Any]:
    """Inverse-Gaussian RUL estimates for every virtual machine."""
    return fleet_rul()


@app.get("/rul/{machine_id}")
def rul_machine(machine_id: str) -> dict[str, Any]:
    """Inverse-Gaussian RUL for one machine with 90% conformal interval."""
    result = machine_rul(machine_id)
    if result is None:
        raise HTTPException(404, f"machine {machine_id!r} not found")
    return result
