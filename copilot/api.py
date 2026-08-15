"""HTTP surface: ask, stream, replay.

    POST /ask                   one question, verified answer + evidence
    GET  /stream/alerts         SSE feed of alerts with lead time
    GET  /stream/margins        SSE feed of scored cycles
    GET  /envelope              the true failure boundary, for plotting
    GET  /health                readiness, versions, gate status
    GET  /                      minimal console

Thin by design. Every endpoint delegates to the same engine the CLI uses, so
there is exactly one code path from question to verified answer and the API
cannot drift from the terminal.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from copilot.engine import Engine
from copilot.ops import execute
from copilot.ir import parse_plan, PlanError
from copilot.reliability import audit_calibration, check_invariants
from copilot.session import SessionState
from copilot.stream import StreamScorer, replay

app = FastAPI(
    title="Industrial Copilot — Margin Engine",
    description="Computes distance to the failure boundary. No number is authored by a model.",
    version="0.1.0",
)

_engine: Engine | None = None
_sessions: dict[str, SessionState] = {}


def engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine.build()
    return _engine


# --------------------------------------------------------------------------
# Ask
# --------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = "default"
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


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    eng = engine()
    state = _sessions.setdefault(request.session_id, SessionState())
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
    )


@app.delete("/session/{session_id}")
def reset_session(session_id: str) -> dict[str, str]:
    _sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


def _sse(payload: dict[str, Any], event: str = "message") -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


async def _tick_stream(
    *, speed: float, limit: int | None, machine: str | None, alerts_only: bool
) -> AsyncIterator[str]:
    scorer = StreamScorer()
    loop = asyncio.get_running_loop()
    ticks = replay(limit=limit, speed=0.0, machine=machine, scorer=scorer)
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
    limit: int | None = Query(None, ge=1),
    machine: str | None = None,
) -> StreamingResponse:
    return StreamingResponse(
        _tick_stream(speed=speed, limit=limit, machine=machine, alerts_only=True),
        media_type="text/event-stream",
    )


@app.get("/stream/margins")
async def stream_margins(
    speed: float = Query(600.0, ge=0.0),
    limit: int | None = Query(500, ge=1),
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
    return _CONSOLE


_CONSOLE = """<!doctype html>
<meta charset="utf-8"><title>Margin Engine</title>
<style>
 body{font:14px ui-monospace,Menlo,monospace;margin:0;background:#0c1315;color:#e6edeb}
 header{padding:14px 20px;border-bottom:1px solid #26383a}
 h1{font-size:15px;margin:0;letter-spacing:.02em}
 .sub{color:#84989a;font-size:12px;margin-top:4px}
 main{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#26383a;min-height:calc(100vh - 62px)}
 section{background:#0c1315;padding:16px 20px;overflow:auto;max-height:calc(100vh - 62px)}
 h2{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:#4ecfbb;margin:0 0 12px}
 input{width:100%;padding:9px 11px;background:#131e20;border:1px solid #26383a;color:#e6edeb;
       font:inherit;border-radius:4px}
 pre{white-space:pre-wrap;line-height:1.55;margin:12px 0 0}
 .meta{color:#84989a;font-size:11px}
 .alert{border-left:2px solid #e0a54b;padding:6px 10px;margin:6px 0;background:#131e20}
 .alert.crossed{border-color:#ec7c6a}
 .alert.sensor{border-color:#84989a}
 .k{color:#4ecfbb}
</style>
<header>
  <h1>Industrial Copilot — Margin Engine</h1>
  <div class="sub">Distance to the failure boundary. No number is authored by a model.</div>
</header>
<main>
  <section>
    <h2>Ask</h2>
    <input id="q" placeholder="Why are we seeing more failures at high rotational speeds?" autofocus>
    <pre id="a" class="meta">Press Enter to ask.</pre>
  </section>
  <section>
    <h2>Live alerts</h2>
    <div id="alerts"></div>
  </section>
</main>
<script>
const q=document.getElementById('q'),a=document.getElementById('a');
q.addEventListener('keydown',async e=>{
  if(e.key!=='Enter'||!q.value.trim())return;
  a.textContent='…';
  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:q.value})});
  const d=await r.json();
  a.textContent=d.answer+'\\n\\n'+(d.verified?'✓ verified':'✗ UNVERIFIED')+
    ` · ${d.tier} tier · ${d.elapsed_ms} ms`;
});
const es=new EventSource('/stream/alerts?speed=2000&limit=4000');
const box=document.getElementById('alerts');
es.addEventListener('alert',e=>{
  const d=JSON.parse(e.data);
  const el=document.createElement('div');
  el.className='alert '+d.kind;
  el.innerHTML=`<span class="k">${d.machine}</span> ${d.mode} · ${d.kind}` +
    (d.lead_time_min!=null?` · crosses in ~${d.lead_time_min} min`:'') +
    `<div class="meta">${d.message}${d.fix?'<br>fix: '+d.fix:''}</div>`;
  box.prepend(el);
  while(box.children.length>40)box.lastChild.remove();
});
</script>
"""
