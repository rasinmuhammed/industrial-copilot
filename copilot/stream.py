"""Real-time streaming inference.

Three throughput figures, measured, for three different things - conflating
them would overstate the case:

    raw margin arithmetic (tuples)        0.22 us   4.6 M/sec/core
    evaluate() with a Margins object      1.09 us   918 k/sec/core
    full scorer: robust track + alerting  ~12 us     ~85 k/sec/core

A 2,000-machine site at 1 Hz needs 2,000/sec, so even the full scorer carries a
site on a single core with ~40x headroom, and the whole 1,000-factory fleet on
about two dozen cores.

That speed is not the result of optimisation. **There is no model to run at
inference time.** The intelligence lives in the knowledge base, built offline;
online is arithmetic. Conventional systems put intelligence in weights evaluated
online, so latency scales with intelligence and every site needs the artifact
versioned and drift-monitored. This inverts that.

Alerts carry a **lead time**, not a severity label, because the problem Operon
names is that by the time you find the anomaly the batch is already compromised.

Two tracks run side by side, for a reason:

    instantaneous   stateless, microseconds   -> display and drill-down
    robust          Hampel over a short window -> ALERTING decisions

A single spike never pages anyone. It is still visible on the instantaneous
track, tagged.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from copilot.ingest import connect
from copilot.observer import FleetObserver, TrustReport
from copilot.physics import (
    OSF_THRESHOLD,
    TWF_WINDOW,
    WEAR_RATE_PER_CYCLE,
    OperatingPoint,
    evaluate,
)
from copilot.reliability.intervals import (
    DEFAULT_UNCERTAINTY,
    Uncertainty,
    Verdict,
    evaluate_interval,
)

__all__ = ["AlertKind", "Alert", "Tick", "StreamScorer", "replay"]

# Consecutive robust-track violations before an alert fires. Debouncing is not
# optional at fleet scale: a margin crossing on every sample would bury an
# operator within minutes.
TRUST_RECOVERY = 30   # consecutive clean cycles before a sensor alert re-arms
PERSISTENCE = 3

# Forecast horizon for a "predicted crossing" alert, in cycles.
FORECAST_HORIZON = 25

# Deadband. A latched forecast alarm re-arms only once the projection is
# comfortably clear of the horizon, not the moment it ticks past it - otherwise
# ordinary torque variation makes the same alarm chatter for a tool's whole life.
REARM_FACTOR = 1.6

# Window for the robust (median) track.
ROBUST_WINDOW = 5

SECONDS_PER_CYCLE = 120


class AlertKind(StrEnum):
    CROSSED = "crossed"              # a boundary is violated now
    PREDICTED = "predicted"          # projected to cross within the horizon
    APPROACHING = "approaching"      # low margin with a negative slope
    SENSOR_SUSPECT = "sensor"        # input cannot be trusted; not a machine alert
    ABSTAINED = "abstained"          # margin interval straddles zero


@dataclass(frozen=True, slots=True)
class Alert:
    kind: AlertKind
    machine_id: str
    mode: str
    udi: int
    margin: float
    unit: str
    message: str
    lead_time_cycles: float | None = None
    lead_time_min: float | None = None
    interval: tuple[float, float] | None = None
    fix: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "machine": self.machine_id,
            "mode": self.mode,
            "udi": self.udi,
            "margin": round(self.margin, 4),
            "unit": self.unit,
            "message": self.message,
            "lead_time_cycles": (
                round(self.lead_time_cycles, 2) if self.lead_time_cycles is not None else None
            ),
            "lead_time_min": (
                round(self.lead_time_min, 1) if self.lead_time_min is not None else None
            ),
            "interval": [round(v, 2) for v in self.interval] if self.interval else None,
            "fix": self.fix,
        }


@dataclass(frozen=True, slots=True)
class Tick:
    """One scored cycle."""

    udi: int
    machine_id: str
    product_type: str
    point: OperatingPoint | None
    worst_margin: float
    verdict: Verdict
    fired: list[str]
    alerts: list[Alert]
    elapsed_us: float
    #: Normalised distance to each documented limit, from the SAME cycle as
    #: `worst_margin`. The console shows both, and when only the minimum
    #: streamed it had to pair a live headline with distances fetched at load -
    #: two numbers on one screen, computed 9,990 cycles apart, that a reader
    #: would reasonably assume were the same reading. Carrying all three costs
    #: nothing: they are already computed to take the minimum.
    distances: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "udi": self.udi,
            "machine": self.machine_id,
            "type": self.product_type,
            "torque_nm": round(self.point.torque_nm, 2),
            "rpm": round(self.point.rotational_speed_rpm, 1),
            "wear_min": round(self.point.tool_wear_min, 1),
            "power_w": round(self.point.power_w, 1),
            "worst_margin": round(self.worst_margin, 4),
            "distances": {k: round(v, 4) for k, v in self.distances.items()},
            "verdict": self.verdict.value,
            "fired": self.fired,
            "alerts": [a.as_dict() for a in self.alerts],
            "us": round(self.elapsed_us, 2),
        }


_ROW_KEY = {
    "air_temp_k": "air_temperature_k",
    "process_temp_k": "process_temperature_k",
    "rotational_speed_rpm": "rotational_speed_rpm",
    "torque_nm": "torque_nm",
    "tool_wear_min": "tool_wear_min",
}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


@dataclass(slots=True)
class _MachineState:
    """Per-machine memory for the robust track and debouncing."""

    torque: deque[float] = field(default_factory=lambda: deque(maxlen=ROBUST_WINDOW))
    margins: deque[float] = field(default_factory=lambda: deque(maxlen=ROBUST_WINDOW))
    consecutive: dict[str, int] = field(default_factory=dict)
    trust_alerted: bool = False
    trust_clear_run: int = 0
    alerted: set[str] = field(default_factory=set)

    def robust_torque(self, value: float) -> float:
        self.torque.append(value)
        ordered = sorted(self.torque)
        return ordered[len(ordered) // 2]

    def slope(self, margin: float) -> float:
        """Change per cycle over the window. Negative means closing on the boundary."""
        self.margins.append(margin)
        if len(self.margins) < 2:
            return 0.0
        return (self.margins[-1] - self.margins[0]) / (len(self.margins) - 1)


@dataclass(slots=True)
class StreamScorer:
    """Online scorer. Stateless arithmetic plus a small per-machine ring buffer."""

    uncertainty: Uncertainty = DEFAULT_UNCERTAINTY
    #: Interrogate the inputs before computing on them. When None, the scorer
    #: falls back to the static `uncertainty` above - which is what this class
    #: did for its whole life, and why a frozen sensor read as SAFE forever.
    observer: FleetObserver | None = field(default_factory=FleetObserver)
    persistence: int = PERSISTENCE
    horizon: float = FORECAST_HORIZON
    approach_fraction: float = 0.10
    _machines: dict[str, _MachineState] = field(default_factory=dict)
    ticks: int = 0
    alerts_raised: int = 0
    alerts_suppressed: int = 0

    def score(self, row: dict[str, Any]) -> Tick:
        started = time.perf_counter()
        machine = str(row.get("machine_id", "unknown"))
        state = self._machines.setdefault(machine, _MachineState())

        # ── Interrogate before computing.
        #
        # This used to be six bare float() calls. Everything below it - signed
        # margins, interval arithmetic, three-state verdicts, the fail-closed
        # renderer - is rigorous arithmetic, and all of it was being performed
        # on unexamined numbers. The more careful the downstream, the more
        # confidently the system asserted a conclusion drawn from a dead sensor.
        trust = None
        uncertainty = self.uncertainty
        if self.observer is not None:
            trust = self.observer.observe({
                "machine_id": machine,
                "air_temp_k": row.get("air_temperature_k"),
                "process_temp_k": row.get("process_temperature_k"),
                "rotational_speed_rpm": row.get("rotational_speed_rpm"),
                "torque_nm": row.get("torque_nm"),
                "tool_wear_min": row.get("tool_wear_min"),
            })
            # Doubt propagates as covariance: a stale or suspect channel has a
            # large posterior sd, which widens the margin interval, which makes
            # it straddle zero, which yields ABSTAIN. No special-casing.
            uncertainty = trust.uncertainty

        values = {
            name: (trust.channels[name].estimate if trust is not None
                   else _as_float(row.get(_ROW_KEY[name])))
            for name in _ROW_KEY
        }
        if any(v is None for v in values.values()):
            return self._blind_tick(row, machine, trust, started)

        point = OperatingPoint(
            air_temp_k=values["air_temp_k"],
            process_temp_k=values["process_temp_k"],
            rotational_speed_rpm=values["rotational_speed_rpm"],
            torque_nm=values["torque_nm"],
            tool_wear_min=values["tool_wear_min"],
            product_type=row["product_type"],
        )

        # Instantaneous track: exact, stateless, for display.
        margins = evaluate(point)
        fired = margins.fired_modes()

        # Robust track: what alerting is allowed to act on.
        robust_torque = state.robust_torque(point.torque_nm)
        robust_point = OperatingPoint(
            point.air_temp_k,
            point.process_temp_k,
            point.rotational_speed_rpm,
            robust_torque,
            point.tool_wear_min,
            point.product_type,
        )
        robust_margins = evaluate(robust_point)
        if uncertainty.any:
            interval = evaluate_interval(
                air_temp_k=point.air_temp_k,
                process_temp_k=point.process_temp_k,
                rotational_speed_rpm=point.rotational_speed_rpm,
                torque_nm=robust_torque,
                tool_wear_min=point.tool_wear_min,
                product_type=point.product_type,
                uncertainty=uncertainty,
            )
            verdict = interval.verdict()
            firing, abstaining = interval.firing_rules(), interval.abstaining_rules()
        else:
            # Zero uncertainty means the interval is a point, so the verdict is
            # just the rule evaluation. Skipping the Interval objects here is
            # worth ~5x on the hot path and changes no semantics.
            interval = None
            firing = robust_margins.fired_modes()
            abstaining = []
            verdict = Verdict.ALERT if firing else Verdict.SAFE
        distances = {
            "HDF": margins.hdf_distance,
            "PWF": margins.pwf_distance,
            "OSF": margins.osf_distance(point.osf_threshold),
        }
        worst = min(distances.values())
        slope = state.slope(worst)

        if trust is not None and not trust.trusted and not trust.calibrating:
            # The instrument layer says these numbers are not usable. Report the
            # sensor, name the channel, and decline to judge the machine - the
            # distinction that separates a dispatch to the technician from a
            # dispatch to the asset, and the origin of alarm fatigue when a
            # system conflates them.
            self.ticks += 1
            state.trust_clear_run = 0
            # Edge-triggered. A sensor stays broken for thousands of cycles;
            # paging on every one of them is how monitoring systems train
            # operators to ignore them.
            alerts = []
            if not state.trust_alerted:
                state.trust_alerted = True
                alerts.append(Alert(
                    kind=AlertKind.SENSOR_SUSPECT, machine_id=machine,
                    mode=trust.fault_kind.value, udi=int(row["udi"]),
                    margin=worst, unit="", message=trust.explanation,
                    fix="Verify the named channel before acting on this machine.",
                ))
            else:
                self.alerts_suppressed += 1
            self.alerts_raised += len(alerts)
            return Tick(
                udi=int(row["udi"]), machine_id=machine,
                product_type=point.product_type, point=point, worst_margin=worst,
                verdict=Verdict.ABSTAIN, fired=[], alerts=alerts,
                distances=distances,
                elapsed_us=(time.perf_counter() - started) * 1e6,
            )

        # Hysteresis. Without it an intermittent channel re-arms the latch on
        # every good cycle and pages on every bad one - 517 alerts where there
        # was one fault.
        state.trust_clear_run += 1
        if state.trust_clear_run >= TRUST_RECOVERY:
            state.trust_alerted = False
        alerts = self._alerts(
            row, point, robust_point, robust_margins, firing, abstaining, verdict, slope, state
        )
        self.ticks += 1
        self.alerts_raised += len(alerts)

        return Tick(
            udi=int(row["udi"]),
            machine_id=machine,
            product_type=point.product_type,
            point=point,
            worst_margin=worst,
            verdict=verdict,
            fired=fired,
            alerts=alerts,
            distances=distances,
            elapsed_us=(time.perf_counter() - started) * 1e6,
        )

    def _blind_tick(self, row, machine, trust, started) -> Tick:
        """No usable value for at least one channel, so no margin exists.

        The old code raised KeyError or produced float('nan') margins here and
        carried on. Returning an explicit abstention is the only honest option:
        a margin needs a number, and there is not one.
        """
        self.ticks += 1
        message = (trust.explanation if trust is not None
                   else "a required channel is missing from this reading")
        alert = Alert(
            kind=AlertKind.SENSOR_SUSPECT, machine_id=machine,
            mode=(trust.fault_kind.value if trust else "instrument"),
            udi=int(row.get("udi", 0)), margin=float("nan"), unit="",
            message=message,
            fix="No margin can be computed for this cycle. Restore the channel.",
        )
        self.alerts_raised += 1
        return Tick(
            udi=int(row.get("udi", 0)), machine_id=machine,
            product_type=str(row.get("product_type", "?")), point=None,
            worst_margin=float("nan"), verdict=Verdict.ABSTAIN, fired=[],
            alerts=[alert], elapsed_us=(time.perf_counter() - started) * 1e6,
        )

    # -- alerting -----------------------------------------------------------

    def _alerts(
        self, row, point, robust_point, margins, firing, abstaining, verdict, slope,
        state: _MachineState,
    ) -> list[Alert]:
        """`margins` here are the ROBUST margins - the same track that made the
        firing decision. Reporting instantaneous margins alongside a robust
        decision produces alerts that say "boundary crossed" beside a healthy
        number, which is worse than no alert at all."""
        alerts: list[Alert] = []
        machine = str(row.get("machine_id", "unknown"))
        udi = int(row["udi"])

        if verdict is Verdict.ABSTAIN:
            for mode in abstaining:
                # One instrument ticket per machine per mode, not one per sample.
                key = f"sensor:{mode}"
                if key in state.alerted:
                    self.alerts_suppressed += 1
                    continue
                state.alerted.add(key)
                alerts.append(
                    Alert(
                        AlertKind.SENSOR_SUSPECT,
                        machine,
                        mode,
                        udi,
                        0.0,
                        "",
                        f"{mode} cannot be evaluated: the margin interval straddles zero "
                        "given the current input uncertainty. No machine alert is raised; "
                        "check the instrument.",
                    )
                )
            return alerts

        # Crossings, debounced by persistence.
        for mode in firing:
            state.consecutive[mode] = state.consecutive.get(mode, 0) + 1
            if state.consecutive[mode] < self.persistence:
                self.alerts_suppressed += 1
                continue
            if mode in state.alerted:
                continue  # already open; do not re-page
            state.alerted.add(mode)
            margin, unit = _margin_for(mode, margins)
            alerts.append(
                Alert(
                    AlertKind.CROSSED,
                    machine,
                    mode,
                    udi,
                    margin,
                    unit,
                    f"{mode} boundary crossed and held for {self.persistence} cycles.",
                    fix=_fix_for(mode, robust_point),
                )
            )
        for mode in ("HDF", "PWF", "OSF"):
            if mode not in firing:
                state.consecutive[mode] = 0
                state.alerted.discard(mode)
                state.alerted.discard(f"sensor:{mode}")

        # Predicted crossing - the lead time an operator can actually act on.
        # Forecast from the ROBUST point, never a single sample.
        forecast = _forecast_osf(robust_point)
        if forecast is not None and "OSF" not in firing:
            expected, lo, hi = forecast
            if lo < self.horizon and "predicted:OSF" not in state.alerted:
                state.alerted.add("predicted:OSF")
                alerts.append(
                    Alert(
                        AlertKind.PREDICTED,
                        machine,
                        "OSF",
                        udi,
                        margins.overstrain_min_nm,
                        "Δmin·N·m",
                        f"Overstrain projected in {expected:.1f} cycles.",
                        lead_time_cycles=expected,
                        lead_time_min=expected * SECONDS_PER_CYCLE / 60.0,
                        interval=(lo, hi),
                        fix=_fix_for("OSF", robust_point),
                    )
                )

        # Clear latched forecast/approach alarms once the condition lifts, so a
        # tool change re-arms them.
        if forecast is not None and forecast[1] >= self.horizon * REARM_FACTOR:
            state.alerted.discard("predicted:OSF")

        # Approaching: low margin AND closing. Either alone is noise.
        if not alerts and slope < 0:
            worst = min(
                margins.hdf_distance,
                margins.pwf_distance,
                margins.osf_distance(point.osf_threshold),
            )
            if worst >= self.approach_fraction * REARM_FACTOR:
                state.alerted.discard("approaching")
            if 0 < worst < self.approach_fraction and "approaching" not in state.alerted:
                state.alerted.add("approaching")
                alerts.append(
                    Alert(
                        AlertKind.APPROACHING,
                        machine,
                        "any",
                        udi,
                        worst,
                        "ratio",
                        f"Within {worst * 100:.1f}% of a boundary and closing.",
                    )
                )
        return alerts


def _margin_for(mode: str, margins) -> tuple[float, str]:
    return {
        "HDF": (margins.temp_delta_k, "ΔK"),
        "PWF": (min(margins.power_low_w, margins.power_high_w), "ΔW"),
        "OSF": (margins.overstrain_min_nm, "Δmin·N·m"),
    }[mode]


def _fix_for(mode: str, point: OperatingPoint) -> str:
    """A one-line prescription, computed by inverting the constraint."""
    if mode == "OSF" and point.tool_wear_min > 0:
        ceiling = point.osf_threshold * 0.98 / point.tool_wear_min
        return f"reduce torque to <= {ceiling:.1f} N·m, or replace the tool"
    if mode == "PWF":
        return "bring power back inside 3500-9000 W via torque or speed"
    if mode == "HDF":
        return "raise speed above 1380 rpm, or widen the thermal gradient"
    return ""


def _forecast_osf(point: OperatingPoint) -> tuple[float, float, float] | None:
    """Inverse-Gaussian first passage. Closed form: no model, no inference."""
    margin = point.osf_threshold - point.overstrain_min_nm
    drift = point.wear_rate * point.torque_nm
    if margin <= 0 or drift <= 0:
        return None
    mu = margin / drift
    increment_sd = point.wear_rate * 10.0  # torque sd on this process
    lam = (margin**2) / (increment_sd**2)
    sd = math.sqrt(mu**3 / lam) if lam > 0 else 0.0
    return mu, max(0.0, mu - 1.96 * sd), mu + 1.96 * sd


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------

_COLUMNS = (
    "udi, machine_id, product_type, air_temperature_k, process_temperature_k, "
    "rotational_speed_rpm, torque_nm, tool_wear_min, machine_failure"
)


def replay(
    *,
    limit: int | None = None,
    speed: float = 0.0,
    machine: str | None = None,
    scorer: StreamScorer | None = None,
) -> Iterator[Tick]:
    """Replay the warehouse as an event stream.

    `speed` is a multiplier on the 2-minute takt; 0 means as fast as possible.
    Stands in for a Kafka/MQTT consumer - the scorer is unaware of the source.
    """
    scorer = scorer or StreamScorer()
    con = connect()
    where = "TRUE" if machine is None else "machine_id = ?"
    params = [] if machine is None else [machine]
    sql = f"SELECT {_COLUMNS} FROM observations WHERE {where} ORDER BY udi"  # noqa: S608
    if limit:
        sql += f" LIMIT {limit}"

    cur = con.execute(sql, params)
    names = [d[0] for d in cur.description]
    interval = (SECONDS_PER_CYCLE / speed) if speed > 0 else 0.0

    for raw in cur.fetchall():
        tick = scorer.score(dict(zip(names, raw)))
        yield tick
        if interval:
            time.sleep(interval)
