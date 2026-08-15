"""Gate 2 — is the instrument telling the truth?

The nastiest documented drift property is that *"the sensor may still produce
values that look normal, but the measurement slowly becomes less trustworthy."*
A margin engine reads sensors directly, so a drifting sensor silently moves
every margin with it.

The discriminator is physics that must hold regardless of operating point. An
invariant break means **instrumentation**. Invariants holding while margins
shift means **operations**.

Measured on this dataset by injection:

    scenario                    HDF alerts   z(dT)    z(rpm)   verdict
    baseline                        115        0.1      0.0    ok
    air sensor drifts -0.4 K         53       40.0      0.0    SENSOR
    process genuinely slows         188        0.1    -22.3    PROCESS

Read the second row. A 0.4 K thermocouple drift makes heat-dissipation alerts
drop 54%. A conventional copilot reports "failures down 54%, good month" while
the plant goes blind to the mode it believes it solved — a safety incident
dressed as a KPI improvement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import duckdb

__all__ = [
    "DriftVerdict",
    "InvariantResult",
    "DriftReport",
    "check_invariants",
    "diagnose_drift",
]

TABLE = "observations"

# Reference values, established on the published data. See docs/01-DATASET.md §9.
REFERENCE_TEMP_DELTA_MEAN = 10.0
REFERENCE_TEMP_DELTA_SD = 1.0
REFERENCE_RPM_TORQUE_CORR = -0.875

# Standard deviations of the mean beyond which a shift is not chance.
Z_THRESHOLD = 5.0


class DriftVerdict(StrEnum):
    """Distinct from reliability.Verdict, which is an ALERT/SAFE/ABSTAIN state.

    This answers a different question: not "should we alert?" but "is the change
    we are seeing coming from the instrument or from the plant?"
    """

    OK = "ok"
    SENSOR = "sensor"
    PROCESS = "process"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class InvariantResult:
    code: str
    description: str
    holds: bool
    observed: float
    expected: float
    detail: str = ""


@dataclass(frozen=True, slots=True)
class DriftReport:
    verdict: DriftVerdict
    invariants: list[InvariantResult]
    z_temp_delta: float
    z_rotational_speed: float
    z_torque: float
    explanation: str

    @property
    def violations(self) -> list[InvariantResult]:
        return [i for i in self.invariants if not i.holds]


def check_invariants(
    con: duckdb.DuckDBPyConnection,
    *,
    where: str = "TRUE",
    params: list | None = None,
    table: str = TABLE,
) -> list[InvariantResult]:
    """Evaluate the physics that must hold at any operating point."""
    params = params or []
    row = con.execute(
        f"""SELECT
              sum(CASE WHEN process_temperature_k <= air_temperature_k THEN 1 ELSE 0 END),
              avg(temp_delta_k), stddev_samp(temp_delta_k),
              corr(rotational_speed_rpm, torque_nm),
              avg(power_w), count(*)
            FROM {table} WHERE {where}""",  # noqa: S608
        params,
    ).fetchone()
    thermo_violations, dt_mean, dt_sd, corr, mean_power, n = row

    return [
        InvariantResult(
            "I1",
            "process temperature exceeds air temperature",
            int(thermo_violations or 0) == 0,
            float(thermo_violations or 0),
            0.0,
            "thermodynamically required; a violation is a wiring or calibration fault",
        ),
        InvariantResult(
            "I2a",
            "temperature differential centres on its design value",
            abs(float(dt_mean) - REFERENCE_TEMP_DELTA_MEAN) < 0.25,
            float(dt_mean),
            REFERENCE_TEMP_DELTA_MEAN,
            "a shifted mean with stable dispersion indicates a temperature sensor offset",
        ),
        InvariantResult(
            "I2b",
            "temperature differential dispersion is stable",
            abs(float(dt_sd) - REFERENCE_TEMP_DELTA_SD) < 0.25,
            float(dt_sd),
            REFERENCE_TEMP_DELTA_SD,
            "growing dispersion indicates a noisy or failing sensor",
        ),
        InvariantResult(
            "I3",
            "speed and torque remain inversely coupled",
            abs(float(corr) - REFERENCE_RPM_TORQUE_CORR) < 0.10,
            float(corr),
            REFERENCE_RPM_TORQUE_CORR,
            "the drive holds a power target; broken coupling means a control-loop change",
        ),
        InvariantResult(
            "I4",
            "mean mechanical power is stable",
            n == 0 or abs(float(mean_power) - 6280.0) / 6280.0 < 0.10,
            float(mean_power or 0.0),
            6280.0,
            "a shift with intact coupling indicates a genuine change in duty",
        ),
    ]


def diagnose_drift(
    con: duckdb.DuckDBPyConnection,
    *,
    window_where: str,
    window_params: list | None = None,
    baseline_where: str = "TRUE",
    baseline_params: list | None = None,
    table: str = TABLE,
    baseline_table: str | None = None,
) -> DriftReport:
    """Separate a drifting instrument from a changing process.

    Compares a recent window against a baseline. The key asymmetry: a sensor
    fault moves *its own* signal while leaving unrelated signals alone, whereas
    a process change moves the signals the process actually controls.
    """
    # The baseline may live in a different table — in production, a stored
    # reference period rather than a slice of the live one.
    stats = _window_stats(con, window_where, window_params or [], table)
    base = _window_stats(con, baseline_where, baseline_params or [], baseline_table or table)

    z_dt = _z_of_mean(stats["dt_mean"], base["dt_mean"], base["dt_sd"], stats["n"])
    z_rpm = _z_of_mean(stats["rpm_mean"], base["rpm_mean"], base["rpm_sd"], stats["n"])
    z_tq = _z_of_mean(stats["tq_mean"], base["tq_mean"], base["tq_sd"], stats["n"])

    invariants = check_invariants(con, where=window_where, params=window_params or [], table=table)
    broken = [i for i in invariants if not i.holds]

    # A temperature-differential shift with no movement in the mechanical
    # signals cannot be the process: the process does not control ambient.
    thermal_only = abs(z_dt) > Z_THRESHOLD and abs(z_rpm) < Z_THRESHOLD and abs(z_tq) < Z_THRESHOLD
    mechanical = abs(z_rpm) > Z_THRESHOLD or abs(z_tq) > Z_THRESHOLD

    if thermal_only or (broken and not mechanical):
        verdict = DriftVerdict.SENSOR
        explanation = (
            "The temperature differential has shifted while speed and torque are "
            "unchanged. The process does not control ambient temperature, so this "
            "is an instrument fault, not a change in operations. Any change in "
            "heat-dissipation alerts is an artefact — investigate the thermocouple "
            "before acting on the alert counts."
        )
    elif mechanical and thermal_only:
        verdict = DriftVerdict.BOTH
        explanation = "Both the thermal and mechanical signals have moved; treat separately."
    elif mechanical:
        verdict = DriftVerdict.PROCESS
        explanation = (
            "Speed or torque has moved while the physics invariants still hold. "
            "This is a genuine change in operations, and any change in alert rates "
            "reflects the plant rather than the instruments."
        )
    else:
        verdict = DriftVerdict.OK
        explanation = "No significant drift in either the instruments or the process."

    return DriftReport(
        verdict=verdict,
        invariants=invariants,
        z_temp_delta=z_dt,
        z_rotational_speed=z_rpm,
        z_torque=z_tq,
        explanation=explanation,
    )


def _window_stats(
    con: duckdb.DuckDBPyConnection, where: str, params: list, table: str = TABLE
) -> dict[str, float]:
    row = con.execute(
        f"""SELECT avg(temp_delta_k), stddev_samp(temp_delta_k),
                   avg(rotational_speed_rpm), stddev_samp(rotational_speed_rpm),
                   avg(torque_nm), stddev_samp(torque_nm), count(*)
            FROM {table} WHERE {where}""",  # noqa: S608
        params,
    ).fetchone()
    return {
        "dt_mean": float(row[0] or 0.0),
        "dt_sd": float(row[1] or 1.0),
        "rpm_mean": float(row[2] or 0.0),
        "rpm_sd": float(row[3] or 1.0),
        "tq_mean": float(row[4] or 0.0),
        "tq_sd": float(row[5] or 1.0),
        "n": int(row[6] or 0),
    }


def _z_of_mean(observed: float, baseline: float, baseline_sd: float, n: int) -> float:
    """How many standard errors the window mean sits from the baseline mean."""
    if n <= 0 or baseline_sd <= 0:
        return 0.0
    return (observed - baseline) / (baseline_sd / math.sqrt(n))
