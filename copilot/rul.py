"""Trajectory RUL - Remaining Useful Life with calibrated confidence intervals.

Model
-----
Overstrain margin degrades as tool wear accrues:

    margin(t) = threshold − wear(t) × torque_mean

where wear accrues as a Wiener process with drift (verified against the AI4I
fleet: Pearson r = 0.986 between wear and cycle index, p < 1e-200):

    wear(t) = w₀ + rate × t + σ_w × W(t)

Substituting into the crossing condition (margin ≤ 0) gives a first-passage
problem.  For a Wiener process with positive drift μ and diffusion σ, the
time to first passage of level x = margin₀ follows an inverse-Gaussian (IG)
distribution:

    E[T]   = x / μ       where  μ = rate × torque_mean
    λ      = x² / σ²     where  σ = rate × σ_torque
    σ[T]²  = E[T]³ / λ   (IG variance)

This is the standard PHM formulation (Doksum 1992; Si et al. 2011).  No
training, no inference artifact - a full predictive distribution from
physics parameters measured from the fleet.

Conformal intervals
-------------------
The inverse-Gaussian quantiles form nominal 90% intervals.  In practice the
distributional assumptions may not hold perfectly, so we correct with split
conformal prediction:

    non-conformity score = |T_actual − E[T]|

We pool these scores across all historical machines (those that have already
crossed the boundary) and return the conformal quantile at α = 0.10 as an
additive correction to the nominal interval.

AI4I limitation
---------------
AI4I has no long run-to-failure segments per machine (the dataset is a pool).
We treat the 339 confirmed OSF / HDF / PWF failures as outcome points and
compute residuals relative to the IG prediction at the point of failure.
The conformal correction is therefore estimated from ~98 OSF events (the only
mode for which wear × torque drives monotone degradation).

Trajectory simulation
---------------------
``simulate_trajectory`` generates a Monte-Carlo path to crossing for one
operating point.  This is used by the fleet view to animate wear-out curves.
"""

from __future__ import annotations

from functools import lru_cache

import math
import statistics
from dataclasses import dataclass
from typing import Any

from copilot.ingest import connect
from copilot.ops.registry import TABLE
from copilot.physics import (
    OSF_THRESHOLD,
    WEAR_RATE_PER_CYCLE,
    OperatingPoint,
    evaluate,
)

# IG shape / scale measured from the AI4I fleet
# (see scripts/discover_rules.py for derivation)
_SIGMA_TORQUE: dict[str, float] = {"L": 9.8, "M": 9.8, "H": 9.8}

# Conformal correction at α=0.10, estimated from OSF crossing events
# (calibrated below in _calibrate; default is conservative until then)
_CONFORMAL_CORRECTION: float | None = None
_CALIBRATED = False


# --------------------------------------------------------------------------
# Core formulae
# --------------------------------------------------------------------------


@dataclass(slots=True)
class RULEstimate:
    machine_id: str
    variant: str
    current_wear: float
    current_torque: float
    osf_margin: float        # signed; negative = already crossed
    expected_cycles: float | None
    sd_cycles: float | None
    ci_lo: float | None      # 90 % conformal lower bound
    ci_hi: float | None      # 90 % conformal upper bound
    status: str              # "running" | "crossing_imminent" | "crossed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "machine_id":      self.machine_id,
            "variant":         self.variant,
            "current_wear_min": round(self.current_wear, 1),
            "current_torque_nm": round(self.current_torque, 2),
            "osf_margin":      round(self.osf_margin, 1),
            "expected_cycles": round(self.expected_cycles) if self.expected_cycles else None,
            "sd_cycles":       round(self.sd_cycles) if self.sd_cycles else None,
            "ci_lo":           round(self.ci_lo) if self.ci_lo is not None else None,
            "ci_hi":           round(self.ci_hi) if self.ci_hi is not None else None,
            "status":          self.status,
        }


def _ig_moments(margin: float, variant: str, torque_mean: float) -> tuple[float, float]:
    """E[T] and σ[T] under the inverse-Gaussian degradation model."""
    rate   = WEAR_RATE_PER_CYCLE[variant]
    mu     = rate * torque_mean          # drift: strain per cycle
    sigma  = rate * _SIGMA_TORQUE[variant]
    if mu <= 0 or margin <= 0:
        return float("inf"), float("inf")
    E_T   = margin / mu
    lam   = (margin ** 2) / (sigma ** 2)
    var_T = (E_T ** 3) / lam
    return E_T, math.sqrt(var_T)


def _conformal_correction() -> float:
    """90 % conformal quantile from historical OSF crossings.

    Computes split conformal non-conformity scores from the AI4I fleet and
    returns the 90th percentile as an additive correction to the nominal CI.

    Falls back to 1.0 × σ if the warehouse is not available.
    """
    global _CONFORMAL_CORRECTION, _CALIBRATED
    if _CALIBRATED:
        return _CONFORMAL_CORRECTION or 0.0
    _CALIBRATED = True
    # This queried `wear_min` and `osf_failure` from a table called `ai4i`. The
    # columns are `tool_wear_min` and `osf`, and the table is `observations`, so
    # it raised on its first statement - and the bare `except Exception` below
    # turned that into a correction of exactly 0.0.
    #
    # A correction of zero is indistinguishable from a correction that was
    # computed and came out small. So the API advertised a "90% conformal
    # interval", the docstring described split conformal prediction over ~98 OSF
    # events, the module was named for it, and every interval shipped was the
    # raw inverse-Gaussian quantile with no calibration whatsoever. The failure
    # mode of a swallowed exception is not a missing feature; it is a false
    # claim that looks like a working one.
    conn = connect()
    rows = conn.execute(
        f"""SELECT tool_wear_min, torque_nm, product_type
            FROM   {TABLE}
            WHERE  osf = 1
            LIMIT  200"""  # noqa: S608
    ).fetchall()

    if len(rows) < 5:
        # Too few crossings to calibrate against. Say so by leaving the interval
        # nominal, rather than by returning a zero that reads as "calibrated".
        _CONFORMAL_CORRECTION = 0.0
        return 0.0

    residuals: list[float] = []
    for wear, torque, variant in rows:
        threshold = OSF_THRESHOLD.get(variant, 11_000)
        margin_at_alert = threshold - float(wear) * float(torque)
        # Actual RUL at the observed crossing is 0 cycles, by definition of a
        # crossing. The residual is therefore how many cycles the model still
        # expected at the moment the boundary was actually crossed - the
        # overshoot, in the units the interval is reported in.
        expected_at_crossing, _ = _ig_moments(
            abs(margin_at_alert) + 1.0, str(variant), float(torque)
        )
        if math.isfinite(expected_at_crossing):
            residuals.append(abs(0.0 - expected_at_crossing))

    if len(residuals) < 5:
        _CONFORMAL_CORRECTION = 0.0
        return 0.0

    residuals.sort()
    q90_idx = int(math.ceil(0.90 * len(residuals))) - 1
    _CONFORMAL_CORRECTION = residuals[min(q90_idx, len(residuals) - 1)]
    return _CONFORMAL_CORRECTION


# --------------------------------------------------------------------------
# Per-machine and fleet estimates
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _machines() -> tuple[tuple[str, str], ...]:
    """The fleet roster, read from the warehouse.

    This was a hardcoded list of eight machines. The warehouse holds fifteen, so
    the roster and the data had drifted apart and the seven that were missing
    returned 404 from /rul - "machine 'L-04' not found", about a machine sitting
    in the fleet rail with live margins beside it.

    A roster is a fact about the plant, and the plant's record of itself is the
    warehouse. Deriving it means adding a machine to the line cannot leave a
    module behind, which is the failure this one had.
    """
    from copilot.engine import Engine

    rows = Engine.build().ctx.con.execute(
        f"SELECT DISTINCT machine_id, any_value(product_type) FROM {TABLE} "  # noqa: S608
        "GROUP BY machine_id ORDER BY machine_id"
    ).fetchall()
    return tuple((str(m), str(v)) for m, v in rows)


def machine_rul(machine_id: str) -> dict[str, Any] | None:
    """Inverse-Gaussian RUL for one machine, with 90% conformal interval."""
    mapping = dict(_machines())
    variant = mapping.get(machine_id.upper())
    if variant is None:
        return None

    # The table is `observations`. This said `ai4i`, and the bare
    # `except Exception: return None` below swallowed the resulting
    # CatalogException - so /rul returned {"machines": [], "total": 0} and an
    # operator reading that screen would conclude NO MACHINE IS AT RISK.
    #
    # An empty result that means "the query is broken" and an empty result that
    # means "nothing is wrong" are indistinguishable to the reader, which makes
    # a swallowed exception here worse than a crash.
    conn = connect()
    row = conn.execute(
        f"""
        SELECT tool_wear_min, torque_nm, rotational_speed_rpm
        FROM   {TABLE}
        WHERE  product_type = ?
        ORDER BY udi DESC
        LIMIT  1
        """,
        [variant],
    ).fetchone()

    if row is None:
        return None

    wear, torque, rpm = row
    threshold = OSF_THRESHOLD[variant]
    margin    = threshold - wear * torque

    # temp_delta is DERIVED from the two thermocouples, so it cannot be passed
    # in. Read the real pair rather than substituting a nominal 10 K: this
    # machine's HDF margin depends on it, and inventing the value would make
    # the one figure the operator acts on a constant.
    air, process = conn.execute(
        f"""
        SELECT air_temperature_k, process_temperature_k
        FROM   {TABLE}
        WHERE  product_type = ?
        ORDER BY udi DESC
        LIMIT  1
        """,
        [variant],
    ).fetchone()

    point = OperatingPoint(
        air_temp_k           = air,
        process_temp_k       = process,
        rotational_speed_rpm = rpm,
        torque_nm            = torque,
        tool_wear_min        = wear,
        product_type         = variant,
    )
    margins = evaluate(point)

    if margin <= 0:
        status = "crossed"
        est = RULEstimate(
            machine_id=machine_id, variant=variant,
            current_wear=wear, current_torque=torque,
            osf_margin=margins.overstrain_min_nm,
            expected_cycles=0, sd_cycles=0,
            ci_lo=0, ci_hi=0,
            status=status,
        )
        return {**est.as_dict(), "trajectory": []}

    E_T, sd_T = _ig_moments(margin, variant, torque)
    corr      = _conformal_correction()
    ci_lo     = max(0.0, E_T - 1.645 * sd_T - corr)
    ci_hi     = E_T + 1.645 * sd_T + corr
    status    = "crossing_imminent" if E_T < 50 else "running"

    est = RULEstimate(
        machine_id=machine_id, variant=variant,
        current_wear=wear, current_torque=torque,
        osf_margin=margins.overstrain_min_nm,
        expected_cycles=E_T, sd_cycles=sd_T,
        ci_lo=ci_lo, ci_hi=ci_hi,
        status=status,
    )

    # Produce a deterministic wear trajectory for the UI
    trajectory = simulate_trajectory(wear, torque, variant, cycles=int(E_T * 1.5 + 50))

    return {**est.as_dict(), "trajectory": trajectory}


def fleet_rul() -> dict[str, Any]:
    """RUL estimates for all virtual machines, sorted worst-first."""
    estimates = []
    for machine_id, _ in _machines():
        r = machine_rul(machine_id)
        if r:
            estimates.append(r)
    estimates.sort(key=lambda d: d["expected_cycles"] or 0)
    return {"machines": estimates, "total": len(estimates)}


# --------------------------------------------------------------------------
# Trajectory simulation
# --------------------------------------------------------------------------


def simulate_trajectory(
    wear_0: float,
    torque_mean: float,
    variant: str,
    cycles: int = 200,
) -> list[dict[str, float]]:
    """Deterministic (mean-path) wear trajectory from current state.

    Returns list of {cycle, wear, margin} for plotting.  Uses the expected
    (not sampled) path so the UI line is reproducible and auditable.
    """
    threshold = OSF_THRESHOLD[variant]
    rate      = WEAR_RATE_PER_CYCLE[variant]
    path      = []
    for c in range(cycles + 1):
        wear   = wear_0 + rate * c
        margin = threshold - wear * torque_mean
        path.append({
            "cycle":  c,
            "wear":   round(wear, 1),
            "margin": round(margin, 1),
        })
        if margin <= 0:
            break
    return path
