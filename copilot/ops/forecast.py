"""`forecast` - when will this cross, not whether it crossed.

Predicting *whether* a cycle failed is arithmetic; we already do it exactly.
Predicting *when* a machine will cross a boundary is a genuine estimation
problem, and it is the one the plant actually cares about - Operon's framing is
that by the time you find the anomaly the batch is already compromised.

Strain is wear x torque. Wear accrues at a documented per-variant rate (2/3/5
min per cycle for L/M/H, verified in the data), and torque is roughly N(40, 10).
Strain is therefore a Wiener process with drift, and time-to-threshold is a
first-passage problem with a closed-form **inverse-Gaussian** solution:

    margin      = threshold - wear x torque
    drift       = wear_rate x torque            [strain gained per cycle]
    E[cycles]   = margin / drift
    lambda      = margin^2 / (wear_rate x sigma_torque)^2
    sd[cycles]  = sqrt(E^3 / lambda)

No training, no inference, no model artifact - a full predictive distribution
from arithmetic. This is the standard PHM formulation (Wiener degradation ->
inverse-Gaussian remaining useful life), applied where it genuinely fits.

TWF is handled differently and deliberately: it is stochastic, so the op returns
a hazard, never a crossing time. Reporting a certainty there would be a lie.
"""

from __future__ import annotations

import math

from copilot.evidence import EvidenceBundle, Interval, Quality, Severity
from copilot.ir import AnalysisPlan, OpName
from copilot.ops.registry import (
    TABLE,
    ExecutionContext,
    cohort_where,
    new_bundle,
    register,
)
from copilot.hazard import fit_hazard
from copilot.physics import (
    OSF_THRESHOLD,
    TWF_WINDOW,
    WEAR_RATE_PER_CYCLE,
    OperatingPoint,
    evaluate,
)
from copilot.stats import _z

# Measured in-window TWF rate: 43 of the 790 rows inside 200-240 min.
TWF_IN_WINDOW_RATE = 43 / 790

# Cycle time under the synthetic takt (see README > Assumptions).
SECONDS_PER_CYCLE = 120


@register(OpName.FORECAST)
def forecast(plan: AnalysisPlan, ctx: ExecutionContext) -> EvidenceBundle:
    bundle = new_bundle(plan, ctx)
    point, sigma_torque = _resolve(plan, ctx, bundle)
    if point is None:
        return bundle

    threshold = OSF_THRESHOLD[point.product_type]
    wear_rate = WEAR_RATE_PER_CYCLE[point.product_type]
    margins = evaluate(point)

    bundle.put("at.product_type", point.product_type, unit="")
    bundle.put("at.tool_wear", point.tool_wear_min, unit="min")
    bundle.put("at.torque", point.torque_nm, unit="N·m")
    bundle.put("at.wear_rate", wear_rate, unit="min", note="wear gained per cycle")
    bundle.put("osf.margin", margins.overstrain_min_nm, unit="Δmin·N·m", sig_figs=6)

    # --- already past the boundary? ---------------------------------------
    fired = margins.fired_modes()
    if "OSF" in fired:
        bundle.put("osf.cycles_to_crossing", None, quality=Quality.ABSTAIN)
        bundle.put("osf.status", "already exceeded", unit="")
        bundle.warn(
            "data_quality",
            "This operating point has already crossed the overstrain limit; there is "
            "nothing left to forecast. Use envelope for the corrective setpoint.",
            severity=Severity.CRITICAL,
        )
    else:
        _osf_first_passage(
            bundle, point, threshold, wear_rate, sigma_torque, plan.confidence
        )

    _twf_hazard(bundle, point, wear_rate, ctx)

    bundle.put("current.fired_modes", ", ".join(fired) if fired else "none", unit="")
    bundle.warn(
        "data_quality",
        "The forecast assumes stationary drift: constant torque distribution and a "
        "constant wear rate. A step change in duty cycle invalidates it, which is "
        "what the knowledge-base calibration monitor exists to catch.",
        severity=Severity.INFO,
    )
    bundle.summary = (
        f"forecast at wear {point.tool_wear_min:.0f} min, "
        f"torque {point.torque_nm:.1f} N·m ({point.product_type})"
    )
    return bundle


def _osf_first_passage(
    bundle: EvidenceBundle,
    point: OperatingPoint,
    threshold: float,
    wear_rate: float,
    sigma_torque: float,
    confidence: float,
) -> None:
    """Inverse-Gaussian first passage to the overstrain limit."""
    margin = threshold - point.overstrain_min_nm
    drift = wear_rate * point.torque_nm  # strain gained per cycle

    if drift <= 0:
        bundle.put("osf.cycles_to_crossing", None, quality=Quality.ABSTAIN)
        bundle.warn(
            "abstained",
            "Torque is zero, so no strain accrues and no crossing occurs.",
            severity=Severity.INFO,
        )
        return

    mu = margin / drift
    # Diffusion comes from torque variability; each cycle's strain increment is
    # wear_rate x torque, so its sd is wear_rate x sigma_torque.
    increment_sd = wear_rate * max(sigma_torque, 1e-9)
    lam = (margin**2) / (increment_sd**2)
    sd = math.sqrt(mu**3 / lam) if lam > 0 else 0.0

    half = _z(confidence) * sd
    lo, hi = max(0.0, mu - half), mu + half

    bundle.put("osf.cycles_to_crossing", mu, unit="count", sig_figs=4)
    bundle.put(
        "osf.cycles_interval",
        mu,
        unit="count",
        ci=Interval(lo=lo, hi=hi),
        sig_figs=4,
        note=f"{int(confidence * 100)}% first-passage interval",
    )
    bundle.put("osf.cycles_lower", lo, unit="count", sig_figs=4)
    bundle.put("osf.cycles_upper", hi, unit="count", sig_figs=4)

    # Lead time in minutes is what an operator can act on.
    bundle.put(
        "osf.lead_time_min",
        mu * SECONDS_PER_CYCLE / 60.0,
        unit="min",
        sig_figs=4,
        note="at the synthetic 2-minute takt",
    )
    bundle.put(
        "osf.crossing_wear",
        threshold / point.torque_nm,
        unit="min",
        sig_figs=5,
        note="wear at which this torque crosses the limit",
    )
    bundle.put("osf.status", "approaching", unit="")

    if mu < 10:
        bundle.warn(
            "data_quality",
            f"Fewer than ten cycles of headroom remain before the overstrain limit. "
            f"The interval is [{lo:.1f}, {hi:.1f}] cycles.",
            severity=Severity.CRITICAL,
            affects=["osf.cycles_to_crossing"],
        )


_HAZARD_CACHE: dict[int, Any] = {}


def _wear_hazard(con):
    """Fit the wear hazard once per connection and keep it.

    Fitting scans the wear column, which is far too much to repeat per question
    and trivial to do once. Keyed on the connection so a different database -
    another plant, or a test fixture - gets its own curve rather than the first
    one that happened to be fitted.
    """
    if con is None:
        return None
    key = id(con)
    if key not in _HAZARD_CACHE:
        rows = con.execute(
            f"SELECT tool_wear_min, TWF FROM {TABLE} WHERE tool_wear_min >= 180"
        ).fetchall()
        _HAZARD_CACHE[key] = fit_hazard(
            [r[0] for r in rows], [bool(r[1]) for r in rows],
            lower=180, upper=260,
        ) if rows else None
    return _HAZARD_CACHE[key]


def _twf_hazard(
    bundle: EvidenceBundle, point: OperatingPoint, wear_rate: float,
    ctx: ExecutionContext,
) -> None:
    """TWF is stochastic. Report a hazard and cycles-to-window, never a crossing."""
    wear = point.tool_wear_min
    start, end = TWF_WINDOW

    if wear >= end:
        bundle.put("twf.status", "past the replacement window", unit="")
        return

    if wear >= start:
        remaining = (end - wear) / wear_rate
        # The hazard is NOT flat across the window. Measured, it climbs from
        # 0.2% below 200 min to 14% above 230 - so a single in-window rate tells
        # an operator at 235 minutes exactly what it tells one at 195, which is
        # the difference between a warning and a shrug.
        #
        # The curve is fitted under a monotonicity constraint, because a tool
        # does not become safer by being used more. See copilot/hazard.py.
        curve = _wear_hazard(ctx.cursor)
        band = curve.at(wear) if curve else None
        rate = band.fitted if band else TWF_IN_WINDOW_RATE
        cumulative = 1.0 - (1.0 - rate) ** max(remaining, 0.0)
        bundle.put("twf.status", "inside the replacement window", unit="")
        bundle.put("twf.per_cycle_probability", rate * 100.0, unit="%", sig_figs=2)
        if band is not None:
            bundle.put("twf.hazard_band_low", band.lower_edge, unit="min", sig_figs=4)
            bundle.put("twf.hazard_band_high", band.upper_edge, unit="min", sig_figs=4)
            bundle.put("twf.hazard_ci_low", band.ci_low * 100.0, unit="%", sig_figs=2)
            bundle.put("twf.hazard_ci_high", band.ci_high * 100.0, unit="%", sig_figs=2)
            bundle.put("twf.hazard_n", band.n, unit="count", sig_figs=8)
            bundle.put("twf.flat_rate_would_be", TWF_IN_WINDOW_RATE * 100.0,
                       unit="%", sig_figs=2)
        bundle.put("twf.cycles_left_in_window", remaining, unit="count", sig_figs=3)
        bundle.put("twf.cumulative_probability", cumulative * 100.0, unit="%", sig_figs=3)
        bundle.warn(
            "data_quality",
            "Tool wear is inside the documented 200-240 min window. TWF is stochastic, "
            "so this is a probability of failure, not a predicted crossing time. No "
            "deterministic forecast is possible for this mode.",
            severity=Severity.WARNING,
            affects=["twf.cumulative_probability"],
        )
        return

    cycles = (start - wear) / wear_rate
    bundle.put("twf.status", "before the replacement window", unit="")
    bundle.put("twf.cycles_to_window", cycles, unit="count", sig_figs=4)
    bundle.put("twf.wear_to_window", start - wear, unit="Δmin", sig_figs=4)
    bundle.put("twf.per_cycle_probability_in_window", TWF_IN_WINDOW_RATE * 100.0,
               unit="%", sig_figs=2)


def _resolve(
    plan: AnalysisPlan, ctx: ExecutionContext, bundle: EvidenceBundle
) -> tuple[OperatingPoint | None, float]:
    """Operating point from explicit params, or from the filtered cohort mean.

    Also returns the torque dispersion, which sets the width of the interval -
    a forecast whose uncertainty came from nowhere is not a forecast.
    """
    params = plan.params or {}
    where, filter_params = cohort_where(plan, None)

    sigma_row = ctx.cursor.execute(
        f"SELECT stddev_samp(torque_nm) FROM {TABLE} WHERE {where}", filter_params  # noqa: S608
    ).fetchone()
    sigma = float(sigma_row[0]) if sigma_row and sigma_row[0] else 10.0

    if "tool_wear_min" in params and "torque_nm" in params:
        ptype = str(params.get("product_type", "L")).upper()
        if ptype not in OSF_THRESHOLD:
            ptype = "L"
        air = float(params.get("air_temp_k", 300.0))
        return (
            OperatingPoint(
                air_temp_k=air,
                process_temp_k=float(params.get("process_temp_k", air + 10.0)),
                rotational_speed_rpm=float(params.get("rotational_speed_rpm", 1500.0)),
                torque_nm=float(params["torque_nm"]),
                tool_wear_min=float(params["tool_wear_min"]),
                product_type=ptype,  # type: ignore[arg-type]
            ),
            sigma,
        )

    row = ctx.cursor.execute(
        "SELECT avg(air_temperature_k), avg(process_temperature_k), "  # noqa: S608
        "avg(rotational_speed_rpm), avg(torque_nm), avg(tool_wear_min), "
        f"any_value(product_type), count(*) FROM {TABLE} WHERE {where}",
        filter_params,
    ).fetchone()

    if not row or not row[6]:
        bundle.put("osf.cycles_to_crossing", None, quality=Quality.ABSTAIN)
        bundle.warn(
            "abstained",
            "No operating point given and no cycles match. Supply tool_wear_min and "
            "torque_nm, or filter to a machine.",
            severity=Severity.CRITICAL,
        )
        return None, sigma

    if int(row[6]) > 1:
        bundle.warn(
            "data_quality",
            f"Forecasting from the MEAN of {int(row[6])} cycles. A fleet-level average "
            "hides per-machine variation; filter to one machine for an actionable "
            "lead time.",
            severity=Severity.WARNING,
        )
    return (
        OperatingPoint(
            air_temp_k=float(row[0]),
            process_temp_k=float(row[1]),
            rotational_speed_rpm=float(row[2]),
            torque_nm=float(row[3]),
            tool_wear_min=float(row[4]),
            product_type=row[5],
        ),
        sigma,
    )


__all__ = ["forecast"]
