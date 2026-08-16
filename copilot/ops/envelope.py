"""`envelope` — the safe operating window, and the minimal change to reach it.

This is the operator a classifier fundamentally cannot provide. Because the
constraints are analytic, we **invert** them: instead of scoring a setpoint an
engineer proposes, we solve for the setpoint that satisfies every constraint.

At fixed tool wear W, thermal gradient D and variant threshold T:

    OSF   wear x torque <= T             ->  torque <= T / W
    PWF   3500 <= torque x omega <= 9000 ->  3500/omega <= torque <= 9000/omega
    HDF   not (D < 8.6 and rpm < 1380)   ->  if D < 8.6, require rpm >= 1380

So the feasible torque band at a given speed is a closed interval, computed
exactly. The boundary this produces is the *true* one, not a decision surface.
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
from copilot.physics import (
    HDF_SPEED_LIMIT,
    HDF_TEMP_LIMIT,
    OSF_THRESHOLD,
    PWF_HIGH,
    PWF_LOW,
    RAD_PER_RPM,
    OperatingPoint,
    evaluate,
)

# Speeds sampled when reporting the envelope as a curve.
_CURVE_POINTS = 12

# A prescription that lands exactly on the limit is operationally useless: the
# next cycle's torque noise puts it back over. Setpoints are moved this far
# INSIDE the boundary by default. Override with params.safety_factor.
DEFAULT_SAFETY_FACTOR = 0.02


@register(OpName.ENVELOPE)
def envelope(plan: AnalysisPlan, ctx: ExecutionContext) -> EvidenceBundle:
    bundle = new_bundle(plan, ctx)
    point = _resolve_point(plan, ctx, bundle)
    if point is None:
        return bundle

    threshold = OSF_THRESHOLD[point.product_type]
    bundle.put("at.product_type", point.product_type, unit="")
    bundle.put("at.tool_wear", point.tool_wear_min, unit="min")
    bundle.put("at.temp_delta", point.temp_delta_k, unit="ΔK")
    bundle.put("at.rotational_speed", point.rotational_speed_rpm, unit="rpm")
    bundle.put("at.torque", point.torque_nm, unit="N·m")
    bundle.put("at.power", point.power_w, unit="W")
    bundle.put("osf.threshold", threshold, unit="min·N·m", sig_figs=6)

    # --- current state ----------------------------------------------------
    margins = evaluate(point)
    fired = margins.fired_modes()
    bundle.put("current.fired", ", ".join(fired) if fired else "none", unit="")
    bundle.put("current.safe", "no" if fired else "yes", unit="")

    # --- HDF: is speed constrained at all? --------------------------------
    hdf_binds = point.temp_delta_k < HDF_TEMP_LIMIT
    bundle.put("hdf.constrains_speed", "yes" if hdf_binds else "no", unit="")
    if hdf_binds:
        bundle.put("hdf.min_safe_speed", HDF_SPEED_LIMIT, unit="rpm", sig_figs=6)
        bundle.warn(
            "data_quality",
            f"The thermal gradient is below {HDF_TEMP_LIMIT} K, so heat dissipation "
            f"constrains speed: the cycle must run at or above {HDF_SPEED_LIMIT:.0f} rpm. "
            "Widening the gradient removes the constraint entirely.",
            severity=Severity.WARNING,
            affects=["hdf.min_safe_speed"],
        )

    # --- torque band at the current speed ---------------------------------
    band = _torque_band(point.rotational_speed_rpm, point.tool_wear_min, threshold)
    if band is None:
        bundle.put("safe.torque_min", None, quality=Quality.ABSTAIN)
        bundle.put("safe.torque_max", None, quality=Quality.ABSTAIN)
        bundle.warn(
            "abstained",
            "No torque satisfies every constraint at this speed and wear. The tool "
            "must be replaced, or the speed changed, before a safe setpoint exists.",
            severity=Severity.CRITICAL,
        )
    else:
        bundle.put("safe.torque_min", band.lo, unit="N·m")
        bundle.put("safe.torque_max", band.hi, unit="N·m")
        bundle.put("safe.torque_width", band.width, unit="ΔN·m")
        bundle.put(
            "safe.binding_ceiling",
            "overstrain" if threshold / max(point.tool_wear_min, 1e-9) < PWF_HIGH / point.omega_rad_s
            else "power overload",
            unit="",
        )

    # --- the envelope as a curve, for plotting ----------------------------
    _emit_curve(bundle, point, threshold)

    # --- prescription: minimal single-variable change ---------------------
    if fired:
        safety = float((plan.params or {}).get("safety_factor", DEFAULT_SAFETY_FACTOR))
        safety = min(max(safety, 0.0), 0.5)
        _prescribe(bundle, point, threshold, safety)

    bundle.summary = (
        f"operating envelope at wear {point.tool_wear_min:.0f} min, "
        f"{point.rotational_speed_rpm:.0f} rpm ({point.product_type} variant)"
    )
    return bundle


def _torque_band(rpm: float, wear: float, threshold: float) -> Interval | None:
    """Closed-form feasible torque interval at a fixed speed and wear."""
    omega = rpm * RAD_PER_RPM
    if omega <= 0:
        return None
    lo = PWF_LOW / omega                      # stall floor
    hi = PWF_HIGH / omega                     # overload ceiling
    if wear > 0:
        hi = min(hi, threshold / wear)        # overstrain ceiling
    if lo > hi:
        return None
    return Interval(lo=lo, hi=hi)


def _emit_curve(bundle: EvidenceBundle, point: OperatingPoint, threshold: float) -> None:
    """Sample the boundary so a UI can draw the true feasible region."""
    hdf_binds = point.temp_delta_k < HDF_TEMP_LIMIT
    lo_speed = HDF_SPEED_LIMIT if hdf_binds else 1200.0
    hi_speed = 2800.0
    step = (hi_speed - lo_speed) / (_CURVE_POINTS - 1)

    feasible = 0
    for i in range(_CURVE_POINTS):
        rpm = lo_speed + i * step
        band = _torque_band(rpm, point.tool_wear_min, threshold)
        key = f"curve.{int(round(rpm))}rpm"
        if band is None:
            bundle.put(f"{key}.torque_min", None, quality=Quality.ABSTAIN)
            continue
        feasible += 1
        bundle.put(f"{key}.torque_min", band.lo, unit="N·m", sig_figs=4)
        bundle.put(f"{key}.torque_max", band.hi, unit="N·m", sig_figs=4)
    bundle.put("curve.feasible_speeds", feasible, unit="count", sig_figs=8)
    bundle.put("curve.sampled_speeds", _CURVE_POINTS, unit="count", sig_figs=8)


def _prescribe(
    bundle: EvidenceBundle, point: OperatingPoint, threshold: float, safety: float
) -> None:
    """Solve for the minimal single-variable change restoring every margin.

    Candidates are evaluated by *magnitude of intervention*, and each is verified
    by re-running the full physics — a prescription that fixes one boundary while
    crossing another is not a prescription.

    Every target is placed `safety` inside the boundary rather than on it. A
    setpoint sitting exactly at the limit is put back over it by the next
    cycle's torque noise, which makes it worthless advice.
    """
    candidates: list[tuple[str, str, float, str, OperatingPoint]] = []
    bundle.put("fix.safety_factor", safety * 100.0, unit="%", sig_figs=3,
               note="target placed this far inside the boundary")

    # Torque down to the overstrain ceiling.
    if point.tool_wear_min > 0:
        ceiling = threshold * (1.0 - safety) / point.tool_wear_min
        if point.torque_nm > ceiling:
            delta = ceiling - point.torque_nm
            candidates.append(
                ("torque_nm", "reduce torque", delta, "N·m",
                 point.perturb(torque_nm=delta))
            )

    # Torque into the power band.
    omega = point.omega_rad_s
    if omega > 0:
        if point.power_w > PWF_HIGH:
            delta = PWF_HIGH * (1.0 - safety) / omega - point.torque_nm
            candidates.append(
                ("torque_nm", "reduce torque", delta, "N·m",
                 point.perturb(torque_nm=delta))
            )
        elif point.power_w < PWF_LOW:
            delta = PWF_LOW * (1.0 + safety) / omega - point.torque_nm
            candidates.append(
                ("torque_nm", "increase torque", delta, "N·m",
                 point.perturb(torque_nm=delta))
            )

    # Speed above the HDF limit.
    if point.temp_delta_k < HDF_TEMP_LIMIT and point.rotational_speed_rpm < HDF_SPEED_LIMIT:
        delta = HDF_SPEED_LIMIT * (1.0 + safety) - point.rotational_speed_rpm
        candidates.append(
            ("rotational_speed_rpm", "increase speed", delta, "rpm",
             point.perturb(rotational_speed_rpm=delta))
        )

    # Tool replacement resets wear entirely — always available, never minimal.
    candidates.append(
        ("tool_wear_min", "replace the tool", -point.tool_wear_min, "min",
         point.perturb(tool_wear_min=-point.tool_wear_min))
    )

    # Keep only interventions that actually clear every boundary.
    verified = [c for c in candidates if not evaluate(c[4]).fired_modes()]
    if not verified:
        bundle.put("fix.available", "no", unit="")
        bundle.warn(
            "abstained",
            "No single-variable change restores every margin at this operating point. "
            "A combined change is required — for example replacing the tool and "
            "reducing torque together.",
            severity=Severity.CRITICAL,
        )
        return

    verified.sort(key=lambda c: abs(c[2]))
    field, action, delta, unit, result = verified[0]
    after = evaluate(result)

    bundle.put("fix.available", "yes", unit="")
    bundle.put("fix.action", action, unit="")
    bundle.put("fix.variable", field, unit="")
    bundle.put("fix.delta", delta, unit=f"Δ{unit}" if unit != "min" else "Δmin")
    bundle.put("fix.new_value", getattr(result, field), unit=unit)
    bundle.put("fix.resulting_power", result.power_w, unit="W")
    bundle.put("fix.resulting_overstrain_margin", after.overstrain_min_nm, unit="Δmin·N·m")
    bundle.put("fix.alternatives", len(verified) - 1, unit="count", sig_figs=8)

    if len(verified) > 1:
        alt_field, alt_action, alt_delta, alt_unit, _ = verified[1]
        bundle.put("fix.alternative_action", alt_action, unit="")
        bundle.put(
            "fix.alternative_delta",
            alt_delta,
            unit=f"Δ{alt_unit}" if alt_unit != "min" else "Δmin",
        )


def _resolve_point(
    plan: AnalysisPlan, ctx: ExecutionContext, bundle: EvidenceBundle
) -> OperatingPoint | None:
    """Take the operating point from explicit params, or from the filtered cohort.

    An explicit point lets an engineer ask about a hypothetical setpoint that has
    never been run; a filtered cohort answers about real observed conditions.
    """
    params = plan.params or {}
    explicit = {
        k: params[k]
        for k in (
            "air_temp_k",
            "process_temp_k",
            "rotational_speed_rpm",
            "torque_nm",
            "tool_wear_min",
        )
        if k in params
    }

    if len(explicit) >= 3:
        ptype = str(params.get("product_type", "L")).upper()
        if ptype not in OSF_THRESHOLD:
            bundle.warn(
                "data_quality",
                f"Unknown product variant {ptype!r}; defaulting to L (the strictest "
                "overstrain limit).",
                severity=Severity.INFO,
            )
            ptype = "L"
        air = float(explicit.get("air_temp_k", 300.0))
        return OperatingPoint(
            air_temp_k=air,
            process_temp_k=float(explicit.get("process_temp_k", air + 10.0)),
            rotational_speed_rpm=float(explicit.get("rotational_speed_rpm", 1500.0)),
            torque_nm=float(explicit.get("torque_nm", 40.0)),
            tool_wear_min=float(explicit.get("tool_wear_min", 0.0)),
            product_type=ptype,  # type: ignore[arg-type]
        )

    where, filter_params = cohort_where(plan, None)
    row = ctx.cursor.execute(
        "SELECT avg(air_temperature_k), avg(process_temperature_k), "  # noqa: S608
        "avg(rotational_speed_rpm), avg(torque_nm), avg(tool_wear_min), "
        f"any_value(product_type), count(*) FROM {TABLE} WHERE {where}",
        filter_params,
    ).fetchone()

    if not row or not row[6]:
        bundle.put("current.safe", None, quality=Quality.ABSTAIN)
        bundle.warn(
            "abstained",
            "No operating point given and no cycles match, so there is no envelope "
            "to report. Supply params such as tool_wear_min and rotational_speed_rpm.",
            severity=Severity.CRITICAL,
        )
        return None

    if int(row[6]) > 1:
        bundle.warn(
            "data_quality",
            f"No explicit setpoint given, so the envelope is reported at the MEAN "
            f"conditions of {int(row[6])} matching cycles. Individual cycles vary "
            "around this point.",
            severity=Severity.INFO,
        )
    return OperatingPoint(
        air_temp_k=float(row[0]),
        process_temp_k=float(row[1]),
        rotational_speed_rpm=float(row[2]),
        torque_nm=float(row[3]),
        tool_wear_min=float(row[4]),
        product_type=row[5],
    )


__all__ = ["envelope"]
