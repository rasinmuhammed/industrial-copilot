"""`counterfactual` - "if torque dropped 5 Nm, what changes?"

Recomputes rule firings over a cohort under a hypothetical parameter change.
Because the constraints are analytic this is exact, not simulated.

The correctness point most implementations get wrong: **base variables are
coupled through the derived ones.** Reducing torque lowers power *and*
overstrain, so it moves a cycle relative to the PWF boundary and the OSF
boundary simultaneously - and can rescue one while breaking the other. We
therefore recompute the full physics from perturbed base variables rather than
editing a stored column.
"""

from __future__ import annotations

from copilot.evidence import EvidenceBundle, Quality, Severity
from copilot.ir import AnalysisPlan, OpName
from copilot.knowledge import metric_index
from copilot.ops.registry import (
    TABLE,
    ExecutionContext,
    cohort_where,
    label_for,
    new_bundle,
    register,
    unit_for,
)
from copilot.physics import BASE_VARIABLES, OperatingPoint, evaluate
from copilot.units import unit as resolve_unit

_FETCH = (
    "air_temperature_k, process_temperature_k, rotational_speed_rpm, "
    "torque_nm, tool_wear_min, product_type, machine_failure"
)


@register(OpName.COUNTERFACTUAL)
def counterfactual(plan: AnalysisPlan, ctx: ExecutionContext) -> EvidenceBundle:
    bundle = new_bundle(plan, ctx)
    changes: dict[str, float] = dict(plan.params["changes"])

    # Reject derived quantities loudly. "Reduce power by 500 W" is ambiguous
    # until the engineer says whether that comes from torque or from speed.
    derived = sorted(set(changes) - BASE_VARIABLES)
    if derived:
        for name in derived:
            bundle.put(f"rejected.{name}", "derived quantity", unit="")
        bundle.put("cf.verdict", None, quality=Quality.ABSTAIN)
        bundle.warn(
            "abstained",
            f"Cannot vary {', '.join(label_for(d) for d in derived)} directly - "
            f"{'it is a' if len(derived) == 1 else 'they are'} derived "
            f"{'quantity' if len(derived) == 1 else 'quantities'}. Vary a base "
            f"variable instead ({', '.join(sorted(BASE_VARIABLES))}); the derived "
            "values follow.",
            severity=Severity.CRITICAL,
        )
        return bundle

    where, params = cohort_where(plan, None)
    cur = ctx.cursor.execute(
        f"SELECT {_FETCH} FROM {TABLE} WHERE {where}", params  # noqa: S608
    )
    rows = cur.fetchall()

    if not rows:
        bundle.put("cf.verdict", None, quality=Quality.ABSTAIN)
        bundle.warn("abstained", "No cycles match this scope.", severity=Severity.CRITICAL)
        return bundle

    # Record the intervention itself, in slots, so the narrator can restate it
    # without retyping a number.
    for name, delta in changes.items():
        unit = unit_for(name)
        bundle.put(
            f"change.{name}",
            delta,
            unit=resolve_unit(unit).as_delta().symbol if unit else "",
            note=f"applied to {label_for(name)}",
        )

    before = {"HDF": 0, "PWF": 0, "OSF": 0, "any": 0}
    after = {"HDF": 0, "PWF": 0, "OSF": 0, "any": 0}
    rescued = broken = 0
    margin_shift = {"power_low": 0.0, "power_high": 0.0, "overstrain": 0.0, "speed": 0.0}

    for air, proc, rpm, torque, wear, ptype, _failed in rows:
        base = OperatingPoint(
            air_temp_k=float(air),
            process_temp_k=float(proc),
            rotational_speed_rpm=float(rpm),
            torque_nm=float(torque),
            tool_wear_min=float(wear),
            product_type=ptype,
        )
        moved = base.perturb(**changes)
        m0, m1 = evaluate(base), evaluate(moved)

        fired0, fired1 = m0.fired_modes(), m1.fired_modes()
        for mode in fired0:
            before[mode] += 1
        for mode in fired1:
            after[mode] += 1
        before["any"] += bool(fired0)
        after["any"] += bool(fired1)
        if fired0 and not fired1:
            rescued += 1
        elif fired1 and not fired0:
            broken += 1

        margin_shift["power_low"] += m1.power_low_w - m0.power_low_w
        margin_shift["power_high"] += m1.power_high_w - m0.power_high_w
        margin_shift["overstrain"] += m1.overstrain_min_nm - m0.overstrain_min_nm
        margin_shift["speed"] += m1.speed_rpm - m0.speed_rpm

    n = len(rows)
    bundle.put("cohort.n", n, unit="count", sig_figs=8)

    for mode in ("HDF", "PWF", "OSF"):
        bundle.put(f"before.{mode}", before[mode], unit="count", sig_figs=8)
        bundle.put(f"after.{mode}", after[mode], unit="count", sig_figs=8)
        bundle.put(f"delta.{mode}", after[mode] - before[mode], unit="count", sig_figs=8)

    bundle.put("before.any_mode", before["any"], unit="count", sig_figs=8)
    bundle.put("after.any_mode", after["any"], unit="count", sig_figs=8)
    bundle.put("delta.any_mode", after["any"] - before["any"], unit="count", sig_figs=8)
    bundle.put("cycles.rescued", rescued, unit="count", sig_figs=8)
    bundle.put("cycles.newly_failing", broken, unit="count", sig_figs=8)

    bundle.put("before.rate", before["any"] / n * 100.0, unit="%", n=n, sig_figs=3)
    bundle.put("after.rate", after["any"] / n * 100.0, unit="%", n=n, sig_figs=3)

    # Mean margin movement, so "it helped" is quantified rather than asserted.
    bundle.put("shift.power_low_margin", margin_shift["power_low"] / n, unit="ΔW")
    bundle.put("shift.power_high_margin", margin_shift["power_high"] / n, unit="ΔW")
    bundle.put("shift.overstrain_margin", margin_shift["overstrain"] / n, unit="Δmin·N·m")

    net = after["any"] - before["any"]
    verdict = "no net change" if net == 0 else ("improvement" if net < 0 else "degradation")
    bundle.put("cf.verdict", verdict, unit="")

    # The trade-off is the interesting part and it is easy to miss.
    if broken and rescued:
        bundle.warn(
            "data_quality",
            f"This change is a trade-off, not a pure win: {rescued} cycle(s) stop "
            f"failing but {broken} start. Base variables are coupled - moving torque "
            "shifts power and overstrain together, so relieving one boundary can "
            "cross another.",
            severity=Severity.WARNING,
            affects=["cycles.rescued", "cycles.newly_failing"],
        )
    elif broken:
        bundle.warn(
            "data_quality",
            f"This change makes things worse: {broken} cycle(s) begin failing and none "
            "are rescued.",
            severity=Severity.CRITICAL,
            affects=["cycles.newly_failing"],
        )

    bundle.warn(
        "data_quality",
        "Counterfactual over the deterministic modes only. TWF is stochastic and RNF "
        "is parameter-independent, so neither responds to a setpoint change.",
        severity=Severity.INFO,
    )

    bundle.provenance = bundle.provenance.model_copy(update={"row_count": n})
    described = ", ".join(
        f"{label_for(k)} {v:+g} {unit_for(k)}".strip() for k, v in changes.items()
    )
    bundle.summary = f"counterfactual ({described}) over {n} cycles: {verdict}"
    return bundle


__all__ = ["counterfactual"]
