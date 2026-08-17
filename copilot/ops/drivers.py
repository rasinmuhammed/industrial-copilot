"""`drivers` - rank which variables separate failures from healthy operation.

Deliberately an *associational* op, and its wording is constrained to say so.
Ranking by separation is not attribution: on this dataset rpm and torque
correlate at -0.875, so a naive ranking will credit rotational speed for an
effect that torque is driving. Use `root_cause` for attribution, which computes
the mechanism instead of ranking correlates.

Two things this op does that a SHAP bar chart does not:

  * flags collinear pairs among the top drivers, so a reader cannot mistake a
    confounded ranking for a causal one;
  * reports each driver's separation with a confidence interval, so a small
    cohort cannot masquerade as a strong signal.
"""

from __future__ import annotations

from copilot.evidence import EvidenceBundle, Quality, Severity
from copilot.ir import AnalysisPlan, OpName
from copilot.ops.compare import COLLINEARITY_THRESHOLD
from copilot.ops.registry import (
    TABLE,
    ExecutionContext,
    cohort_where,
    column_for,
    label_for,
    new_bundle,
    register,
    unit_for,
)
from copilot.stats import MIN_REPORTABLE_N, cohens_d, cohens_d_interval

# Sensor and derived-physics metrics. Margins are excluded from the default set:
# they are algebraic restatements of the rules, so they would trivially top the
# ranking and tell the reader nothing they did not already know.
DEFAULT_DRIVERS = (
    "torque_nm",
    "rotational_speed_rpm",
    "tool_wear_min",
    "temp_delta_k",
    "power_w",
    "overstrain_min_nm",
    "air_temp_k",
    "process_temp_k",
)


@register(OpName.DRIVERS)
def drivers(plan: AnalysisPlan, ctx: ExecutionContext) -> EvidenceBundle:
    bundle = new_bundle(plan, ctx)
    metrics = list(plan.metrics) or list(DEFAULT_DRIVERS)
    where, params = cohort_where(plan, None)

    # Cohorts default to failed vs healthy, which is what the question almost
    # always means. An explicit pair overrides.
    if len(plan.cohorts) >= 2:
        a_name, b_name = plan.cohorts[0].name, plan.cohorts[1].name
        a_where, a_params = cohort_where(plan, a_name)
        b_where, b_params = cohort_where(plan, b_name)
    else:
        a_name, b_name = "failed", "healthy"
        a_where = f"({where}) AND machine_failure = 1"
        b_where = f"({where}) AND machine_failure = 0"
        a_params = b_params = params

    stats = {}
    for name, w, p in ((a_name, a_where, a_params), (b_name, b_where, b_params)):
        selects = []
        for m in metrics:
            col = column_for(m)
            selects += [f"avg({col})", f"stddev_samp({col})"]
        row = ctx.cursor.execute(
            f"SELECT count(*), {', '.join(selects)} FROM {TABLE} WHERE {w}", p  # noqa: S608
        ).fetchone()
        stats[name] = {"n": int(row[0]), "values": row[1:]}

    n_a, n_b = stats[a_name]["n"], stats[b_name]["n"]
    bundle.put(f"{a_name}.n", n_a, unit="count", sig_figs=8)
    bundle.put(f"{b_name}.n", n_b, unit="count", sig_figs=8)

    if n_a < 2 or n_b < 2:
        bundle.put("drivers.top", None, quality=Quality.ABSTAIN)
        bundle.warn(
            "abstained",
            f"Cohort sizes ({a_name}={n_a}, {b_name}={n_b}) are too small to rank drivers.",
            severity=Severity.CRITICAL,
        )
        return bundle

    ranked = []
    for i, metric in enumerate(metrics):
        mean_a, sd_a = float(stats[a_name]["values"][2 * i]), float(
            stats[a_name]["values"][2 * i + 1] or 0.0
        )
        mean_b, sd_b = float(stats[b_name]["values"][2 * i]), float(
            stats[b_name]["values"][2 * i + 1] or 0.0
        )
        d = cohens_d(mean_a, sd_a, n_a, mean_b, sd_b, n_b)
        ci = cohens_d_interval(d, n_a, n_b, plan.confidence)
        ranked.append((metric, d, ci, mean_a, mean_b))

    ranked.sort(key=lambda r: abs(r[1]), reverse=True)
    low = min(n_a, n_b) < MIN_REPORTABLE_N

    for rank, (metric, d, ci, mean_a, mean_b) in enumerate(ranked, start=1):
        unit = unit_for(metric)
        # The rank is itself a number the narrator needs to state. Emitting it as
        # a slot keeps the "no digits in prose" rule absolute rather than carving
        # out an exception for list numbering.
        bundle.put(f"rank{rank}.position", rank, unit="count", sig_figs=8)
        bundle.put(f"rank{rank}.metric", label_for(metric), unit="")
        bundle.put(f"rank{rank}.field", metric, unit="")
        bundle.put(
            f"rank{rank}.separation",
            d,
            unit="",
            ci=ci,
            quality=Quality.LOW_SAMPLE if low else Quality.OK,
        )
        bundle.put(f"rank{rank}.direction", "higher" if d > 0 else "lower", unit="")
        bundle.put(f"rank{rank}.{a_name}_mean", mean_a, unit=unit, n=n_a)
        bundle.put(f"rank{rank}.{b_name}_mean", mean_b, unit=unit, n=n_b)
        # A driver whose interval crosses zero has not been shown to separate.
        if ci.verdict() == "straddles":
            bundle.put(f"rank{rank}.significant", "no", unit="")
        else:
            bundle.put(f"rank{rank}.significant", "yes", unit="")

    bundle.put("drivers.top", label_for(ranked[0][0]), unit="")
    bundle.put("drivers.considered", len(ranked), unit="count", sig_figs=8)

    _flag_confounding(bundle, ctx, where, params, [r[0] for r in ranked[:4]])

    bundle.warn(
        "data_quality",
        "This ranks how strongly each variable separates the two cohorts. It is "
        "associational: a high-ranking variable is not thereby a cause. Use "
        "root_cause for mechanism.",
        severity=Severity.INFO,
    )
    bundle.provenance = bundle.provenance.model_copy(update={"row_count": n_a + n_b})
    bundle.summary = (
        f"ranked {len(ranked)} drivers separating {a_name} (n={n_a}) from "
        f"{b_name} (n={n_b}); top is {label_for(ranked[0][0])}"
    )
    return bundle


def _flag_confounding(
    bundle: EvidenceBundle,
    ctx: ExecutionContext,
    where: str,
    params: list,
    top: list[str],
) -> None:
    """Warn when two highly-ranked drivers are collinear.

    Without this, a reader sees "rotational speed separates failures" and
    reasonably concludes speed matters - when r(rpm, torque) = -0.875 means the
    two cannot be told apart by this analysis.
    """
    if len(top) < 2:
        return
    pairs = [(top[i], top[j]) for i in range(len(top)) for j in range(i + 1, len(top))]
    selects = ", ".join(f"corr({column_for(x)}, {column_for(y)})" for x, y in pairs)
    row = ctx.cursor.execute(
        f"SELECT {selects} FROM {TABLE} WHERE {where}", params  # noqa: S608
    ).fetchone()

    for (x, y), r in zip(pairs, row):
        if r is None or abs(r) < COLLINEARITY_THRESHOLD:
            continue
        bundle.put(f"corr.{x}__{y}", float(r), unit="", sig_figs=3)
        bundle.warn(
            "collinearity",
            f"{label_for(x)} and {label_for(y)} both rank highly but are "
            f"{'inversely ' if r < 0 else ''}correlated at r = {r:.3f}. Their "
            "individual contributions cannot be separated by this ranking.",
            severity=Severity.WARNING,
            affects=[x, y],
        )


__all__ = ["drivers", "DEFAULT_DRIVERS"]
