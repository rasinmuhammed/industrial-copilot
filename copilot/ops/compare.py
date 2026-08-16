"""`compare` — contrast two cohorts across metrics.

Answers the brief's third example question directly: "compare the operating
conditions of machines that failed versus those that did not."

Carries the confounding detector. In this dataset r(rpm, torque) = -0.875, so
*every* analysis attributing an effect to rotational speed is confounded by
torque. An op that reports a difference without saying so is producing a
technically-true, practically-misleading answer — which is worse than a wrong
one because it survives review.
"""

from __future__ import annotations

from copilot.evidence import EvidenceBundle, Quality, Severity
from copilot.ir import AnalysisPlan, OpName
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
from copilot.stats import (
    MIN_REPORTABLE_N,
    cohens_d,
    cohens_d_interval,
    mean_interval,
)
from copilot.units import unit as resolve_unit

# |r| above which two reported metrics are flagged as confounded.
COLLINEARITY_THRESHOLD = 0.7

# |d| interpretation bands (Cohen's conventions), used for wording only.
_EFFECT_BANDS = ((0.2, "negligible"), (0.5, "small"), (0.8, "medium"))


def _effect_label(d: float) -> str:
    magnitude = abs(d)
    for bound, name in _EFFECT_BANDS:
        if magnitude < bound:
            return name
    return "large"


@register(OpName.COMPARE)
def compare(plan: AnalysisPlan, ctx: ExecutionContext) -> EvidenceBundle:
    bundle = new_bundle(plan, ctx)
    a, b = plan.cohorts[0], plan.cohorts[1]

    stats: dict[str, dict[str, float | int]] = {}
    for cohort in (a, b):
        where, params = cohort_where(plan, cohort.name)
        selects = []
        for metric in plan.metrics:
            col = column_for(metric)
            selects += [f"avg({col})", f"stddev_samp({col})"]
        row = ctx.cursor.execute(
            f"SELECT count(*), {', '.join(selects)} FROM {TABLE} WHERE {where}",  # noqa: S608
            params,
        ).fetchone()
        n = int(row[0])
        stats[cohort.name] = {"n": n}
        for i, metric in enumerate(plan.metrics):
            stats[cohort.name][f"{metric}.mean"] = row[1 + 2 * i]
            stats[cohort.name][f"{metric}.sd"] = row[2 + 2 * i] or 0.0

    n_a = int(stats[a.name]["n"])
    n_b = int(stats[b.name]["n"])
    bundle.put(f"{a.name}.n", n_a, unit="count", sig_figs=8)
    bundle.put(f"{b.name}.n", n_b, unit="count", sig_figs=8)

    if n_a == 0 or n_b == 0:
        empty = a.name if n_a == 0 else b.name
        bundle.warn(
            "abstained",
            f"Cohort '{empty}' matched no rows, so no comparison is possible.",
            severity=Severity.CRITICAL,
        )
        for metric in plan.metrics:
            bundle.put(
                f"delta.{metric}", None, unit=unit_for(metric), quality=Quality.ABSTAIN
            )
        bundle.summary = f"comparison abstained: cohort '{empty}' is empty"
        return bundle

    for metric in plan.metrics:
        unit = unit_for(metric)
        delta_unit = resolve_unit(unit).as_delta().symbol if unit else ""
        mean_a = float(stats[a.name][f"{metric}.mean"])
        mean_b = float(stats[b.name][f"{metric}.mean"])
        sd_a = float(stats[a.name][f"{metric}.sd"])
        sd_b = float(stats[b.name][f"{metric}.sd"])

        low = min(n_a, n_b) < MIN_REPORTABLE_N
        quality = Quality.LOW_SAMPLE if low else Quality.OK

        bundle.put(
            f"{a.name}.{metric}.mean", mean_a, unit=unit, n=n_a,
            ci=mean_interval(mean_a, sd_a, n_a, plan.confidence), quality=quality,
        )
        bundle.put(
            f"{b.name}.{metric}.mean", mean_b, unit=unit, n=n_b,
            ci=mean_interval(mean_b, sd_b, n_b, plan.confidence), quality=quality,
        )
        bundle.put(f"{a.name}.{metric}.sd", sd_a, unit=unit, n=n_a)
        bundle.put(f"{b.name}.{metric}.sd", sd_b, unit=unit, n=n_b)

        # The difference is a DELTA quantity — declaring it as such prevents it
        # being compared against an absolute threshold downstream.
        bundle.put(f"delta.{metric}", mean_a - mean_b, unit=delta_unit, quality=quality)

        d = cohens_d(mean_a, sd_a, n_a, mean_b, sd_b, n_b)
        bundle.put(
            f"effect.{metric}.cohens_d", d, unit="",
            ci=cohens_d_interval(d, n_a, n_b, plan.confidence), quality=quality,
        )
        bundle.put(f"effect.{metric}.magnitude", _effect_label(d), unit="")

    _detect_collinearity(bundle, plan, ctx)

    bundle.warn(
        "data_quality",
        "This is an associational comparison. Differences between cohorts do not "
        "establish that a metric caused the failures; use root_cause for attribution.",
        severity=Severity.INFO,
    )
    bundle.provenance = bundle.provenance.model_copy(update={"row_count": n_a + n_b})
    bundle.summary = (
        f"compared {a.name} (n={n_a}) vs {b.name} (n={n_b}) across "
        + ", ".join(label_for(m) for m in plan.metrics)
    )
    return bundle


def _detect_collinearity(
    bundle: EvidenceBundle, plan: AnalysisPlan, ctx: ExecutionContext
) -> None:
    """Flag metric pairs whose correlation makes attribution ambiguous.

    Computed over the *combined* population, which is the relevant scope: if two
    reported drivers move together across the data, a difference in one cannot be
    cleanly separated from a difference in the other.
    """
    if len(plan.metrics) < 2:
        return

    where, params = cohort_where(plan, None)
    pairs = [
        (plan.metrics[i], plan.metrics[j])
        for i in range(len(plan.metrics))
        for j in range(i + 1, len(plan.metrics))
    ]
    selects = ", ".join(f"corr({column_for(x)}, {column_for(y)})" for x, y in pairs)
    row = ctx.cursor.execute(
        f"SELECT {selects} FROM {TABLE} WHERE {where}", params  # noqa: S608
    ).fetchone()

    for (x, y), r in zip(pairs, row):
        if r is None:
            continue
        bundle.put(f"corr.{x}__{y}", float(r), unit="", sig_figs=3)
        if abs(r) >= COLLINEARITY_THRESHOLD:
            direction = "inversely " if r < 0 else ""
            bundle.warn(
                "collinearity",
                f"{label_for(x)} and {label_for(y)} are strongly {direction}coupled "
                f"(r = {r:.3f}). Differences attributed to one may be driven by the "
                f"other; they cannot be separated by this comparison.",
                severity=Severity.WARNING,
                affects=[f"delta.{x}", f"delta.{y}"],
            )


__all__ = ["compare"]
