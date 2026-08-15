"""`describe` — distribution summary for one or more metrics over a cohort.

Answers acceptance criterion 1, "understand machine behaviour": what are typical
operating conditions here?

Every statistic is emitted as a Slot with its unit and n. There is no path by
which a number leaves this op without provenance.
"""

from __future__ import annotations

from copilot.evidence import EvidenceBundle, Interval, Quality
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
from copilot.stats import MIN_REPORTABLE_N, mean_interval

_AGGREGATES = (
    ("n", "count({col})"),
    ("mean", "avg({col})"),
    ("sd", "stddev_samp({col})"),
    ("min", "min({col})"),
    ("p25", "quantile_cont({col}, 0.25)"),
    ("median", "quantile_cont({col}, 0.50)"),
    ("p75", "quantile_cont({col}, 0.75)"),
    ("max", "max({col})"),
)


@register(OpName.DESCRIBE)
def describe(plan: AnalysisPlan, ctx: ExecutionContext) -> EvidenceBundle:
    # A describe with no explicit cohort still has one scope: call it "all".
    cohorts = [c.name for c in plan.cohorts] or [None]

    selects: list[str] = []
    for metric in plan.metrics:
        col = column_for(metric)
        selects += [f"{expr.format(col=col)} AS {metric}__{stat}" for stat, expr in _AGGREGATES]

    sql_shown = None
    bundle = new_bundle(plan, ctx)

    for cohort in cohorts:
        prefix = cohort or "all"
        where, params = cohort_where(plan, cohort)
        sql = f"SELECT {', '.join(selects)} FROM {TABLE} WHERE {where}"  # noqa: S608
        sql_shown = sql_shown or sql
        row = ctx.con.execute(sql, params).fetchone()
        if row is None:  # pragma: no cover - aggregate always returns a row
            continue
        values = dict(zip([f"{m}__{s}" for m in plan.metrics for s, _ in _AGGREGATES], row))

        for metric in plan.metrics:
            unit = unit_for(metric)
            n = int(values[f"{metric}__n"] or 0)

            if n == 0:
                bundle.put(
                    f"{prefix}.{metric}.mean",
                    None,
                    unit=unit,
                    n=0,
                    quality=Quality.ABSTAIN,
                    note="no rows matched this cohort",
                )
                continue

            mean = float(values[f"{metric}__mean"])
            sd = float(values[f"{metric}__sd"] or 0.0)
            ci = mean_interval(mean, sd, n, plan.confidence)
            low_sample = n < MIN_REPORTABLE_N

            bundle.put(
                f"{prefix}.{metric}.mean",
                mean,
                unit=unit,
                n=n,
                ci=ci,
                quality=Quality.LOW_SAMPLE if low_sample else Quality.OK,
            )
            bundle.put(f"{prefix}.{metric}.sd", sd, unit=unit, n=n)
            bundle.put(f"{prefix}.{metric}.n", n, unit="count", sig_figs=8)
            for stat in ("min", "p25", "median", "p75", "max"):
                bundle.put(
                    f"{prefix}.{metric}.{stat}",
                    float(values[f"{metric}__{stat}"]),
                    unit=unit,
                    n=n,
                )

            if low_sample:
                bundle.warn(
                    "low_sample",
                    f"Only {n} rows match for {label_for(metric)} in cohort "
                    f"'{prefix}'. Treat the mean as indicative, not precise.",
                    affects=[f"{prefix}.{metric}.mean"],
                )

    bundle.provenance = bundle.provenance.model_copy(update={"sql": sql_shown})
    _stamp_row_count(bundle, plan, ctx)
    bundle.summary = "described " + ", ".join(label_for(m) for m in plan.metrics)
    return bundle


def _stamp_row_count(bundle: EvidenceBundle, plan: AnalysisPlan, ctx: ExecutionContext) -> None:
    where, params = cohort_where(plan, None)
    total = ctx.con.execute(
        f"SELECT count(*) FROM {TABLE} WHERE {where}", params  # noqa: S608
    ).fetchone()[0]
    bundle.provenance = bundle.provenance.model_copy(update={"row_count": int(total)})


__all__ = ["describe"]
