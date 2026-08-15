"""`rate` — failure rates and counts, optionally grouped or binned.

Answers acceptance criterion 2, "analyse historical data". This is also the op
that carries premise verification: a question like "why are we seeing MORE
failures at high rpm?" is answered by testing the claim first.

Every rate carries a Wilson interval. Groups below the reporting floor are
emitted as counts with intervals rather than as a percentage — 4.2% from 12
observations is a lie of precision.
"""

from __future__ import annotations

from typing import Any

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
)
from copilot.stats import MIN_REPORTABLE_N, wilson_interval

FAILURE_COLUMN = "machine_failure"
_SLUG_TRANSLATION = str.maketrans({" ": "_", "-": "_", ".": "_", "[": "", "]": "", ")": "", "(": ""})


def _slug(value: Any) -> str:
    return str(value).strip().translate(_SLUG_TRANSLATION).lower() or "unknown"


@register(OpName.RATE)
def rate(plan: AnalysisPlan, ctx: ExecutionContext) -> EvidenceBundle:
    bundle = new_bundle(plan, ctx)
    where, params = cohort_where(plan, None)

    if plan.bin is not None:
        groups, sql = _binned_groups(plan, ctx, where, params)
        axis_label = label_for(plan.bin.field)
    elif plan.group_by:
        groups, sql = _dimension_groups(plan, ctx, where, params)
        axis_label = ", ".join(label_for(d) for d in plan.group_by)
    else:
        groups, sql = _overall(ctx, where, params)
        axis_label = ""

    total_n = sum(n for _, _, n in groups)
    total_fail = sum(f for _, f, _ in groups)

    for key, failures, n in groups:
        prefix = _slug(key)
        ci = wilson_interval(failures, n, plan.confidence)
        low = n < MIN_REPORTABLE_N

        # The label is itself a number ("1168-1405"). It must be a slot, or the
        # narrator would have to write digits and the verifier would reject it.
        bundle.put(f"{prefix}.label", str(key), unit="")
        bundle.put(f"{prefix}.failures", failures, unit="count", sig_figs=8)
        bundle.put(f"{prefix}.n", n, unit="count", sig_figs=8)
        bundle.put(
            f"{prefix}.failure_rate",
            (failures / n * 100.0) if n else None,
            unit="%",
            n=n,
            ci=None if not n else _pct(ci),
            quality=Quality.ABSTAIN if not n else (Quality.LOW_SAMPLE if low else Quality.OK),
            sig_figs=3,
        )
        if low and n:
            bundle.warn(
                "low_sample",
                f"Group '{key}' has only {n} rows (floor is {MIN_REPORTABLE_N}). "
                f"The rate is reported with its interval; do not treat it as precise.",
                affects=[f"{prefix}.failure_rate"],
            )

    if len(groups) > 1:
        bundle.put("overall.n", total_n, unit="count", sig_figs=8)
        bundle.put("overall.failures", total_fail, unit="count", sig_figs=8)
        bundle.put(
            "overall.failure_rate",
            total_fail / total_n * 100.0 if total_n else None,
            unit="%",
            n=total_n,
            ci=_pct(wilson_interval(total_fail, total_n, plan.confidence)) if total_n else None,
            quality=Quality.OK if total_n else Quality.ABSTAIN,
            sig_figs=3,
        )
        _verify_monotone_premise(bundle, plan, groups, axis_label)

    bundle.provenance = bundle.provenance.model_copy(
        update={"sql": sql, "row_count": total_n}
    )
    bundle.summary = f"failure rate{' by ' + axis_label if axis_label else ''}"
    return bundle


def _pct(ci):
    from copilot.evidence import Interval

    return Interval(lo=ci.lo * 100.0, hi=ci.hi * 100.0)


def _overall(ctx: ExecutionContext, where: str, params: list[Any]):
    sql = (
        f"SELECT sum({FAILURE_COLUMN})::BIGINT, count(*)::BIGINT "  # noqa: S608
        f"FROM {TABLE} WHERE {where}"
    )
    failures, n = ctx.con.execute(sql, params).fetchone()
    return [("all", int(failures or 0), int(n or 0))], sql


def _dimension_groups(plan: AnalysisPlan, ctx: ExecutionContext, where: str, params: list[Any]):
    cols = [column_for(d) for d in plan.group_by]
    key_expr = " || '/' || ".join(f"CAST({c} AS VARCHAR)" for c in cols)
    sql = (
        f"SELECT {key_expr} AS grp, sum({FAILURE_COLUMN})::BIGINT, count(*)::BIGINT "  # noqa: S608
        f"FROM {TABLE} WHERE {where} GROUP BY grp ORDER BY grp"
    )
    rows = ctx.con.execute(sql, params).fetchall()
    return [(r[0], int(r[1] or 0), int(r[2])) for r in rows], sql


def _binned_groups(plan: AnalysisPlan, ctx: ExecutionContext, where: str, params: list[Any]):
    """Bin a continuous metric. Quantile binning uses DuckDB's own quantiles so
    the reference implementation and the engine agree on edges."""
    assert plan.bin is not None
    col = column_for(plan.bin.field)

    if plan.bin.method == "explicit":
        edges = [float(e) for e in plan.bin.bins]  # type: ignore[union-attr]
    else:
        k = int(plan.bin.bins)  # type: ignore[arg-type]
        if plan.bin.method == "quantile":
            qs = [i / k for i in range(1, k)]
            inner = ", ".join(f"quantile_cont({col}, {q})" for q in qs)
            cut_row = ctx.con.execute(
                f"SELECT min({col}), {inner}, max({col}) FROM {TABLE} WHERE {where}",  # noqa: S608
                params,
            ).fetchone()
            edges = [float(v) for v in cut_row]
        else:  # equal width
            lo, hi = ctx.con.execute(
                f"SELECT min({col}), max({col}) FROM {TABLE} WHERE {where}", params  # noqa: S608
            ).fetchone()
            step = (float(hi) - float(lo)) / k
            edges = [float(lo) + i * step for i in range(k + 1)]

    # Deduplicate while preserving order — quantiles collide on skewed columns.
    edges = sorted(set(edges))
    if len(edges) < 2:
        # The cohort has collapsed the axis to one value; fall back to an
        # ungrouped rate rather than emitting an empty CASE expression.
        return _overall(ctx, where, params)
    span = edges[-1] - edges[0]
    cases = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        cond = f"{col} >= {lo} AND {col} {'<=' if last else '<'} {hi}"
        label = f"{_fmt_edge(lo, span)}-{_fmt_edge(hi, span)}"
        cases.append(f"WHEN {cond} THEN '{label}'")
    bucket = "CASE " + " ".join(cases) + " ELSE 'out_of_range' END"

    sql = (
        f"SELECT {bucket} AS grp, sum({FAILURE_COLUMN})::BIGINT, count(*)::BIGINT, "  # noqa: S608
        f"min({col}) AS lo FROM {TABLE} WHERE {where} GROUP BY grp ORDER BY lo"
    )
    rows = ctx.con.execute(sql, params).fetchall()
    return [(r[0], int(r[1] or 0), int(r[2])) for r in rows], sql


def _verify_monotone_premise(
    bundle: EvidenceBundle, plan: AnalysisPlan, groups, axis_label: str
) -> None:
    """Gate 1. Test whether the rate actually rises across the axis.

    The brief's own example question — "why are we seeing more failures at high
    rotational speeds?" — is false on this data: the relationship is U-shaped and
    failures are 5.4x more common at LOW speed. A copilot that answers the
    question as asked confabulates a supporting story.
    """
    if not plan.verify_premise or plan.bin is None or len(groups) < 3:
        return

    rates = [(f / n * 100.0) if n else 0.0 for _, f, n in groups]
    first, last, peak = rates[0], rates[-1], max(rates)
    rising = all(b >= a for a, b in zip(rates, rates[1:]))
    falling = all(b <= a for a, b in zip(rates, rates[1:]))

    if rising or falling:
        shape = "increases" if rising else "decreases"
        bundle.put("premise.shape", shape, unit="")
        return

    # Non-monotone. Say so, and say which end dominates.
    interior_peak = peak > max(first, last) * 1.05
    shape = "U-shaped" if (first > min(rates) * 1.05 and last > min(rates) * 1.05) else "non-monotonic"
    if interior_peak:
        shape = "peaked in the middle"

    bundle.put("premise.shape", shape, unit="")
    bundle.put("premise.first_group_rate", first, unit="%", sig_figs=3)
    bundle.put("premise.last_group_rate", last, unit="%", sig_figs=3)
    if last > 0:
        bundle.put("premise.low_high_ratio", first / last, unit="ratio", sig_figs=2)

    dominant = "lowest" if first > last else "highest"
    bundle.warn(
        "premise_refuted",
        f"Failure rate across {axis_label} is {shape}, not monotonic. "
        f"The {dominant} band has the higher rate. Any claim that failures simply "
        f"rise with {axis_label} is not supported by this data — lead with that "
        f"before explaining mechanism.",
        severity=Severity.CRITICAL,
        affects=["premise.shape"],
    )


__all__ = ["rate"]

def _fmt_edge(value: float, span: float) -> str:
    """Bin labels an engineer can read. 6 significant figures on a 0-253 range
    produces '42.1667', which is noise, not precision."""
    if span >= 100:
        return f"{value:.0f}"
    if span >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"
