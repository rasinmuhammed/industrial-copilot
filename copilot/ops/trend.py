"""`trend` - behaviour along an axis: time, or a physical variable such as wear.

Two axes are supported and they answer different questions:

  * `time_grain` - "is torque climbing?"  Depends on the SYNTHETIC timeline, so
    every answer using it carries a disclosure.
  * `bin` on a metric - "how does failure rate vary with tool wear?"  This is
    the physically meaningful one on AI4I, because wear is a real degradation
    trajectory (verified: deltas of exactly 2/3/5 min with 119 tool resets).

Slope is reported with a confidence interval. A slope without one is not a
finding - it is a direction with unknown significance.

Slot namespaces are kept disjoint on purpose: per-bucket values live under
`bucket.*`, derived statistics under `slope.*`, `axis.*` and `changepoint.*`.
Without that separation `slope.failure_rate` and `bucket.0-42.failure_rate`
share a suffix, and anything iterating by suffix silently mixes them.
"""

from __future__ import annotations

import math
from typing import Any

from copilot.evidence import EvidenceBundle, Interval, Quality, Severity
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
from copilot.stats import MIN_REPORTABLE_N, _z, wilson_interval
from copilot.units import unit as resolve_unit

# A slope is called out as a real move only if its CI excludes zero.
_GRAIN_SQL = {
    "hour": "date_trunc('hour', ts)",
    "day": "date_trunc('day', ts)",
    "shift": "date_trunc('day', ts) || '-' || shift",
}


@register(OpName.TREND)
def trend(plan: AnalysisPlan, ctx: ExecutionContext) -> EvidenceBundle:
    bundle = new_bundle(plan, ctx)
    where, params = cohort_where(plan, None)

    if plan.bin is not None:
        buckets, axis_name, axis_unit, sql = _metric_axis(plan, ctx, where, params)
    elif plan.time_grain is not None:
        buckets, axis_name, axis_unit, sql = _time_axis(plan, ctx, where, params)
    else:
        bundle.put("trend.verdict", None, quality=Quality.ABSTAIN)
        bundle.warn(
            "abstained",
            "A trend needs an axis: either a time grain or a binned metric.",
            severity=Severity.CRITICAL,
        )
        return bundle

    if len(buckets) < 3:
        bundle.put("trend.verdict", None, quality=Quality.ABSTAIN)
        bundle.warn(
            "abstained",
            f"Only {len(buckets)} points along {axis_name}; at least 3 are needed "
            "to describe a trend.",
            severity=Severity.WARNING,
        )
        return bundle

    bundle.put("axis.name", axis_name, unit="")
    bundle.put("axis.points", len(buckets), unit="count", sig_figs=8)

    xs = [float(i) for i in range(len(buckets))]
    total_n = sum(b["n"] for b in buckets)

    # Failure rate along the axis - always computed; it is the question most
    # often behind "how does X vary with Y".
    rates = [(b["failures"] / b["n"] * 100.0) if b["n"] else 0.0 for b in buckets]
    for b, r in zip(buckets, rates):
        key = f"bucket.{_slug(b['label'])}"
        # Bucket labels are numeric ranges; they must be slots, not prose.
        bundle.put(f"{key}.label", str(b["label"]), unit="")
        bundle.put(f"{key}.n", b["n"], unit="count", sig_figs=8)
        bundle.put(f"{key}.failures", b["failures"], unit="count", sig_figs=8)
        ci = wilson_interval(b["failures"], b["n"], plan.confidence) if b["n"] else None
        bundle.put(
            f"{key}.failure_rate",
            r if b["n"] else None,
            unit="%",
            n=b["n"],
            ci=Interval(lo=ci.lo * 100, hi=ci.hi * 100) if ci else None,
            quality=(
                Quality.ABSTAIN if not b["n"]
                else Quality.LOW_SAMPLE if b["n"] < MIN_REPORTABLE_N
                else Quality.OK
            ),
            sig_figs=3,
        )

    _emit_slope(bundle, "failure_rate", xs, rates, "Δ%", plan.confidence, axis_name)

    # Metric trends, when metrics were requested.
    for metric in plan.metrics:
        unit = unit_for(metric)
        delta_unit = resolve_unit(unit).as_delta().symbol if unit else ""
        ys = [b["metrics"][metric] for b in buckets]
        for b, y in zip(buckets, ys):
            bundle.put(f"bucket.{_slug(b['label'])}.{metric}", y, unit=unit, n=b["n"])
        _emit_slope(bundle, metric, xs, ys, delta_unit, plan.confidence, axis_name)

    _detect_changepoint(bundle, buckets, rates, axis_name)

    bundle.provenance = bundle.provenance.model_copy(
        update={"sql": sql, "row_count": total_n}
    )
    bundle.summary = f"trend along {axis_name} over {len(buckets)} points"
    return bundle


def _slug(value: Any) -> str:
    return (
        str(value).strip().replace(" ", "_").replace("-", "_").replace(":", "")
        .replace(".", "_").lower()
        or "unknown"
    )


def _emit_slope(
    bundle: EvidenceBundle,
    name: str,
    xs: list[float],
    ys: list[float],
    unit: str,
    confidence: float,
    axis_name: str,
) -> None:
    """OLS slope per axis step, with an interval. Direction claims require the
    interval to exclude zero."""
    n = len(xs)
    if n < 3:
        return
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = n - 2
    se = math.sqrt(sum(r * r for r in resid) / dof / sxx) if dof > 0 else 0.0
    half = _z(confidence) * se
    ci = Interval(lo=slope - half, hi=slope + half)

    significant = ci.verdict() != "straddles"
    if significant:
        direction = "rising" if slope > 0 else "falling"
    else:
        # An insignificant slope does not mean nothing is happening. A series
        # that jumps at one end has a large residual and a wide interval, and
        # calling it "flat" would be a worse error than admitting non-linearity.
        spread = max(ys) - min(ys)
        typical = sum(abs(y) for y in ys) / n or 1.0
        direction = "not well described by a straight line" if spread > typical else "flat"

    # The axis step is a dimensionless bucket index, so the slope carries the
    # same unit as the delta it measures. "per step" is a note, not a unit -
    # fabricating "ΔW/step" would put an unresolvable symbol into the evidence.
    bundle.put(
        f"slope.{name}",
        slope,
        unit=unit,
        ci=ci,
        note=f"change per step along {axis_name}",
    )
    bundle.put(f"slope.{name}.direction", direction, unit="")
    if not significant:
        bundle.warn(
            "wide_interval",
            f"The {name} slope along {axis_name} is not distinguishable from flat "
            f"at {int(confidence * 100)}% confidence; its interval includes zero.",
            severity=Severity.INFO,
            affects=[f"slope.{name}"],
        )


def _detect_changepoint(
    bundle: EvidenceBundle, buckets: list[dict], rates: list[float], axis_name: str
) -> None:
    """Single-changepoint search by maximum mean shift.

    Deliberately simple and reported as a candidate, not a conclusion - a proper
    CUSUM with control limits belongs in the streaming layer, not in an
    exploratory op.
    """
    n = len(rates)
    if n < 6:
        return
    best_idx, best_gap = None, 0.0
    for i in range(2, n - 1):
        left = sum(rates[:i]) / i
        right = sum(rates[i:]) / (n - i)
        gap = abs(right - left)
        if gap > best_gap:
            best_idx, best_gap = i, gap
    if best_idx is None:
        return
    overall = max(rates) - min(rates)
    if overall <= 0 or best_gap / overall < 0.4:
        return
    bundle.put("changepoint.at", str(buckets[best_idx]["label"]), unit="")
    bundle.put("changepoint.shift", best_gap, unit="Δ%", sig_figs=3)
    bundle.warn(
        "data_quality",
        f"Failure rate shifts markedly around {buckets[best_idx]['label']} along "
        f"{axis_name}. This is a candidate changepoint from a simple mean-shift "
        "search, not a tested inference.",
        severity=Severity.INFO,
        affects=["changepoint.at"],
    )


def _time_axis(plan: AnalysisPlan, ctx: ExecutionContext, where: str, params: list):
    grain = plan.time_grain or "day"
    expr = _GRAIN_SQL[grain]
    metric_selects = "".join(
        f", avg({column_for(m)}) AS m_{i}" for i, m in enumerate(plan.metrics)
    )
    sql = (
        f"SELECT {expr} AS bucket, count(*)::BIGINT AS n, "  # noqa: S608
        f"sum(machine_failure)::BIGINT AS failures{metric_selects} "
        f"FROM {TABLE} WHERE {where} GROUP BY bucket ORDER BY bucket"
    )
    rows = ctx.cursor.execute(sql, params).fetchall()
    buckets = [
        {
            "label": str(r[0]),
            "n": int(r[1]),
            "failures": int(r[2] or 0),
            "metrics": {m: float(r[3 + i]) for i, m in enumerate(plan.metrics)},
        }
        for r in rows
    ]
    return buckets, f"time ({grain})", "", sql


def _metric_axis(plan: AnalysisPlan, ctx: ExecutionContext, where: str, params: list):
    assert plan.bin is not None
    col = column_for(plan.bin.field)
    axis_unit = unit_for(plan.bin.field)

    if plan.bin.method == "explicit":
        edges = [float(e) for e in plan.bin.bins]  # type: ignore[union-attr]
    else:
        k = int(plan.bin.bins)  # type: ignore[arg-type]
        if plan.bin.method == "quantile":
            qs = ", ".join(f"quantile_cont({col}, {i / k})" for i in range(1, k))
            row = ctx.cursor.execute(
                f"SELECT min({col}), {qs}, max({col}) FROM {TABLE} WHERE {where}",  # noqa: S608
                params,
            ).fetchone()
        else:
            lo, hi = ctx.cursor.execute(
                f"SELECT min({col}), max({col}) FROM {TABLE} WHERE {where}", params  # noqa: S608
            ).fetchone()
            step = (float(hi) - float(lo)) / k
            row = [float(lo) + i * step for i in range(k + 1)]
        edges = [float(v) for v in row]

    edges = sorted(set(edges))
    if len(edges) < 2:
        # A filter has collapsed the axis to a single value. Emitting SQL here
        # produces an empty CASE; abstaining is the honest outcome.
        return [], label_for(plan.bin.field), axis_unit, ""
    span = edges[-1] - edges[0]
    cases = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        label = f"{_fmt_edge(lo, span)}-{_fmt_edge(hi, span)}"
        cases.append(
            f"WHEN {col} >= {lo} AND {col} {'<=' if last else '<'} {hi} "
            f"THEN '{label}'"
        )
    bucket = "CASE " + " ".join(cases) + " ELSE 'out_of_range' END"
    metric_selects = "".join(
        f", avg({column_for(m)}) AS m_{i}" for i, m in enumerate(plan.metrics)
    )
    sql = (
        f"SELECT {bucket} AS bucket, count(*)::BIGINT AS n, "  # noqa: S608
        f"sum(machine_failure)::BIGINT AS failures, min({col}) AS lo{metric_selects} "
        f"FROM {TABLE} WHERE {where} GROUP BY bucket ORDER BY lo"
    )
    rows = ctx.cursor.execute(sql, params).fetchall()
    buckets = [
        {
            "label": str(r[0]),
            "n": int(r[1]),
            "failures": int(r[2] or 0),
            "metrics": {m: float(r[4 + i]) for i, m in enumerate(plan.metrics)},
        }
        for r in rows
    ]
    return buckets, label_for(plan.bin.field), axis_unit, sql


__all__ = ["trend"]

def _fmt_edge(value: float, span: float) -> str:
    """Bin labels an engineer can read. 6 significant figures on a 0-253 range
    produces '42.1667', which is noise, not precision."""
    if span >= 100:
        return f"{value:.0f}"
    if span >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"
