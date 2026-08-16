"""`records` — return example rows as evidence.

The drill-down behind every other answer: "show me those cycles". Rows are
capped, ordered deterministically so a replay returns the same set, and carry
their margins so an engineer can see *why* each row is in the set rather than
just that it is.
"""

from __future__ import annotations

from copilot.evidence import EvidenceBundle, Quality, Severity
from copilot.ir import AnalysisPlan, OpName
from copilot.ops.registry import (
    TABLE,
    ExecutionContext,
    cohort_where,
    column_for,
    new_bundle,
    register,
)

# Always returned, so a row is self-explanatory out of context.
_BASE_COLUMNS = (
    "udi",
    "product_type",
    "machine_id",
    "machine_failure",
    "air_temperature_k",
    "process_temperature_k",
    "temp_delta_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "power_w",
    "overstrain_min_nm",
    "worst_normalised_margin",
)

_MODE_COLUMNS = ("twf", "hdf", "pwf", "osf", "rnf")

_ORDER_BY = {
    "closest_to_failure": "worst_normalised_margin ASC",
    "most_recent": "ts DESC",
    "udi": "udi ASC",
}


@register(OpName.RECORDS)
def records(plan: AnalysisPlan, ctx: ExecutionContext) -> EvidenceBundle:
    bundle = new_bundle(plan, ctx)
    where, params = cohort_where(plan, None)

    order_key = str(plan.params.get("order", "udi"))
    if order_key not in _ORDER_BY:
        bundle.warn(
            "data_quality",
            f"Unknown ordering {order_key!r}; falling back to cycle order. "
            f"Valid orderings: {', '.join(_ORDER_BY)}.",
            severity=Severity.INFO,
        )
        order_key = "udi"

    # Requested metrics are appended if they are not already in the base set.
    extra = [
        column_for(m) for m in plan.metrics if column_for(m) not in _BASE_COLUMNS
    ]
    columns = list(_BASE_COLUMNS) + list(_MODE_COLUMNS) + extra
    limit = min(plan.limit, ctx.max_rows)

    total = ctx.cursor.execute(
        f"SELECT count(*) FROM {TABLE} WHERE {where}", params  # noqa: S608
    ).fetchone()[0]

    bundle.put("matched.count", int(total), unit="count", sig_figs=8)
    bundle.put("returned.count", min(int(total), limit), unit="count", sig_figs=8)

    if total == 0:
        bundle.put("records.first_udi", None, quality=Quality.ABSTAIN)
        bundle.warn("abstained", "No cycles match this scope.", severity=Severity.WARNING)
        bundle.summary = "no matching cycles"
        return bundle

    sql = (
        f"SELECT {', '.join(columns)} FROM {TABLE} WHERE {where} "  # noqa: S608
        f"ORDER BY {_ORDER_BY[order_key]} LIMIT {limit}"
    )
    cur = ctx.cursor.execute(sql, params)
    names = [d[0] for d in cur.description]
    rows = [dict(zip(names, r)) for r in cur.fetchall()]
    bundle.rows = rows

    # A couple of scalar slots so the narrator can refer to the set without
    # inventing a count.
    bundle.put("records.first_udi", int(rows[0]["udi"]), unit="count", sig_figs=8)
    bundle.put(
        "records.failures_in_set",
        sum(int(r["machine_failure"]) for r in rows),
        unit="count",
        sig_figs=8,
    )

    if total > limit:
        bundle.warn(
            "data_quality",
            f"{int(total)} cycles match but only {limit} are shown. Narrow the filters "
            "for a complete set, or use an aggregate op.",
            severity=Severity.INFO,
            affects=["returned.count"],
        )

    bundle.provenance = bundle.provenance.model_copy(
        update={"sql": sql, "row_count": int(total)}
    )
    bundle.summary = f"{min(int(total), limit)} of {int(total)} matching cycles"
    return bundle


__all__ = ["records"]
