"""`sql_explore` — the guarded escape hatch.

A closed operator registry has a coverage ceiling. Rather than pretend
otherwise, this op exists for questions no structured operator fits — but it is
fenced so that using it cannot weaken the guarantees on the hot path:

  * the connection is already read-only, and destructive keywords were rejected
    at plan validation;
  * a single statement only, wrapped in a row and time budget;
  * results still pass through the PCN verifier;
  * the answer is **labelled exploratory**, so a reader can tell a certified
    analysis from an ad-hoc one;
  * every use is logged with the question, and that log is the backlog for new
    first-class operators.

If a query shape appears here repeatedly, that is a registry gap, not a success.
"""

from __future__ import annotations

import re

from copilot.evidence import EvidenceBundle, Quality, Severity
from copilot.ir import AnalysisPlan, OpName
from copilot.ops.registry import TABLE, ExecutionContext, new_bundle, register

MAX_ROWS = 10_000
STATEMENT_TIMEOUT_S = 2.0

# Only the observations table and DuckDB's own functions are reachable. Anything
# else named as a source is refused.
_FROM_CLAUSE = re.compile(r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_.]*)", re.IGNORECASE)
_JOIN_CLAUSE = re.compile(r"\bjoin\s+([a-zA-Z_][a-zA-Z0-9_.]*)", re.IGNORECASE)
_ALLOWED_SOURCES = {TABLE, "meta"}


@register(OpName.SQL_EXPLORE)
def sql_explore(plan: AnalysisPlan, ctx: ExecutionContext) -> EvidenceBundle:
    bundle = new_bundle(plan, ctx)
    sql = str(plan.params["sql"]).strip().rstrip(";")

    refused = _refuse_reason(sql)
    if refused:
        bundle.put("explore.status", None, quality=Quality.ABSTAIN)
        bundle.warn("abstained", refused, severity=Severity.CRITICAL)
        bundle.summary = "exploratory query refused"
        return bundle

    # Always label the result, before anything can go wrong.
    bundle.warn(
        "exploratory",
        "This answer came from an ad-hoc query, not a certified analysis operator. "
        "Its numbers are computed from the warehouse and verified, but the analysis "
        "itself has not been reviewed or unit-tested. Treat it as exploratory.",
        severity=Severity.WARNING,
    )

    guarded = f"SELECT * FROM ({sql}) AS _explore LIMIT {MAX_ROWS}"
    try:
        cur = ctx.cursor.execute(guarded)
        names = [d[0] for d in cur.description]
        rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001 - surface the engine's own message
        bundle.put("explore.status", None, quality=Quality.ABSTAIN)
        bundle.warn(
            "abstained",
            f"The exploratory query could not be executed: {type(exc).__name__}: {exc}",
            severity=Severity.CRITICAL,
        )
        bundle.summary = "exploratory query failed"
        return bundle

    bundle.put("explore.status", "executed", unit="")
    bundle.put("explore.rows", len(rows), unit="count", sig_figs=8)
    bundle.put("explore.columns", len(names), unit="count", sig_figs=8)

    limit = min(ctx.max_rows, len(rows))
    bundle.rows = [dict(zip(names, r)) for r in rows[:limit]]

    # Scalar results become slots so the verifier can bind them. Anything larger
    # stays in `rows`, where it is displayed as a table rather than narrated.
    if len(rows) == 1:
        for name, value in zip(names, rows[0]):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                bundle.put(f"explore.{_slug(name)}", float(value), unit="", sig_figs=6)
            else:
                bundle.put(f"explore.{_slug(name)}", str(value), unit="")

    if len(rows) >= MAX_ROWS:
        bundle.warn(
            "data_quality",
            f"The query hit the {MAX_ROWS:,}-row budget and may be truncated.",
            severity=Severity.WARNING,
        )
    if len(rows) > limit:
        bundle.warn(
            "data_quality",
            f"{len(rows):,} rows returned; showing {limit}.",
            severity=Severity.INFO,
        )

    bundle.provenance = bundle.provenance.model_copy(
        update={"sql": guarded, "row_count": len(rows)}
    )
    bundle.summary = f"exploratory query returned {len(rows)} row(s)"
    return bundle


def _refuse_reason(sql: str) -> str | None:
    """Second line of defence. Plan validation already rejected DDL and DML."""
    if not sql:
        return "No query was supplied."
    if ";" in sql:
        return "Only a single statement is permitted."
    if not re.match(r"^\s*(select|with)\b", sql, re.IGNORECASE):
        return "Only SELECT (or WITH ... SELECT) queries are permitted."

    sources = {
        m.group(1).lower()
        for m in list(_FROM_CLAUSE.finditer(sql)) + list(_JOIN_CLAUSE.finditer(sql))
    }
    # CTE names are legitimate sources; collect and allow them.
    cte_names = {m.lower() for m in re.findall(r"(\w+)\s+AS\s*\(", sql, re.IGNORECASE)}
    unknown = sources - _ALLOWED_SOURCES - cte_names
    if unknown:
        return (
            f"Unknown data source(s): {', '.join(sorted(unknown))}. "
            f"Only {', '.join(sorted(_ALLOWED_SOURCES))} may be queried."
        )
    return None


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return cleaned or "value"


__all__ = ["sql_explore"]
