"""Operator dispatch and the SQL compilation layer.

The planner never writes SQL. It emits an `AnalysisPlan`, and this module turns
the plan's declarative filters into parameterised predicates. Values are always
bound, never interpolated — the field name is the only part of a filter that
reaches the SQL text, and it has already been checked against the semantic layer,
so it cannot be attacker-controlled.

Adding an operator means registering it here. The registry is closed at import
time, which is what makes "the model invented an operation" impossible.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import duckdb

from copilot.evidence import EvidenceBundle, Provenance, Severity
from copilot.ir import AnalysisPlan, Filter, FilterOp, OpName
from copilot.knowledge import dimension_index, failure_modes, metric_index
from copilot.units import convert

__all__ = [
    "ExecutionContext",
    "OpFunction",
    "register",
    "get_op",
    "execute",
    "compile_filters",
    "column_for",
    "TABLE",
]

TABLE = "observations"


@dataclass(slots=True)
class ExecutionContext:
    """Everything an op needs that is not the plan itself."""

    con: duckdb.DuckDBPyConnection
    kb_version: str = "1.0.0"
    data_version: str = "unknown"
    tier: str = "grammar"
    max_rows: int = 50
    started: float = field(default_factory=time.perf_counter)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0


class OpFunction(Protocol):
    def __call__(self, plan: AnalysisPlan, ctx: ExecutionContext) -> EvidenceBundle: ...


_REGISTRY: dict[OpName, OpFunction] = {}


def register(op: OpName) -> Callable[[OpFunction], OpFunction]:
    def decorator(fn: OpFunction) -> OpFunction:
        if op in _REGISTRY:
            raise RuntimeError(f"operator {op.value!r} already registered")
        _REGISTRY[op] = fn
        return fn

    return decorator


def get_op(op: OpName) -> OpFunction:
    try:
        return _REGISTRY[op]
    except KeyError:
        raise NotImplementedError(
            f"operator {op.value!r} is declared but not yet implemented"
        ) from None


def registered() -> list[str]:
    return sorted(o.value for o in _REGISTRY)


# --------------------------------------------------------------------------
# Schema resolution
# --------------------------------------------------------------------------


def column_for(name: str) -> str:
    """Semantic-layer name -> physical column.

    Raises KeyError if unknown; validation should already have caught it, so a
    failure here is a bug in the validator, not in the plan.
    """
    metrics, dims = metric_index(), dimension_index()
    if name in metrics:
        return metrics[name]["column"]
    if name in dims:
        column = dims[name].get("column")
        if column is None:
            raise KeyError(f"dimension {name!r} has no backing column")
        return column
    raise KeyError(f"{name!r} is not in the semantic layer")


def unit_for(name: str) -> str:
    return metric_index().get(name, {}).get("unit", "")


def label_for(name: str) -> str:
    metrics, dims = metric_index(), dimension_index()
    if name in metrics:
        return metrics[name]["label"]
    if name in dims:
        return dims[name]["label"]
    return name


# --------------------------------------------------------------------------
# Filter compilation
# --------------------------------------------------------------------------

_SQL_OP = {
    FilterOp.EQ: "=",
    FilterOp.NE: "!=",
    FilterOp.LT: "<",
    FilterOp.LTE: "<=",
    FilterOp.GT: ">",
    FilterOp.GTE: ">=",
}


def _coerce(f: Filter, value: Any) -> Any:
    """Convert a filter value into the column's native unit.

    A plan may legitimately say "air temperature above 25 degC"; the warehouse
    stores kelvin. Dimensional validation has already proven compatibility.
    """
    if not f.unit or not isinstance(value, (int, float)):
        return value
    target = unit_for(f.field)
    if not target or f.unit == target:
        return value
    return convert(float(value), f.unit, target)


def compile_filters(filters: list[Filter]) -> tuple[str, list[Any]]:
    """Return (where_clause, params). Empty filters yield 'TRUE'."""
    if not filters:
        return "TRUE", []

    clauses: list[str] = []
    params: list[Any] = []
    for f in filters:
        col = column_for(f.field)
        if f.op is FilterOp.IS_NULL:
            clauses.append(f"{col} IS NULL")
        elif f.op is FilterOp.BETWEEN:
            lo, hi = f.value  # type: ignore[misc]
            clauses.append(f"{col} BETWEEN ? AND ?")
            params += [_coerce(f, lo), _coerce(f, hi)]
        elif f.op is FilterOp.IN:
            values = list(f.value)  # type: ignore[arg-type]
            placeholders = ", ".join("?" for _ in values)
            clauses.append(f"{col} IN ({placeholders})")
            params += [_coerce(f, v) for v in values]
        else:
            clauses.append(f"{col} {_SQL_OP[f.op]} ?")
            params.append(_coerce(f, f.value))
    return " AND ".join(clauses), params


def cohort_where(plan: AnalysisPlan, cohort_name: str | None = None) -> tuple[str, list[Any]]:
    """Global filters AND the named cohort's filters."""
    filters = list(plan.filters)
    if cohort_name is not None:
        for c in plan.cohorts:
            if c.name == cohort_name:
                filters += c.filters
                break
    return compile_filters(filters)


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def new_bundle(plan: AnalysisPlan, ctx: ExecutionContext, *, sql: str | None = None) -> EvidenceBundle:
    """Create a bundle pre-populated with provenance and mandatory warnings."""
    synthetic = plan.synthetic_used()
    bundle = EvidenceBundle(
        provenance=Provenance(
            plan_hash=plan.hash,
            kb_version=ctx.kb_version,
            data_version=ctx.data_version,
            op=plan.op.value,
            tier=ctx.tier,  # type: ignore[arg-type]
            filters=plan.describe_filters(),
            sql=sql,
            synthetic_dimensions=synthetic,
        )
    )
    if synthetic:
        bundle.warn(
            "synthetic_dimension",
            "This answer relies on "
            + ", ".join(synthetic)
            + ", which "
            + ("is" if len(synthetic) == 1 else "are")
            + " a synthetic overlay not present in the published dataset. "
            "See README > Assumptions.",
            severity=Severity.INFO,
            affects=synthetic,
        )
    return bundle


def execute(plan: AnalysisPlan, ctx: ExecutionContext) -> EvidenceBundle:
    """Run a validated plan. The only public execution entry point."""
    ctx.started = time.perf_counter()
    bundle = get_op(plan.op)(plan, ctx)
    # Provenance is frozen, so stamp elapsed time by rebuilding it.
    bundle.provenance = bundle.provenance.model_copy(
        update={"elapsed_ms": round(ctx.elapsed_ms(), 3)}
    )
    return bundle


def data_fingerprint(con: duckdb.DuckDBPyConnection) -> str:
    """Cheap warehouse fingerprint. Keys the answer cache so a rebuild
    invalidates stale answers automatically."""
    row = con.execute(
        f"SELECT count(*), round(sum(torque_nm) + sum(rotational_speed_rpm), 4) FROM {TABLE}"
    ).fetchone()
    import hashlib

    return hashlib.sha256(str(row).encode()).hexdigest()[:12]


def kb_version() -> str:
    """Version + content hash of the knowledge base, so an answer can be
    replayed against the rules that produced it."""
    import hashlib
    import json

    kb = failure_modes()
    digest = hashlib.sha256(
        json.dumps(kb, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    return f"{kb.get('version', 0)}.{digest}"
