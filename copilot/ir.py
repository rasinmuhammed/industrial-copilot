"""The Analysis IR: a validated plan, never SQL and never prose.

A planner (grammar, encoder, SLM or LLM) emits an `AnalysisPlan`. It is checked
against the semantic layer *before* anything executes, so a plan naming a column
that does not exist is rejected rather than discovered at runtime.

The decisive property is the failure mode. Text-to-SQL fails by producing a
plausible wrong answer (BIRD state of the art is ~72-76% execution accuracy). A
closed registry fails by producing NO answer, which is recoverable: escalate,
ask a clarifying question, or fall back to the guarded escape hatch.

Validation runs cheapest-rejection-first:

    1. structural    Pydantic types and enums
    2. vocabulary    every field exists in the semantic layer   <- the barrier
    3. dimensional   comparisons are unit-coherent
    4. cardinality   grouping cannot explode
    5. viability     the op has what it needs to run
    6. op-specific   params match the op's own schema
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from copilot.evidence import plan_hash
from copilot.knowledge import dimension_index, metric_index
from copilot.units import UnitError, assert_compatible
from copilot.units import unit as resolve_unit

__all__ = [
    "OpName",
    "FilterOp",
    "Filter",
    "Cohort",
    "Binning",
    "AnalysisPlan",
    "PlanError",
    "ValidationStage",
    "validate_plan",
]

# Dimensions that exist only because we invented them. Any plan touching one
# forces a SYNTHETIC warning into the answer.
SYNTHETIC_DIMENSIONS = frozenset({"machine_id", "shift"})
SYNTHETIC_TIME = frozenset({"ts"})

# Grouping cardinality ceiling — beyond this a plan must bin or filter.
MAX_GROUPS = 50


class OpName(StrEnum):
    """Closed set. A planner cannot invent an operation."""

    DESCRIBE = "describe"
    RATE = "rate"
    COMPARE = "compare"
    TREND = "trend"
    DRIVERS = "drivers"
    ROOT_CAUSE = "root_cause"
    COUNTERFACTUAL = "counterfactual"
    ENVELOPE = "envelope"
    FORECAST = "forecast"
    RECORDS = "records"
    DATA_QUALITY = "data_quality"
    SQL_EXPLORE = "sql_explore"


class FilterOp(StrEnum):
    EQ = "="
    NE = "!="
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="
    IN = "in"
    BETWEEN = "between"
    IS_NULL = "is_null"


class ValidationStage(StrEnum):
    STRUCTURAL = "structural"
    VOCABULARY = "vocabulary"
    DIMENSIONAL = "dimensional"
    CARDINALITY = "cardinality"
    VIABILITY = "viability"
    OP_SPECIFIC = "op_specific"


class PlanError(ValueError):
    """A plan was rejected. Carries the stage so the repair prompt is specific."""

    def __init__(self, stage: ValidationStage, message: str, *, hint: str = "") -> None:
        self.stage = stage
        self.hint = hint
        super().__init__(message)

    def repair_prompt(self) -> str:
        base = f"The plan was rejected at the {self.stage.value} stage: {self}"
        return f"{base}\nHint: {self.hint}" if self.hint else base


class Filter(BaseModel):
    """One predicate. `field` must resolve to a semantic-layer metric or dimension."""

    model_config = ConfigDict(frozen=True)

    field: str
    op: FilterOp
    value: float | int | str | bool | list[Any] | None = None
    unit: str = ""  # optional; when given it must match the field's unit

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if self.op is FilterOp.IS_NULL:
            return self
        if self.value is None:
            raise ValueError(f"filter on {self.field!r} with op {self.op} needs a value")
        if self.op is FilterOp.BETWEEN:
            if not isinstance(self.value, list) or len(self.value) != 2:
                raise ValueError("'between' requires exactly two bounds")
            lo, hi = self.value
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
                raise ValueError("'between' bounds must be numeric")
            if lo > hi:
                raise ValueError(f"'between' bounds out of order: {lo} > {hi}")
        if self.op is FilterOp.IN and not isinstance(self.value, list):
            raise ValueError("'in' requires a list")
        if self.op in {FilterOp.LT, FilterOp.LTE, FilterOp.GT, FilterOp.GTE}:
            if not isinstance(self.value, (int, float)):
                raise ValueError(f"op {self.op} requires a numeric value")
        return self

    def describe(self) -> str:
        if self.op is FilterOp.IS_NULL:
            return f"{self.field} is null"
        if self.op is FilterOp.BETWEEN:
            lo, hi = self.value  # type: ignore[misc]
            return f"{self.field} between {lo} and {hi}"
        return f"{self.field} {self.op.value} {self.value}"


class Cohort(BaseModel):
    """A named subset. Cohort names become slot-id prefixes, which is how
    the verifier makes mis-attribution unrepresentable."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=32)
    filters: list[Filter] = Field(default_factory=list)


class Binning(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    method: Literal["quantile", "width", "explicit"] = "quantile"
    bins: int | list[float] = 5

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if self.method == "explicit":
            if not isinstance(self.bins, list) or len(self.bins) < 2:
                raise ValueError("explicit binning needs at least two edges")
            if list(self.bins) != sorted(self.bins):
                raise ValueError("explicit bin edges must be ascending")
        elif not isinstance(self.bins, int) or not 2 <= self.bins <= MAX_GROUPS:
            raise ValueError(f"bins must be an int in [2, {MAX_GROUPS}]")
        return self


class AnalysisPlan(BaseModel):
    """What the planner emits. Data, not code, not SQL."""

    model_config = ConfigDict(frozen=True)

    op: OpName
    cohorts: list[Cohort] = Field(default_factory=list)
    filters: list[Filter] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    bin: Binning | None = None
    time_grain: Literal["hour", "shift", "day"] | None = None
    limit: int = Field(default=50, ge=1, le=500)
    confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    effect_size: Literal["cohens_d", "rate_ratio", "risk_diff"] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    verify_premise: bool = True
    explain: bool = True

    # -- derived -----------------------------------------------------------

    @property
    def hash(self) -> str:
        return plan_hash(self.model_dump(mode="json"))

    def all_fields(self) -> list[str]:
        """Every semantic-layer name this plan references."""
        names = [*self.metrics, *self.dimensions, *self.group_by]
        names += [f.field for f in self.filters]
        names += [f.field for c in self.cohorts for f in c.filters]
        if self.bin:
            names.append(self.bin.field)
        return names

    def synthetic_used(self) -> list[str]:
        used = {n for n in self.all_fields() if n in SYNTHETIC_DIMENSIONS}
        if self.time_grain is not None:
            used |= SYNTHETIC_TIME
        return sorted(used)

    def describe_filters(self) -> list[str]:
        return [f.describe() for f in self.filters]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

# Per-op requirements. Keeping this declarative means adding an op cannot
# silently skip validation.
_OP_REQUIREMENTS: dict[OpName, dict[str, Any]] = {
    OpName.DESCRIBE: {"needs_metrics": True},
    OpName.RATE: {"needs_metrics": False},
    OpName.COMPARE: {"needs_metrics": True, "min_cohorts": 2},
    # A trend needs an AXIS, not a metric: "how does failure rate vary with tool
    # wear?" is a complete question with no metric in it.
    OpName.TREND: {"needs_metrics": False, "needs_axis": True},
    OpName.DRIVERS: {"needs_metrics": False},
    OpName.ROOT_CAUSE: {"needs_metrics": False},
    OpName.COUNTERFACTUAL: {"needs_metrics": False, "required_params": ["changes"]},
    OpName.ENVELOPE: {"needs_metrics": False},
    OpName.FORECAST: {"needs_metrics": False},
    OpName.RECORDS: {"needs_metrics": False},
    OpName.DATA_QUALITY: {"needs_metrics": False},
    OpName.SQL_EXPLORE: {"needs_metrics": False, "required_params": ["sql"]},
}


def _known_names() -> tuple[set[str], set[str]]:
    return set(metric_index()), set(dimension_index())


def _suggest(name: str, candidates: set[str]) -> str:
    """Cheap nearest-name hint for the repair prompt."""
    import difflib

    close = difflib.get_close_matches(name, sorted(candidates), n=3, cutoff=0.6)
    return f" Did you mean: {', '.join(close)}?" if close else ""


def _validate_vocabulary(plan: AnalysisPlan) -> None:
    metrics, dimensions = _known_names()
    known = metrics | dimensions

    for name in plan.metrics:
        if name not in metrics:
            raise PlanError(
                ValidationStage.VOCABULARY,
                f"unknown metric {name!r}",
                hint=f"Valid metrics: {', '.join(sorted(metrics))}.{_suggest(name, metrics)}",
            )
    for name in [*plan.dimensions, *plan.group_by]:
        if name not in dimensions:
            raise PlanError(
                ValidationStage.VOCABULARY,
                f"unknown dimension {name!r}",
                hint=f"Valid dimensions: {', '.join(sorted(dimensions))}."
                f"{_suggest(name, dimensions)}",
            )
    for f in [*plan.filters, *(f for c in plan.cohorts for f in c.filters)]:
        if f.field not in known:
            raise PlanError(
                ValidationStage.VOCABULARY,
                f"unknown field {f.field!r} in filter",
                hint=_suggest(f.field, known).strip() or "Check the semantic layer.",
            )
    if plan.bin and plan.bin.field not in known:
        raise PlanError(
            ValidationStage.VOCABULARY,
            f"unknown field {plan.bin.field!r} in binning",
            hint=_suggest(plan.bin.field, known).strip(),
        )


def _validate_dimensional(plan: AnalysisPlan) -> None:
    """Reject unit-incoherent comparisons — the °C-vs-K class of bug."""
    metrics = metric_index()
    for f in [*plan.filters, *(f for c in plan.cohorts for f in c.filters)]:
        if not f.unit or f.field not in metrics:
            continue
        field_unit = metrics[f.field].get("unit", "")
        try:
            assert_compatible(f.unit, field_unit, context=f"filter on {f.field}")
        except UnitError as exc:
            raise PlanError(
                ValidationStage.DIMENSIONAL,
                str(exc),
                hint=f"{f.field} is measured in {field_unit}.",
            ) from exc


def _validate_cardinality(plan: AnalysisPlan) -> None:
    dims = dimension_index()
    for name in plan.group_by:
        values = dims.get(name, {}).get("values")
        if values is None:
            continue  # open-valued (machine_id); the executor caps it
        if len(values) > MAX_GROUPS:
            raise PlanError(
                ValidationStage.CARDINALITY,
                f"grouping by {name!r} would produce {len(values)} groups",
                hint=f"Filter first, or bin. Ceiling is {MAX_GROUPS}.",
            )
    if len(plan.group_by) > 2:
        raise PlanError(
            ValidationStage.CARDINALITY,
            f"grouping by {len(plan.group_by)} dimensions at once is not supported",
            hint="Group by at most two dimensions.",
        )


def _validate_viability(plan: AnalysisPlan) -> None:
    spec = _OP_REQUIREMENTS[plan.op]

    if spec.get("needs_metrics") and not plan.metrics:
        raise PlanError(
            ValidationStage.VIABILITY,
            f"op {plan.op.value!r} requires at least one metric",
            hint="Add the metrics the question is about.",
        )
    if spec.get("needs_axis") and plan.bin is None and plan.time_grain is None:
        raise PlanError(
            ValidationStage.VIABILITY,
            f"op {plan.op.value!r} requires an axis",
            hint="Set `bin` to trend over a metric, or `time_grain` to trend over time.",
        )
    min_cohorts = spec.get("min_cohorts", 0)
    if min_cohorts and len(plan.cohorts) < min_cohorts:
        raise PlanError(
            ValidationStage.VIABILITY,
            f"op {plan.op.value!r} requires {min_cohorts} cohorts, got {len(plan.cohorts)}",
            hint="Define cohorts, e.g. failed vs healthy.",
        )
    names = [c.name for c in plan.cohorts]
    if len(names) != len(set(names)):
        raise PlanError(
            ValidationStage.VIABILITY,
            "cohort names must be unique — they namespace the evidence slots",
        )


def _validate_op_specific(plan: AnalysisPlan) -> None:
    spec = _OP_REQUIREMENTS[plan.op]
    for key in spec.get("required_params", []):
        if key not in plan.params:
            raise PlanError(
                ValidationStage.OP_SPECIFIC,
                f"op {plan.op.value!r} requires params.{key}",
            )

    if plan.op is OpName.COUNTERFACTUAL:
        changes = plan.params.get("changes")
        if not isinstance(changes, dict) or not changes:
            raise PlanError(
                ValidationStage.OP_SPECIFIC,
                "counterfactual params.changes must be a non-empty mapping",
                hint='e.g. {"torque_nm": -5.0}',
            )
        metrics = metric_index()
        for field, delta in changes.items():
            if field not in metrics:
                raise PlanError(
                    ValidationStage.OP_SPECIFIC,
                    f"cannot vary unknown metric {field!r}",
                    hint=_suggest(field, set(metrics)).strip(),
                )
            if not isinstance(delta, (int, float)):
                raise PlanError(
                    ValidationStage.OP_SPECIFIC,
                    f"change for {field!r} must be numeric, got {type(delta).__name__}",
                )

    if plan.op is OpName.SQL_EXPLORE:
        sql = str(plan.params.get("sql", ""))
        lowered = sql.lower()
        banned = ("insert", "update", "delete", "drop", "create", "alter",
                  "attach", "copy", "install", "load", "pragma", "export")
        hit = next((w for w in banned if w in lowered), None)
        if hit:
            raise PlanError(
                ValidationStage.OP_SPECIFIC,
                f"sql_explore is read-only; {hit!r} is not permitted",
            )
        if ";" in sql.rstrip().rstrip(";"):
            raise PlanError(
                ValidationStage.OP_SPECIFIC,
                "sql_explore accepts a single statement",
            )


def validate_plan(plan: AnalysisPlan) -> AnalysisPlan:
    """Run every stage in order. Raises PlanError on the first failure.

    Structural validation already happened in Pydantic by the time we get here.
    """
    _validate_vocabulary(plan)
    _validate_dimensional(plan)
    _validate_cardinality(plan)
    _validate_viability(plan)
    _validate_op_specific(plan)
    return plan


def parse_plan(payload: dict[str, Any] | str) -> AnalysisPlan:
    """Parse and fully validate. The single entry point for planner output."""
    import json

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise PlanError(
                ValidationStage.STRUCTURAL, f"not valid JSON: {exc}"
            ) from exc
    try:
        plan = AnalysisPlan.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError
        raise PlanError(ValidationStage.STRUCTURAL, str(exc)) from exc
    return validate_plan(plan)
