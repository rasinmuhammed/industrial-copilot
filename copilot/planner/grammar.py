"""Tier 1: deterministic natural language to Analysis Plan.

Covers the question vocabulary a plant actually uses, in about a millisecond,
with no model and no credentials. Everything it handles is a question that never
reaches an LLM — which is where most of the latency budget is won.

It is deliberately *conservative*: it emits a confidence score and escalates
rather than guessing. A wrong plan produced quickly is worse than a slow correct
one, so an ambiguous match is a miss, not a coin flip.

The output is an `AnalysisPlan`, exactly like the LLM tier produces, so both go
through the same validation and the same executor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from copilot.ir import AnalysisPlan, Binning, Cohort, Filter, OpName
from copilot.knowledge import dimension_index, metric_index, synonym_map
from copilot.session import SessionState

__all__ = ["GrammarMatch", "plan_from_text", "CONFIDENCE_THRESHOLD"]

# Below this the router escalates rather than trusting the match.
CONFIDENCE_THRESHOLD = 0.55

FAILED_COHORT = Cohort(name="failed", filters=[Filter(field="failure", op="=", value=1)])
HEALTHY_COHORT = Cohort(name="healthy", filters=[Filter(field="failure", op="=", value=0)])

# Intent signals, most specific first. Order matters: "why did X fail" must beat
# the generic "fail" signal that would otherwise route to `rate`.
_INTENT_PATTERNS: list[tuple[OpName, float, re.Pattern[str]]] = [
    (OpName.DATA_QUALITY, 0.95, re.compile(
        r"\b(can i trust|data quality|quality of (the )?data|any (issues|problems) with"
        r"|is the data|label integrity|are the (thresholds|rules) still)\b")),
    (OpName.COUNTERFACTUAL, 0.95, re.compile(
        r"\b(what if|if (we|i) (reduce|reduced|cut|increase|increased|raise|lower)"
        r"|suppose we|were we to)\b")),
    (OpName.FORECAST, 0.9, re.compile(
        r"\b(when will|how long until|how many cycles|time to (failure|crossing)"
        r"|remaining useful life|when do(es)? .* cross|lead time|predict when)\b")),
    (OpName.ENVELOPE, 0.9, re.compile(
        r"\b(safe (operating|range|window|torque|speed)|operating (window|envelope)"
        r"|what should i (do|set|change)|how (do|can) i fix|recommend"
        r"|within limits|acceptable range)\b")),
    (OpName.ROOT_CAUSE, 0.9, re.compile(
        r"\b(why did|root cause|what caused|what causes|causes? of|what made .* fail"
        r"|reason for (the )?fail|diagnose|what went wrong|attribut)\b")),
    (OpName.DRIVERS, 0.85, re.compile(
        r"\b(what (drives|separates|distinguishes|predicts)|which (variables|factors|"
        r"parameters|metrics) (best )?(separate|distinguish|drive|matter|predict)"
        r"|most important (variable|factor)|biggest (driver|factor))\b")),
    (OpName.COMPARE, 0.85, re.compile(
        r"\b(compare|versus|vs\.?|difference between|contrast"
        r"|how do .* differ|failed .* (and|versus|vs) .* (did not|didn't|healthy))\b")),
    # "by shift" on its own is a grouped rate. Treating it as a trend axis
    # misroutes "show failure rate by shift", which is a breakdown, not a series.
    (OpName.TREND, 0.8, re.compile(
        r"\b(trend|over time|as .* (increases|rises|grows)|vary with|varies with"
        r"|relationship (between|with)|change(s)? with|as a function of"
        r"|(trend|series|evolution) (by|per) (hour|day|shift))\b")),
    (OpName.RECORDS, 0.8, re.compile(
        r"\b(show me|list|give me examples|which (cycles|rows|records)|examples of"
        r"|closest to (failing|failure))\b")),
    (OpName.RATE, 0.75, re.compile(
        r"\b(failure rate|how often|how (many|much) fail|rate of failure"
        r"|percentage .* fail|proportion .* fail|breakdown (of|by)|by (variant|type)"
        r"|more failures)\b")),
    (OpName.DESCRIBE, 0.7, re.compile(
        r"\b(what (are|is) (the )?(typical|normal|usual|average)|describe|summar[iy]"
        r"|operating conditions|what.s (been )?happening|conditions (on|for|at)"
        r"|how (hot|fast|much))\b")),
]

# Follow-ups that mutate the previous plan rather than starting over.
_FOLLOWUP = re.compile(
    r"^\s*(and |but |ok |okay )?(what about|how about|and for|same for|now)\b", re.IGNORECASE
)
_DRILL_DOWN = re.compile(
    r"\b(show me (them|those|these)|those (rows|cycles)|the failures|drill|which ones)\b",
    re.IGNORECASE,
)

_VARIANT = re.compile(r"\b([LMH])\b(?:\s*(?:variant|type|quality|grade))?")
_VARIANT_WORD = re.compile(r"\b(low|medium|high)[- ]?(?:quality|grade|variant|type)\b", re.I)
_UDI = re.compile(r"\b(?:udi|cycle|row|record)\s*#?\s*(\d{1,5})\b", re.IGNORECASE)
_MACHINE = re.compile(r"\b(?:machine|asset|unit)\s*#?\s*([LMH]-\d{2})\b", re.IGNORECASE)
_TIME_GRAIN = re.compile(r"\bby (hour|day|shift)\b|\bper (hour|day|shift)\b", re.IGNORECASE)
_NUMBER_WITH_UNIT = re.compile(
    r"([-+]?\d+(?:\.\d+)?)\s*(nm|n·m|rpm|k|w|min|minutes?|%)\b", re.IGNORECASE
)
_CHANGE = re.compile(
    r"\b(reduce|reduced|cut|lower|decrease|increase|increased|raise|add)\b[^.?]*?"
    r"([-+]?\d+(?:\.\d+)?)\s*(nm|n·m|rpm|k|min|minutes?)\b",
    re.IGNORECASE,
)

_VARIANT_WORDS = {"low": "L", "medium": "M", "high": "H"}

# A comparative claim about a NAMED group — "why do H variants fail more" — is a
# premise, and it can be false. Gate 1 previously only tested monotone premises
# over a binned axis, so a false categorical claim sailed through and the system
# answered a different question entirely.
_CATEGORICAL_CLAIM = re.compile(
    r"\b(fail|failing|failure|break|breaking|breakdown)\w*\b[^.?]{0,40}?"
    r"\b(more|most|worse|worst|higher|highest|often|frequently)\b"
    r"|\b(more|most|worse|worst|higher|highest)\b[^.?]{0,40}?"
    r"\b(fail|failing|failure|break|breaking|breakdown)\w*\b",
    re.IGNORECASE,
)


def _categorical_premise(text: str) -> dict[str, str] | None:
    """Extract a claim of the form "<group> fails more" if one is asserted."""
    if not _CATEGORICAL_CLAIM.search(text):
        return None
    variant = None
    if (m := _VARIANT_WORD.search(text)) is not None:
        variant = _VARIANT_WORDS[m.group(1).lower()]
    elif re.search(r"\b(variants?|types?|grades?)\b", text) and (
        v := _VARIANT.search(text.upper())
    ):
        variant = v.group(1)
    if variant:
        return {"field": "product_type", "value": variant, "direction": "more"}
    for shift in ("a", "b", "c"):
        if re.search(rf"\bshift {shift}\b", text, re.IGNORECASE):
            return {"field": "shift", "value": shift.upper(), "direction": "more"}
    return None
_UNIT_TO_METRIC = {
    "nm": "torque_nm", "n·m": "torque_nm",
    "rpm": "rotational_speed_rpm",
    "min": "tool_wear_min", "minute": "tool_wear_min", "minutes": "tool_wear_min",
}


@dataclass(slots=True)
class GrammarMatch:
    plan: AnalysisPlan | None
    confidence: float
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.plan is not None and self.confidence >= CONFIDENCE_THRESHOLD


def plan_from_text(question: str, state: SessionState | None = None) -> GrammarMatch:
    """Attempt a deterministic plan. Returns low confidence rather than guessing."""
    text = question.strip().lower()
    if not text:
        return GrammarMatch(None, 0.0, "empty question")

    # --- follow-ups reuse the previous plan --------------------------------
    if state is not None and state.last_plan is not None:
        followup = _try_followup(text, state)
        if followup is not None:
            return followup

    # A comparative claim about a named group must be TESTED, whatever else the
    # question asks for. Answering around an unchallenged false premise is the
    # failure mode this gate exists to prevent.
    claim = _categorical_premise(text)
    if claim is not None:
        plan = AnalysisPlan(
            op=OpName.RATE,
            group_by=[claim["field"]],
            params={"premise": claim},
            verify_premise=True,
        )
        return GrammarMatch(plan, 0.9, f"testing the claim that {claim['value']} fails more")

    op, confidence = _match_intent(text)
    if op is None:
        return GrammarMatch(None, 0.0, "no intent pattern matched")

    metrics = _extract_metrics(text)
    filters = _extract_filters(text, state, op)

    try:
        plan = _build(op, text, metrics, filters, state)
    except _Unbuildable as exc:
        return GrammarMatch(None, confidence * 0.4, str(exc))

    # Evidence that we understood specifics, not just the verb.
    if metrics:
        confidence = min(1.0, confidence + 0.05)
    if filters:
        confidence = min(1.0, confidence + 0.05)
    return GrammarMatch(plan, confidence, f"matched {op.value}")


class _Unbuildable(Exception):
    """The intent was recognised but the plan cannot be completed from text."""


def _match_intent(text: str) -> tuple[OpName | None, float]:
    for op, confidence, pattern in _INTENT_PATTERNS:
        if pattern.search(text):
            return op, confidence
    return None, 0.0


def _phrase_pattern(phrase: str) -> str:
    """Match a synonym, tolerating a trailing plural on the final word.

    "rotational speeds" must resolve to rotational_speed_rpm. Without this the
    planner silently finds no metric and falls back to a much worse plan.
    """
    return rf"\b{re.escape(phrase)}s?\b"


def _extract_metrics(text: str) -> list[str]:
    """Longest-match synonym resolution over the semantic layer."""
    found: list[str] = []
    synonyms = synonym_map()
    for phrase in sorted(synonyms, key=len, reverse=True):
        if len(phrase) < 3:
            continue
        kind, name = synonyms[phrase]
        if kind != "metric" or name in found:
            continue
        if re.search(_phrase_pattern(phrase), text):
            found.append(name)
    # `failure` is a label, not an operating metric; it is never a describe target.
    return [m for m in found if m != "failure"]


def _extract_dimensions(text: str) -> list[str]:
    found: list[str] = []
    synonyms = synonym_map()
    for phrase in sorted(synonyms, key=len, reverse=True):
        if len(phrase) < 3:
            continue
        kind, name = synonyms[phrase]
        if kind != "dimension" or name in found:
            continue
        if re.search(_phrase_pattern(phrase), text):
            found.append(name)
    return found


# Ops that compute statistics OVER a population. Filtering them to failures
# only would answer "what is the failure rate among failures?" — always 100%.
_POPULATION_OPS = frozenset(
    {OpName.RATE, OpName.TREND, OpName.DRIVERS, OpName.COMPARE, OpName.DATA_QUALITY}
)

# Ops that are about one cycle. A `udi` focus may only be inherited into these;
# carrying a single-cycle focus into an aggregate collapses it to one row.
_SINGLE_ROW_OPS = frozenset(
    {OpName.ROOT_CAUSE, OpName.RECORDS, OpName.ENVELOPE, OpName.FORECAST, OpName.DESCRIBE}
)


def _extract_filters(text: str, state: SessionState | None, op: OpName) -> list[Filter]:
    filters: list[Filter] = []

    if (m := _UDI.search(text)) is not None:
        filters.append(Filter(field="udi", op="=", value=int(m.group(1))))
    if (m := _MACHINE.search(text)) is not None:
        filters.append(Filter(field="machine_id", op="=", value=m.group(1).upper()))

    variant = None
    if (m := _VARIANT_WORD.search(text)) is not None:
        variant = _VARIANT_WORDS[m.group(1).lower()]
    elif (m := _VARIANT.search(text.upper())) is not None:
        # Only trust a bare letter when the sentence is talking about variants.
        if re.search(r"\b(variants?|types?|quality|grades?)\b", text):
            variant = m.group(1)
    if variant:
        filters.append(Filter(field="product_type", op="=", value=variant))

    if op not in _POPULATION_OPS and re.search(r"\bfailed\b|\bfailures?\b", text):
        filters.append(Filter(field="failure", op="=", value=1))

    if not filters and state is not None and state.focus is not None:
        inherited = state.focus.as_filter()
        if inherited is not None and (
            inherited.field != "udi" or op in _SINGLE_ROW_OPS
        ):
            filters.append(inherited)
    return filters


def _build(
    op: OpName,
    text: str,
    metrics: list[str],
    filters: list[Filter],
    state: SessionState | None,
) -> AnalysisPlan:
    dimensions = _extract_dimensions(text)
    grain = None
    if (m := _TIME_GRAIN.search(text)) is not None:
        grain = (m.group(1) or m.group(2)).lower()

    if op is OpName.COMPARE:
        return AnalysisPlan(
            op=op,
            cohorts=[FAILED_COHORT, HEALTHY_COHORT],
            metrics=metrics or _DEFAULT_COMPARE_METRICS,
            effect_size="cohens_d",
        )

    if op is OpName.DESCRIBE:
        return AnalysisPlan(op=op, metrics=metrics or _DEFAULT_DESCRIBE_METRICS, filters=filters)

    if op is OpName.RATE:
        binning = _binning_for(text, metrics)
        group_by = [d for d in dimensions if d not in {"udi", "failure_mode"}]
        if binning is None and not group_by and grain is None:
            # "what's the failure rate?" — a bare overall rate is a valid plan.
            return AnalysisPlan(op=op, filters=filters)
        return AnalysisPlan(
            op=op,
            filters=filters,
            group_by=[] if binning else group_by[:1],
            bin=binning,
            time_grain=grain,  # type: ignore[arg-type]
        )

    if op is OpName.TREND:
        binning = _binning_for(text, metrics)
        if binning is None and grain is None:
            raise _Unbuildable("a trend needs an axis and none was named")
        return AnalysisPlan(
            op=op, filters=filters, bin=binning, time_grain=grain,  # type: ignore[arg-type]
            metrics=[m for m in metrics if binning is None or m != binning.field],
        )

    if op is OpName.DRIVERS:
        return AnalysisPlan(op=op, filters=[f for f in filters if f.field != "failure"],
                            metrics=metrics)

    if op is OpName.ROOT_CAUSE:
        return AnalysisPlan(op=op, filters=filters)

    if op is OpName.RECORDS:
        order = "closest_to_failure" if "closest" in text else "udi"
        return AnalysisPlan(op=op, filters=filters, limit=20, params={"order": order})

    if op is OpName.DATA_QUALITY:
        return AnalysisPlan(op=op)

    if op is OpName.COUNTERFACTUAL:
        changes = _extract_changes(text)
        if not changes:
            raise _Unbuildable("no parameter change could be read from the question")
        return AnalysisPlan(op=op, filters=filters, params={"changes": changes})

    if op in {OpName.ENVELOPE, OpName.FORECAST}:
        params = _extract_operating_point(text)
        if not params and not filters:
            raise _Unbuildable("no operating point or cohort was given")
        return AnalysisPlan(op=op, filters=filters, params=params)

    raise _Unbuildable(f"no builder for {op.value}")


_DEFAULT_COMPARE_METRICS = [
    "torque_nm", "rotational_speed_rpm", "tool_wear_min", "temp_delta_k", "power_w",
]
_DEFAULT_DESCRIBE_METRICS = [
    "air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min",
]


def _binning_for(text: str, metrics: list[str]) -> Binning | None:
    """Pick the axis metric for a binned analysis.

    Prefers a metric named after 'with'/'vs'/'by', which is where the axis
    almost always appears: "how does failure rate vary WITH tool wear".
    """
    if not metrics:
        return None
    axis = metrics[0]
    for match in re.finditer(r"\b(?:with|by|versus|vs\.?|against|across)\s+([a-z ]+)", text):
        tail = match.group(1)
        for metric in metrics:
            synonyms = [metric, *metric_index()[metric].get("synonyms", [])]
            if any(re.search(rf"\b{re.escape(s)}\b", tail) for s in synonyms):
                axis = metric
                break
    method = "width" if re.search(r"\b(equal|even|uniform)\b", text) else "quantile"
    return Binning(field=axis, method=method, bins=5)  # type: ignore[arg-type]


def _extract_changes(text: str) -> dict[str, float]:
    changes: dict[str, float] = {}
    for verb, amount, unit in _CHANGE.findall(text):
        metric = _UNIT_TO_METRIC.get(unit.lower().rstrip("s"))
        if metric is None:
            continue
        magnitude = abs(float(amount))
        negative = verb.lower() in {"reduce", "reduced", "cut", "lower", "decrease"}
        changes[metric] = -magnitude if negative else magnitude
    return changes


def _extract_operating_point(text: str) -> dict[str, float | str]:
    params: dict[str, float | str] = {}
    for amount, unit in _NUMBER_WITH_UNIT.findall(text):
        key = _UNIT_TO_METRIC.get(unit.lower().rstrip("s"))
        if key is not None:
            params[key] = float(amount)
    if (m := _VARIANT_WORD.search(text)) is not None:
        params["product_type"] = _VARIANT_WORDS[m.group(1).lower()]
    elif re.search(r"\b(variants?|types?|grades?)\b", text) and (
        v := _VARIANT.search(text.upper())
    ):
        params["product_type"] = v.group(1)
    return params


def _try_followup(text: str, state: SessionState) -> GrammarMatch | None:
    """Mutate the previous plan instead of re-planning.

    "What about the H variants?" should change one filter and nothing else — that
    keeps the comparison valid and usually keeps the question off the LLM tier.
    """
    assert state.last_plan is not None
    previous = state.last_plan

    if _DRILL_DOWN.search(text):
        return GrammarMatch(
            AnalysisPlan(
                op=OpName.RECORDS,
                filters=list(previous.filters) or [Filter(field="failure", op="=", value=1)],
                limit=20,
                params={"order": "closest_to_failure"},
            ),
            0.85,
            "drill-down into the previous result",
        )

    if not _FOLLOWUP.match(text):
        return None

    variant = None
    if (m := _VARIANT_WORD.search(text)) is not None:
        variant = _VARIANT_WORDS[m.group(1).lower()]
    elif (m := re.search(r"\b([LMH])\b", text.upper())) is not None:
        variant = m.group(1)

    if variant is None:
        return None

    kept = [f for f in previous.filters if f.field != "product_type"]
    kept.append(Filter(field="product_type", op="=", value=variant))
    mutated = previous.model_copy(update={"filters": kept})
    return GrammarMatch(mutated, 0.9, f"reused previous plan with product_type = {variant}")
