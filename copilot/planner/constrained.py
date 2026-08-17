"""Make an invalid plan unreachable instead of merely rejected.

WHAT CHANGED, AND WHY IT MATTERS
--------------------------------
The closed vocabulary has always been the anti-hallucination mechanism: a plan
naming a column that does not exist is refused by the validator. That is
detection. It works, and it is one step too late - the model has already spent
its latency budget producing something unusable, and the repair path is another
round trip.

Constrained decoding moves the same vocabulary from the validator to the
SAMPLER. A JSON schema whose every field is an enum drawn from the semantic
layer means the token that would spell a nonexistent column is masked before it
can be chosen. The model cannot name `bearing_temperature` because those tokens
are not reachable from that position in the grammar.

Detection becomes prevention, and plan validity stops being a measured rate and
becomes a property of the construction.

THE EVIDENCE THIS IS THE RIGHT FIX
----------------------------------
The first distilled planner scored 14.2% exact match against the grammar tier's
98.4%. The failure was not comprehension - it was FORMAT. Its errors were
overwhelmingly right-content-wrong-slot:

    want  counterfactual|-|-|-|-|rotational_speed_rpm-8
    got   counterfactual|-|-|-|rotational_speed_rpm-8|-

The compact notation is positional, so the model had to count pipe separators,
which is close to the worst thing to ask of a transformer - the same weakness
behind every miscounted letter in a word. It understood the question and put the
answer one field to the left.

Two changes follow. The output becomes LABELLED rather than positional, so
nothing has to be counted; and it becomes SCHEMA-CONSTRAINED, so the shape is
guaranteed rather than hoped for.

A first probe of the existing model under a partial schema showed exactly the
predicted behaviour: the JSON structure was correct and it invented
`failure_rate_over_time` and `failure_severity` in the one field left
unconstrained. Enumerate that field and the invention is impossible.

WHERE THIS SITS
---------------
Still the last tier. 93% of questions never reach a model, and 3 ms of grammar
beats any model on latency. This exists for the long tail - the phrasings nobody
anticipated - and for the honest reason that a real plant asks stranger
questions than one dataset can contain.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from copilot.ir import MAX_GROUPS, OpName
from copilot.knowledge import dimension_index, metric_index

__all__ = ["plan_schema", "schema_json", "ConstrainedPlanner"]


@lru_cache(maxsize=1)
def plan_schema() -> dict[str, Any]:
    """A JSON schema for an Analysis Plan, generated from the semantic layer.

    Generated, not written. A hand-maintained schema is a second declaration of
    the vocabulary that drifts from the first - the duplication this project
    removed from physics.py, reappearing in a new place. Adding a metric to the
    YAML extends what the model may emit, with no code change and no way for the
    two to disagree.
    """
    metrics = sorted(metric_index())
    dimensions = sorted(d for d in dimension_index() if d != "failure_mode")
    ops = [op.value for op in OpName]

    metric_enum = {"type": "string", "enum": metrics}
    dimension_enum = {"type": "string", "enum": dimensions}
    field_enum = {"type": "string", "enum": sorted(set(metrics) | set(dimensions))}

    return {
        "type": "object",
        "properties": {
            # `refuse` is part of the vocabulary, and it has to be.
            #
            # Constraining the output space removes the model's ability to
            # decline: if every reachable token spells a valid plan, "I cannot
            # answer that" is unreachable, and the model emits the NEAREST valid
            # plan instead. Measured on the first probe - asked for bearing
            # temperature, a sensor this process does not have, it confidently
            # produced a well-formed plan describing ambient air.
            #
            # That is the silent-substitution failure this project spent
            # considerable effort eliminating, reintroduced by the very
            # mechanism meant to make output safer. A constrained decoder needs
            # an explicit escape hatch or it will always answer.
            "op": {"type": "string", "enum": [*ops, "refuse"]},
            "refuse_reason": {
                "type": "string",
                "enum": [
                    "no such measurement",
                    "not answerable from this data",
                    "question not understood",
                ],
            },
            "metrics": {"type": "array", "items": metric_enum, "maxItems": 8},
            "group_by": {"type": "array", "items": dimension_enum, "maxItems": 2},
            "filters": {
                "type": "array",
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "field": field_enum,
                        "op": {"type": "string",
                               "enum": ["=", "!=", "<", "<=", ">", ">="]},
                        # Deliberately loose: a filter value is data, not
                        # vocabulary. "L" and 9016 are both legitimate, and
                        # enumerating every cycle id would be absurd.
                        "value": {"type": ["string", "number", "boolean"]},
                    },
                    "required": ["field", "op", "value"],
                },
            },
            "bin": {
                "type": "object",
                "properties": {
                    "field": metric_enum,
                    "method": {"type": "string", "enum": ["quantile", "width"]},
                    "bins": {"type": "integer", "minimum": 2, "maximum": MAX_GROUPS},
                },
                "required": ["field"],
            },
            "effect_size": {"type": "string",
                            "enum": ["cohens_d", "rate_ratio", "risk_diff"]},
            "time_grain": {"type": "string", "enum": ["hour", "day", "shift"]},
        },
        "required": ["op"],
        "additionalProperties": False,
    }


def schema_json() -> str:
    return json.dumps(plan_schema(), separators=(",", ":"))


def system_prompt() -> str:
    """Vocabulary in the prompt as well as the schema.

    The schema makes an invalid name unreachable; the prompt makes the RIGHT
    name likelier. Constraining without telling the model what the fields mean
    produces valid plans that answer the wrong question - structurally sound and
    semantically wrong, which is worse than a refusal.
    """
    metrics = metric_index()
    lines = [
        "Translate the engineer's question into one Analysis Plan as JSON.",
        "Emit only the JSON object.",
        "",
        "metrics:",
    ]
    for name in sorted(metrics):
        spec = metrics[name]
        lines.append(f"  {name} - {spec.get('label', name)} ({spec.get('unit', '')})")
    lines += ["", "dimensions:"]
    for name, spec in sorted(dimension_index().items()):
        if name == "failure_mode":
            continue
        lines.append(f"  {name} - {spec.get('label', name)}")
    lines += [
        "",
        "If the question asks about a quantity not listed above, or cannot be",
        "answered from these channels, emit {\"op\": \"refuse\"} with a reason.",
        "Answering about a different sensor than the one asked about is the",
        "worst available outcome.",
        "",
        "ops: describe (summarise metrics) · rate (failure rate, optionally",
        "grouped or binned) · compare (two cohorts) · trend (over an axis) ·",
        "drivers (what separates failures) · root_cause (attribute a failure) ·",
        "counterfactual (what if a parameter changed) · envelope (safe window) ·",
        "forecast (time to crossing) · records (example rows) ·",
        "data_quality (label integrity)",
    ]
    return "\n".join(lines)


class ConstrainedPlanner:
    """Fourth-tier planner. Emits a plan the validator cannot reject on shape.

    Deliberately thin. It owns the schema and the transport; the resulting plan
    still goes through `parse_plan`, because the schema constrains SHAPE and
    VOCABULARY while the validator additionally enforces dimensional
    consistency, physical domains, cardinality and viability. A plan can be
    perfectly well-formed and still ask to add kelvin to newton-metres.
    """

    name = "constrained"

    def __init__(
        self,
        model: str = "margin-planner",
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def propose(self, question: str) -> dict[str, Any]:
        """Return a raw plan dict. Raises on transport failure, never on shape."""
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": question},
            ],
            "stream": False,
            "format": plan_schema(),
            # Greedy. A planner that must emit one canonical plan has no
            # business sampling: two runs of the same question must agree, or
            # the answer is not auditable.
            "options": {"temperature": 0, "top_k": 1, "top_p": 1.0,
                        "num_predict": 256, "seed": 0},
        }).encode()

        request = urllib.request.Request(
            self.endpoint, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        return self._normalise(json.loads(payload["message"]["content"]))

    @staticmethod
    def _normalise(raw: dict[str, Any]) -> dict[str, Any]:
        """Read a half-refusal as a refusal.

        Observed: asked for bearing temperature, the model set
        `refuse_reason: "no such measurement"` and left `op: describe`. It knew
        the question was unanswerable and answered anyway - the two fields
        disagreeing, with the confident one winning.

        A stated reason to decline outranks a plan it contradicts. Failing safe
        here costs a refusal that might have been answerable; failing open costs
        a confident answer about the wrong sensor, which is far worse.
        """
        if raw.get("refuse_reason") and raw.get("op") != "refuse":
            return {"op": "refuse", "refuse_reason": raw["refuse_reason"]}
        return raw
