"""Constrained decoding: make an invalid plan unreachable, not merely rejected.

The first distilled planner scored 14.2% exact match. The failure was FORMAT,
not comprehension — a positional notation made the model count pipe separators
and its errors were overwhelmingly right-content-wrong-slot.

The rebuild changes two things: the target is labelled JSON so nothing is
counted, and the sampler is constrained by a schema generated from the semantic
layer so a token spelling a nonexistent column cannot be chosen.

These tests cover the schema and the client. The model's accuracy is measured in
the notebook against a held-out split; what is asserted here is the property
that does not depend on how well the model trained.
"""

from __future__ import annotations

import json

import pytest

from copilot.ir import OpName, PlanError, parse_plan
from copilot.knowledge import dimension_index, metric_index
from copilot.planner.constrained import ConstrainedPlanner, plan_schema, system_prompt


class TestTheSchemaIsGeneratedNotWritten:
    """A hand-maintained schema is a second declaration of the vocabulary."""

    def test_every_metric_in_the_semantic_layer_is_reachable(self):
        allowed = set(plan_schema()["properties"]["metrics"]["items"]["enum"])
        assert allowed == set(metric_index())

    def test_dimensions_come_from_the_semantic_layer(self):
        allowed = set(plan_schema()["properties"]["group_by"]["items"]["enum"])
        expected = {d for d in dimension_index() if d != "failure_mode"}
        assert allowed == expected

    def test_every_op_is_reachable(self):
        allowed = set(plan_schema()["properties"]["op"]["enum"])
        assert {op.value for op in OpName} <= allowed

    def test_adding_a_metric_to_the_yaml_would_extend_the_schema(self):
        """The point of generating it: no code change, and no way for the two
        declarations to disagree — the duplication removed from physics.py,
        prevented from reappearing here."""
        schema_metrics = plan_schema()["properties"]["metrics"]["items"]["enum"]
        assert "torque_nm" in schema_metrics
        assert "bearing_temp_k" not in schema_metrics


class TestHallucinationIsUnreachable:
    """The closed vocabulary moves from the validator to the sampler."""

    def test_a_nonexistent_column_is_not_in_the_enum(self):
        enum = set(plan_schema()["properties"]["filters"]["items"]
                   ["properties"]["field"]["enum"])
        for invented in ("bearing_temp", "oil_pressure", "vibration_rms",
                         "failure_severity", "failure_rate_over_time"):
            assert invented not in enum

    def test_the_first_probe_invented_exactly_what_was_left_unconstrained(self):
        """Recorded because it is the evidence the approach is right.

        Under a partial schema constraining only `op`, the model produced
        correct JSON structure and invented `failure_rate_over_time` and
        `failure_severity` in the one field left free. Enumerate the field and
        the invention becomes impossible rather than improbable.
        """
        schema = plan_schema()
        assert schema["properties"]["metrics"]["items"].get("enum")
        assert schema["additionalProperties"] is False

    def test_filter_values_are_deliberately_unconstrained(self):
        """A filter VALUE is data, not vocabulary. "L" and 9016 are both
        legitimate, and enumerating every cycle id would be absurd."""
        value = (plan_schema()["properties"]["filters"]["items"]
                 ["properties"]["value"])
        assert "enum" not in value


class TestRefusalMustBeInTheVocabulary:
    """Constraining the output space removes the ability to decline."""

    def test_refuse_is_a_reachable_op(self):
        """If every reachable token spells a valid plan then "I cannot answer
        that" is unreachable, and the model emits the NEAREST valid plan.

        Measured on the first probe: asked for bearing temperature, a sensor
        this process does not have, the constrained model confidently described
        ambient air — the silent-substitution failure reintroduced by the very
        mechanism meant to make output safer.
        """
        assert "refuse" in plan_schema()["properties"]["op"]["enum"]
        assert plan_schema()["properties"]["refuse_reason"]["enum"]

    def test_the_prompt_tells_the_model_declining_is_allowed(self):
        """The schema makes refusal reachable; the prompt makes it likely."""
        prompt = system_prompt()
        assert "refuse" in prompt
        assert "worst available outcome" in prompt

    def test_a_half_refusal_is_read_as_a_refusal(self):
        """Observed: the model set refuse_reason and left op as `describe`. It
        knew the question was unanswerable and answered anyway, with the
        confident field winning. A stated reason to decline outranks a plan it
        contradicts."""
        normalise = ConstrainedPlanner._normalise
        out = normalise({"op": "describe", "metrics": ["air_temp_k"],
                         "refuse_reason": "no such measurement"})
        assert out["op"] == "refuse"

    def test_a_plan_without_a_reason_is_left_alone(self):
        normalise = ConstrainedPlanner._normalise
        raw = {"op": "describe", "metrics": ["torque_nm"]}
        assert normalise(raw) == raw


class TestTheSchemaDoesNotReplaceTheValidator:
    """Shape is guaranteed; meaning still has to be checked."""

    def test_a_schema_valid_plan_can_still_be_rejected(self):
        """The schema constrains vocabulary and shape. The validator
        additionally enforces dimensional consistency, physical domains,
        cardinality and viability — a plan can be perfectly well-formed and
        still ask to group ten thousand cycles one at a time.
        """
        well_formed = {"op": "rate", "group_by": ["udi"]}
        enum = plan_schema()["properties"]["group_by"]["items"]["enum"]
        assert "udi" in enum                      # the schema permits it
        with pytest.raises(PlanError):            # the validator does not
            parse_plan(well_formed)

    def test_a_good_plan_passes_both(self):
        plan = {"op": "rate", "group_by": ["product_type"]}
        assert parse_plan(plan).op is OpName.RATE

    def test_the_schema_serialises_for_transport(self):
        from copilot.planner.constrained import schema_json

        assert json.loads(schema_json()) == plan_schema()
