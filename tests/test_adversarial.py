"""Adversarial suite: actively try to make the system state something false.

Every case here was a real hole found by probing, not a hypothetical. The
standard is not "usually right" — it is that a false statement should be
structurally unreachable, and where it is not, the system should decline.
"""

from __future__ import annotations

import pytest

from copilot.engine import Engine
from copilot.ir import PlanError, ValidationStage, parse_plan
from copilot.planner.exemplars import ExemplarStore, polarity
from copilot.session import SessionState

ROOT_CAUSE = {"op": "root_cause", "filters": [{"field": "udi", "op": "=", "value": 9016}]}


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine.build()


class TestCategoricalFalsePremise:
    """H fails at 2.09%, L at 3.92%. "H fails more" is false.

    Before this gate the question routed to `describe` and was answered with
    air-temperature statistics while the premise went unchallenged — a confident
    answer to a question nobody asked.
    """

    @pytest.mark.parametrize("question", [
        "Why do high quality variants fail more often?",
        "Why is the H variant failing more than L?",
        "Why do H variants break more frequently?",
    ])
    def test_false_categorical_claim_is_refuted(self, engine, question):
        answer = engine.ask(question, SessionState())
        assert not answer.refused
        assert answer.bundle.slots["premise.verdict"].value == "refuted"
        assert any(w.code == "premise_refuted" for w in answer.bundle.warnings)
        assert answer.verified

    def test_refutation_names_the_actual_worst_group(self, engine):
        answer = engine.ask("Why do high quality variants fail more?", SessionState())
        slots = answer.bundle.slots
        assert slots["premise.subject"].value == "H"
        assert slots["premise.highest_group"].value == "L"
        assert slots["premise.subject_rate"].value == pytest.approx(2.09, abs=0.01)
        assert slots["premise.highest_rate"].value == pytest.approx(3.92, abs=0.01)

    def test_a_true_claim_is_supported_not_refuted(self, engine):
        """The gate must not cry wolf on a correct premise."""
        answer = engine.ask("Why do low quality variants fail more?", SessionState())
        assert answer.bundle.slots["premise.verdict"].value == "supported"
        assert not any(w.code == "premise_refuted" for w in answer.bundle.warnings)

    def test_a_neutral_question_asserts_nothing(self, engine):
        answer = engine.ask("What is the failure rate by product variant?", SessionState())
        assert "premise.verdict" not in answer.bundle.slots

    def test_the_refutation_leads_the_answer(self, engine):
        """Burying a refutation makes a technically-true answer misleading."""
        answer = engine.ask("Why do high quality variants fail more?", SessionState())
        head = answer.narration.strip().splitlines()[0].lower()
        assert "not supported" in head or "not the group" in head


class TestPhysicalDomain:
    """An impossible input must be refused, not answered.

    A forecast built on negative tool wear is a confident prediction about a
    state that cannot occur — worse than no answer at all.
    """

    @pytest.mark.parametrize("payload", [
        {"op": "forecast", "params": {"tool_wear_min": -50, "torque_nm": 45}},
        {"op": "envelope", "params": {"rotational_speed_rpm": 0, "torque_nm": 45,
                                      "tool_wear_min": 100}},
        {"op": "describe", "metrics": ["torque_nm"],
         "filters": [{"field": "torque_nm", "op": "<", "value": -5}]},
        {"op": "describe", "metrics": ["air_temp_k"],
         "filters": [{"field": "air_temp_k", "op": ">", "value": 5000}]},
    ])
    def test_impossible_values_are_rejected(self, payload):
        with pytest.raises(PlanError) as exc:
            parse_plan(payload)
        assert exc.value.stage is ValidationStage.DOMAIN

    def test_absurd_counterfactual_delta_is_rejected(self):
        """A change larger than the variable's whole range means a misread unit."""
        with pytest.raises(PlanError) as exc:
            parse_plan({"op": "counterfactual", "params": {"changes": {"torque_nm": -999}}})
        assert exc.value.stage is ValidationStage.DOMAIN

    def test_legitimate_values_still_pass(self):
        parse_plan({"op": "forecast", "params": {"tool_wear_min": 150, "torque_nm": 45}})
        parse_plan({"op": "counterfactual", "params": {"changes": {"torque_nm": -5}}})

    def test_rejection_explains_the_physical_limit(self):
        with pytest.raises(PlanError) as exc:
            parse_plan({"op": "forecast", "params": {"tool_wear_min": -50, "torque_nm": 45}})
        assert "cannot be less than" in exc.value.hint


class TestNegation:
    """Bag-of-ngram similarity is nearly blind to negation: "why did cycle 9016
    fail" and "...not fail" score 0.885, well above the reuse threshold."""

    @pytest.mark.parametrize("text,expected", [
        ("why did cycle 9016 fail", False),
        ("why did cycle 9016 not fail", True),
        ("which machines did not fail", True),
        ("show me machines excluding failures", True),
        ("compare failed and healthy", False),
    ])
    def test_polarity_detection(self, text, expected):
        assert polarity(text) is expected

    def test_a_negated_question_does_not_reuse_an_affirmative_plan(self):
        store = ExemplarStore()
        store.record("why did cycle 9016 fail", parse_plan(ROOT_CAUSE), tier="llm")
        assert store.suggest("why did cycle 4045 fail", None) is not None
        assert store.suggest("why did cycle 4045 not fail", None) is None

    def test_polarity_gate_is_symmetric(self):
        store = ExemplarStore()
        store.record("which cycles did not fail", parse_plan({"op": "records"}), tier="llm")
        assert store.suggest("which cycles did not fail today", None) is not None
        assert store.suggest("which cycles failed", None) is None


class TestDegenerateInputs:
    """Every op must abstain or warn rather than emit a plausible number."""

    @pytest.mark.parametrize("payload", [
        {"op": "forecast", "params": {"tool_wear_min": 150, "torque_nm": 0}},
        {"op": "drivers", "filters": [{"field": "tool_wear_min", "op": ">", "value": 9999}]},
        {"op": "trend", "bin": {"field": "tool_wear_min", "method": "quantile", "bins": 5},
         "filters": [{"field": "udi", "op": "=", "value": 100}]},
        {"op": "describe", "metrics": ["torque_nm"],
         "filters": [{"field": "udi", "op": "=", "value": 99999}]},
        {"op": "compare", "metrics": ["torque_nm"],
         "cohorts": [{"name": "a", "filters": [{"field": "product_type", "op": "=", "value": "L"}]},
                     {"name": "b", "filters": [{"field": "tool_wear_min", "op": ">", "value": 9999}]}]},
    ])
    def test_degenerate_input_abstains_and_never_fabricates(self, engine, payload):
        from copilot.ops import execute

        bundle = execute(parse_plan(payload), engine.ctx)
        assert bundle.abstained or any(w.severity == "critical" for w in bundle.warnings)

    def test_no_operator_ever_emits_nan_or_infinity(self, engine):
        """A NaN rendered into prose is a number that means nothing."""
        import math

        from copilot.ops import execute

        payloads = [
            {"op": "forecast", "params": {"tool_wear_min": 150, "torque_nm": 0}},
            {"op": "envelope", "params": {"rotational_speed_rpm": 1, "torque_nm": 1,
                                          "tool_wear_min": 100000}},
            {"op": "rate", "filters": [{"field": "udi", "op": ">", "value": 99999}]},
            {"op": "drivers", "filters": [{"field": "tool_wear_min", "op": ">", "value": 9999}]},
        ]
        for payload in payloads:
            bundle = execute(parse_plan(payload), engine.ctx)
            for sid, slot in bundle.slots.items():
                if isinstance(slot.value, float):
                    assert math.isfinite(slot.value), f"{sid} = {slot.value} in {payload['op']}"


class TestNoFabricationEndToEnd:
    def test_adversarial_questions_never_produce_an_unverified_answer(self, engine):
        """Either a verified answer or an honest refusal. Never a confident guess."""
        hostile = [
            "Why do high quality variants fail more often?",
            "Why did cycle 999999 fail?",
            "What is the vibration signature?",
            "Why did the random failures happen?",
            "Show me failures at negative torque",
            "purple monkey dishwasher",
            "What will happen next Tuesday?",
            "Why did cycle 9016 not fail?",
        ]
        for question in hostile:
            answer = engine.ask(question, SessionState())
            assert answer.refused or answer.verified, f"{question!r} produced an unverified answer"

    def test_every_stated_numeral_traces_to_a_slot(self, engine):
        hostile = [
            "Why do high quality variants fail more often?",
            "Why is the H variant failing more than L?",
            "How does failure rate vary with tool wear?",
            "What if we reduce torque by 5 Nm?",
        ]
        for question in hostile:
            answer = engine.ask(question, SessionState())
            if answer.refused:
                continue
            residual = answer.narration
            permitted = {s.render() for s in answer.bundle.slots.values() if s.value is not None}
            for value in sorted(permitted, key=len, reverse=True):
                residual = residual.replace(value, " ")
            leaked = [t for t in residual.split() if any(c.isdigit() for c in t)]
            assert not leaked, f"{question!r} leaked {leaked}"


class TestTierConsistency:
    """The same question must get the same analysis whichever tier answers it.

    A silent divergence between tiers is the nastiest failure available: the
    first ask refutes a false premise, and an identical repeat quietly does not.
    """

    def test_a_premise_check_survives_a_cache_hit(self, engine):
        question = "Why do high quality variants fail more?"
        first = engine.ask(question, SessionState())
        for _ in range(3):
            engine.ask("Why do low quality variants fail more?", SessionState())
            engine.ask("What is the failure rate by product variant?", SessionState())
        repeat = engine.ask(question, SessionState())

        assert repeat.tier in {"cache", "grammar", "exemplar"}
        assert repeat.bundle.slots["premise.verdict"].value == "refuted"
        assert repeat.bundle.slots["premise.subject"].value == "H"
        assert first.bundle.slots["premise.verdict"].value == \
               repeat.bundle.slots["premise.verdict"].value

    def test_a_premise_cannot_be_tested_against_only_its_own_group(self):
        """Verifying "H is worst" requires the other groups to compare against."""
        with pytest.raises(PlanError) as exc:
            parse_plan({
                "op": "rate", "group_by": ["product_type"],
                "params": {"premise": {"field": "product_type", "value": "H",
                                       "direction": "more"}},
                "filters": [{"field": "product_type", "op": "=", "value": "H"}],
            })
        assert exc.value.stage is ValidationStage.VIABILITY

    def test_a_scoped_breakdown_may_group_and_filter_the_same_field(self):
        """"Failure rate for L variants" wants one number for L. That is a valid
        scoped breakdown, not a defeated comparison — the rule must not overreach."""
        plan = parse_plan({"op": "rate", "group_by": ["product_type"],
                           "filters": [{"field": "product_type", "op": "=", "value": "L"}]})
        assert plan.group_by == ["product_type"]

    def test_scoped_breakdown_survives_a_cache_hit(self, engine):
        first = engine.ask("What is the failure rate for L variants?", SessionState())
        repeat = engine.ask("What is the failure rate for L variants?", SessionState())
        assert first.bundle.provenance.row_count == repeat.bundle.provenance.row_count == 6000

    def test_rebind_drops_a_filter_that_would_collapse_the_grouping(self):
        from copilot.planner.exemplars import plan_shape, rebind
        from copilot.planner.grammar import plan_from_text

        question = "Why do high quality variants fail more?"
        shape = plan_shape(plan_from_text(question).plan)
        rebound = rebind(shape, question, None)
        assert rebound is not None
        assert "product_type" in rebound.group_by
        # The claimed field is not filtered — the comparison needs the other groups.
        assert not any(f.field == "product_type" for f in rebound.filters)
        assert rebound.params["premise"]["value"] == "H"

    def test_repeated_asks_agree_on_every_slot(self, engine):
        """Determinism is a precondition for the replay handle meaning anything."""
        for question in ["What is the overall failure rate?",
                         "Why did cycle 9016 fail?",
                         "How does failure rate vary with tool wear?"]:
            a = engine.ask(question, SessionState())
            b = engine.ask(question, SessionState())
            assert a.bundle.provenance.plan_hash == b.bundle.provenance.plan_hash
            assert {k: v.value for k, v in a.bundle.slots.items()} == \
                   {k: v.value for k, v in b.bundle.slots.items()}
