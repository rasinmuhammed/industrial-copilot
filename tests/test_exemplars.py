"""Phase 9a: the verified-exemplar store.

Continual learning with no training step. Every answered question emits
(question, plan, did_it_verify?) and that label is free and objective - the plan
either passed the numeric verifier or it did not.
"""

from __future__ import annotations

import numpy as np
import pytest

from copilot.engine import Engine
from copilot.ir import parse_plan
from copilot.planner.cache import PlanCache
from copilot.planner.exemplars import (
    EXACT_THRESHOLD,
    REUSE_THRESHOLD,
    ExemplarStore,
    HashingEmbedder,
    plan_shape,
    rebind,
)
from copilot.session import SessionState

ROOT_CAUSE_9016 = {"op": "root_cause", "filters": [{"field": "udi", "op": "=", "value": 9016}]}
COMPARE = {
    "op": "compare",
    "cohorts": [
        {"name": "failed", "filters": [{"field": "failure", "op": "=", "value": 1}]},
        {"name": "healthy", "filters": [{"field": "failure", "op": "=", "value": 0}]},
    ],
    "metrics": ["overstrain_min_nm", "tool_wear_min"],
    "effect_size": "cohens_d",
}


class TestEmbedder:
    def test_vectors_are_unit_length(self):
        v = HashingEmbedder().embed("why did cycle 9016 fail")
        assert float(v @ v) == pytest.approx(1.0, abs=1e-5)

    def test_entity_differences_do_not_separate_questions(self):
        e = HashingEmbedder()
        sim = float(e.embed("why did cycle 9016 fail") @ e.embed("why did cycle 4045 fail"))
        assert sim > EXACT_THRESHOLD

    def test_paraphrases_are_near(self):
        e = HashingEmbedder()
        sim = float(
            e.embed("what is the failure rate by variant")
            @ e.embed("failure rate broken down by product type")
        )
        assert sim > REUSE_THRESHOLD

    def test_unrelated_questions_are_far(self):
        e = HashingEmbedder()
        sim = float(e.embed("why did cycle 9016 fail") @ e.embed("what is the average torque"))
        assert sim < 0.4

    def test_embedding_is_deterministic_within_a_process(self):
        e = HashingEmbedder()
        assert np.allclose(e.embed("failure rate by shift"), e.embed("failure rate by shift"))


class TestPlanShape:
    def test_shape_drops_entity_filters(self):
        shape = plan_shape(parse_plan(ROOT_CAUSE_9016))
        assert shape["op"] == "root_cause"
        assert "filters" not in shape

    def test_shape_keeps_the_analysis(self):
        shape = plan_shape(parse_plan(COMPARE))
        assert shape["metrics"] == ["overstrain_min_nm", "tool_wear_min"]
        assert shape["effect_size"] == "cohens_d"

    def test_shape_drops_raw_sql_but_keeps_op_config(self):
        sql = plan_shape(parse_plan({"op": "sql_explore",
                                     "params": {"sql": "SELECT 1 FROM observations"}}))
        assert "sql" not in sql.get("params", {})
        records = plan_shape(parse_plan({"op": "records",
                                         "params": {"order": "closest_to_failure"}}))
        assert records["params"]["order"] == "closest_to_failure"

    def test_rebind_takes_entities_from_the_new_question(self):
        shape = plan_shape(parse_plan(ROOT_CAUSE_9016))
        plan = rebind(shape, "why did cycle 4045 fail", None)
        assert plan is not None
        assert any(f.field == "udi" and f.value == 4045 for f in plan.filters)


class TestExemplarStore:
    def test_records_and_retrieves(self):
        store = ExemplarStore()
        assert store.record("why did cycle 9016 fail", parse_plan(ROOT_CAUSE_9016), tier="llm")
        assert len(store) == 1
        result = store.suggest("why did cycle 4045 fail", None)
        assert result is not None
        plan, score, _ = result
        assert plan.op.value == "root_cause"
        assert score > REUSE_THRESHOLD

    def test_retrieved_plan_points_at_the_new_entity(self):
        """The exemplar supplies the analysis; the question supplies the row."""
        store = ExemplarStore()
        store.record("why did cycle 9016 fail", parse_plan(ROOT_CAUSE_9016), tier="llm")
        plan, _, _ = store.suggest("why did cycle 2750 fail", None)
        assert any(f.field == "udi" and f.value == 2750 for f in plan.filters)
        assert not any(f.value == 9016 for f in plan.filters)

    def test_unrelated_question_is_declined(self):
        store = ExemplarStore()
        store.record("why did cycle 9016 fail", parse_plan(ROOT_CAUSE_9016), tier="llm")
        assert store.suggest("what is the average torque", None) is None

    def test_duplicate_question_updates_rather_than_appends(self):
        store = ExemplarStore()
        store.record("failure rate by variant", parse_plan({"op": "rate"}), tier="llm")
        added = store.record(
            "failure rate by variant",
            parse_plan({"op": "rate", "group_by": ["product_type"]}),
            tier="llm",
        )
        assert not added and len(store) == 1
        assert store._exemplars[0].shape["group_by"] == ["product_type"]

    def test_usage_is_counted(self):
        store = ExemplarStore()
        store.record("why did cycle 9016 fail", parse_plan(ROOT_CAUSE_9016), tier="llm")
        store.suggest("why did cycle 4045 fail", None)
        store.suggest("why did cycle 2750 fail", None)
        assert store._exemplars[0].uses == 2
        assert store.hit_rate == 1.0

    def test_capacity_evicts_the_least_used(self):
        store = ExemplarStore(capacity=2)
        for i, q in enumerate(["rate by variant", "average torque here",
                               "which cycles are closest to failing"]):
            store.record(q, parse_plan({"op": "rate"}), tier="llm")
        assert len(store) == 2

    def test_round_trips_through_disk(self, tmp_path):
        path = tmp_path / "exemplars.jsonl"
        store = ExemplarStore(path=path)
        store.record("why did cycle 9016 fail", parse_plan(ROOT_CAUSE_9016), tier="llm")
        store.save()

        reloaded = ExemplarStore(path=path).load()
        assert len(reloaded) == 1
        assert reloaded.suggest("why did cycle 4045 fail", None) is not None

    def test_exports_training_pairs_for_distillation(self):
        """The store is also the corpus a planner LoRA would be distilled from."""
        store = ExemplarStore()
        store.record("why did cycle 9016 fail", parse_plan(ROOT_CAUSE_9016), tier="llm")
        pairs = store.export_training_pairs()
        assert pairs and set(pairs[0]) == {"question", "plan", "op", "uses"}
        assert pairs[0]["plan"]["op"] == "root_cause"


class TestCacheEntityBinding:
    """The cache key erases entities, so the stored value must not contain them.

    Before this was fixed, "why did cycle 2750 fail" returned cycle 4045's
    answer - the key collapsed to one entry and the cached plan had a concrete
    udi baked in.
    """

    def test_cache_rebinds_to_the_asked_entity(self):
        cache = PlanCache()
        cache.put("why did cycle 9016 fail", parse_plan(ROOT_CAUSE_9016))
        plan = cache.get("why did cycle 2750 fail")
        assert plan is not None
        assert any(f.field == "udi" and f.value == 2750 for f in plan.filters)

    def test_engine_resolves_each_cycle_to_its_own_row(self):
        engine = Engine.build()
        seen = {}
        for udi in (9016, 4045, 2750):
            answer = engine.ask(f"why did cycle {udi} fail", SessionState())
            seen[udi] = answer.bundle.slots["cycle.udi"].value
        assert seen == {9016: 9016, 4045: 4045, 2750: 2750}


class TestLearningLoop:
    def test_only_verified_answers_are_learned(self):
        engine = Engine.build()
        engine.router.exemplars.clear()
        engine.ask("purple monkey dishwasher", SessionState())   # refused
        assert len(engine.router.exemplars) == 0
        engine.ask("what is the overall failure rate?", SessionState())
        assert len(engine.router.exemplars) == 1

    def test_cache_and_exemplar_hits_teach_nothing_new(self):
        """The store should grow with novelty, not with traffic."""
        engine = Engine.build()
        engine.router.exemplars.clear()
        engine.ask("what is the overall failure rate?", SessionState())
        size = len(engine.router.exemplars)
        for _ in range(5):
            engine.ask("what is the overall failure rate?", SessionState())
        assert len(engine.router.exemplars) == size

    def test_a_model_planned_question_becomes_a_cheap_one(self):
        """The point of the tier: solve it once expensively, then never again."""
        engine = Engine.build()
        engine.router.exemplars.clear()
        engine.router.cache.clear()

        # Neither wording may contain a grammar trigger word, or tier 1 answers
        # it and tier 2 is never exercised.
        novel = "Set side by side the strain accrual of breakdowns and clean runs"
        paraphrase = "Set side by side the strain accrual of breakdowns and clean cycles"
        from copilot.planner.grammar import plan_from_text

        assert not plan_from_text(novel).usable
        assert not plan_from_text(paraphrase).usable
        engine.router.learn(novel, parse_plan(COMPARE), "llm")
        routed = engine.router.route(paraphrase, SessionState())
        assert routed.tier == "exemplar"
        assert routed.plan.op.value == "compare"
        assert routed.elapsed_ms < 50
