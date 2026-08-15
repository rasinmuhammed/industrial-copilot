"""Phase 3: session state, routing, narration and the PCN verifier.

These test the guarantees the brief names directly — latency (tier routing),
context engineering (flat token budget), and hallucination reduction (fail-closed
numeric verification).
"""

from __future__ import annotations

import pytest

from copilot.engine import Engine
from copilot.evidence import EvidenceBundle, Provenance, Quality, Slot
from copilot.ir import OpName, parse_plan
from copilot.planner.cache import PlanCache, normalise
from copilot.planner.grammar import plan_from_text
from copilot.session import SessionState
from copilot.verify import Rejection, verify


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine.build()


def _bundle(**slots: tuple[float | str, str]) -> EvidenceBundle:
    b = EvidenceBundle(
        provenance=Provenance(plan_hash="x", kb_version="1", data_version="1", op="describe")
    )
    for sid, (value, unit) in slots.items():
        b.add(Slot(id=sid.replace("__", "."), value=value, unit=unit))
    return b


# --------------------------------------------------------------------------
# The verifier — Gate 4
# --------------------------------------------------------------------------


class TestVerifier:
    def test_resolves_slot_references(self):
        b = _bundle(failed__torque_nm__mean=(50.17, "N·m"))
        result = verify("Failed cycles ran at {{failed.torque_nm.mean}}.", b)
        assert result.ok
        assert "50.17 N·m" in result.text

    def test_rejects_a_bare_numeral(self):
        """The core PCN guarantee: a model-authored number cannot reach a human."""
        b = _bundle(failed__torque_nm__mean=(50.17, "N·m"))
        result = verify("Failed cycles ran at 50.17 N·m.", b)
        assert not result.ok
        assert result.rejections[0][0] is Rejection.UNSOURCED_NUMERAL

    def test_rejects_an_invented_number_near_a_real_one(self):
        b = _bundle(failed__torque_nm__mean=(50.17, "N·m"))
        result = verify("Torque averaged {{failed.torque_nm.mean}}, up 12% year on year.", b)
        assert not result.ok

    def test_rejects_an_unknown_slot(self):
        b = _bundle(failed__torque_nm__mean=(50.17, "N·m"))
        result = verify("Speed was {{failed.rotational_speed_rpm.mean}}.", b)
        assert not result.ok
        assert result.rejections[0][0] is Rejection.UNKNOWN_SLOT

    def test_mis_attribution_is_unrepresentable(self):
        """A real number quoted against the wrong cohort is the subtle failure.
        Slot ids carry the cohort, so the reference cannot be detached."""
        b = _bundle(
            failed__torque_nm__mean=(50.17, "N·m"),
            healthy__torque_nm__mean=(39.63, "N·m"),
        )
        good = verify("Failed averaged {{failed.torque_nm.mean}}.", b)
        assert good.ok and "50.17" in good.text
        assert good.slots_used == ["failed.torque_nm.mean"]

    def test_numbers_from_the_question_are_permitted(self):
        b = _bundle(all__n=(10000, "count"))
        result = verify(
            "You asked about 1380 rpm; there are {{all.n}} cycles.",
            b,
            question="What happens below 1380 rpm?",
        )
        assert result.ok

    def test_mode_codes_are_not_numerals(self):
        b = _bundle(cause__verdict=("OSF", ""))
        assert verify("The cause was {{cause.verdict}}, not HDF or PWF.", b).ok

    def test_abstained_slot_renders_as_words(self):
        b = EvidenceBundle(
            provenance=Provenance(plan_hash="x", kb_version="1", data_version="1", op="describe")
        )
        b.add(Slot(id="all.torque_nm.mean", value=None, quality=Quality.ABSTAIN))
        result = verify("The mean is {{all.torque_nm.mean}}.", b)
        assert result.ok
        assert "not determined" in result.text

    def test_empty_draft_is_rejected(self):
        assert not verify("   ", _bundle()).ok

    def test_rejection_reason_is_actionable(self):
        b = _bundle(all__n=(10, "count"))
        reason = verify("There were 10 cycles.", b).reason()
        assert "slot.id" in reason


# --------------------------------------------------------------------------
# Session state — context engineering
# --------------------------------------------------------------------------


class TestSessionState:
    def test_token_budget_stays_flat_across_turns(self, engine):
        """The property that matters at plant scale: turn 40 costs like turn 2."""
        state = SessionState()
        sizes = []
        for i in range(12):
            engine.ask(f"What's the failure rate for {'LMH'[i % 3]} variants?", state)
            sizes.append(state.token_estimate())
        assert max(sizes[4:]) < 2 * max(sizes[:4]) + 40
        assert len(state.turns) <= 6

    def test_focus_is_dropped_when_a_plan_pins_nothing(self, engine):
        """A lingering focus would make the scope line lie."""
        state = SessionState()
        engine.ask("Why did cycle 9016 fail?", state)
        assert state.focus is not None and state.focus.kind == "cycle"
        engine.ask("What's the overall failure rate?", state)
        assert state.focus is None

    def test_scope_line_reflects_the_current_answer(self, engine):
        state = SessionState()
        engine.ask("What's the failure rate for L variants?", state)
        assert "L variant" in state.scope_line()

    def test_follow_up_mutates_one_filter(self, engine):
        state = SessionState()
        first = engine.ask("What's the failure rate for L variants?", state)
        second = engine.ask("What about H?", state)
        assert first.plan is not None and second.plan is not None
        assert first.plan.op is second.plan.op is OpName.RATE
        assert second.bundle.slots["h.n"].value == 1003

    def test_single_cycle_focus_does_not_leak_into_aggregates(self, engine):
        """Inheriting a udi focus into a trend collapses it to one row."""
        state = SessionState()
        engine.ask("Why did cycle 9016 fail?", state)
        answer = engine.ask("How does failure rate vary with tool wear?", state)
        assert answer.verified
        assert answer.bundle.provenance.row_count == 10000


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


class TestRouting:
    def test_normalisation_collapses_entity_specifics(self):
        """Tenant independence: one cache entry serves every factory."""
        assert normalise("Why did cycle 9016 fail?") == normalise("Why did cycle 4045 fail?")

    def test_normalisation_collapses_synonyms(self):
        assert normalise("what is the average rpm") == normalise(
            "what is the average rotational speed"
        )

    def test_cache_returns_an_equivalent_plan(self):
        """The cache stores a SHAPE and rebinds entities, so the returned plan is
        equivalent rather than identical — that is what stops one cache entry
        answering every "why did cycle N fail" with the same row."""
        cache = PlanCache()
        plan = parse_plan({"op": "rate", "group_by": ["product_type"]})
        cache.put("failure rate by variant", plan)
        got = cache.get("failure rate by variant")
        assert got is not None and got.hash == plan.hash
        assert cache.hit_rate == 1.0

    def test_cache_evicts_beyond_capacity(self):
        cache = PlanCache(capacity=2)
        plan = parse_plan({"op": "rate"})
        # Distinct SHAPES — trailing integers all normalise to the same key.
        for question in ("failure rate by variant", "what drives failures",
                         "why did the cycle fail", "show me the worst cycles",
                         "can i trust this data"):
            cache.put(question, plan)
        assert len(cache) == 2

    @pytest.mark.parametrize(
        "question,expected",
        [
            ("Why did cycle 9016 fail?", OpName.ROOT_CAUSE),
            ("What's the failure rate by product variant?", OpName.RATE),
            ("Compare failed versus healthy machines", OpName.COMPARE),
            ("How does failure rate vary with tool wear?", OpName.TREND),
            ("What drives failures?", OpName.DRIVERS),
            ("Can I trust this data?", OpName.DATA_QUALITY),
            ("What if we reduce torque by 5 Nm?", OpName.COUNTERFACTUAL),
        ],
    )
    def test_grammar_routes_common_questions(self, question, expected):
        match = plan_from_text(question)
        assert match.usable and match.plan.op is expected

    def test_grammar_handles_plural_metric_names(self):
        """'rotational speeds' must resolve, or the flagship question misroutes."""
        match = plan_from_text("Why are we seeing more failures at high rotational speeds?")
        assert match.usable
        assert match.plan.bin is not None
        assert match.plan.bin.field == "rotational_speed_rpm"

    def test_population_ops_are_never_filtered_to_failures(self):
        """Filtering a rate to failures answers 'the failure rate among failures'."""
        match = plan_from_text("Why are we seeing more failures at high rotational speeds?")
        assert all(f.field != "failure" for f in match.plan.filters)

    def test_gibberish_is_not_forced_into_a_plan(self):
        assert not plan_from_text("purple monkey dishwasher").usable

    def test_unroutable_question_refuses_with_guidance(self, engine):
        answer = engine.ask("purple monkey dishwasher", SessionState())
        assert answer.refused
        assert "torque" in answer.text  # names what it can answer


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


class TestEndToEnd:
    def test_the_flagship_question_refutes_its_premise(self, engine):
        answer = engine.ask(
            "Why are we seeing more failures at high rotational speeds?", SessionState()
        )
        assert answer.verified
        assert "U-shaped" in answer.text
        assert answer.bundle.slots["premise.low_high_ratio"].value == pytest.approx(5.4, abs=0.1)

    def test_every_demo_question_verifies(self, engine):
        from copilot.cli import DEMO

        state = SessionState()
        for _label, question in DEMO:
            answer = engine.ask(question, state)
            assert answer.verified, f"{question!r} failed verification:\n{answer.text}"
            assert not answer.refused

    def test_answers_carry_a_replay_handle(self, engine):
        answer = engine.ask("What's the overall failure rate?", SessionState())
        assert answer.replay_handle
        assert answer.bundle.provenance.kb_version
        assert answer.bundle.provenance.data_version

    def test_grammar_tier_is_fast(self, engine):
        answer = engine.ask("What's the failure rate by product variant?", SessionState())
        assert answer.tier == "grammar"
        assert answer.plan_ms < 50  # generous for CI; typically well under 1 ms

    def test_no_answer_contains_an_unsourced_numeral(self, engine):
        """The headline hallucination metric, asserted end to end."""
        from copilot.cli import DEMO

        state = SessionState()
        for _label, question in DEMO:
            answer = engine.ask(question, state)
            # Scan only the region PCN covers: the narrated prose. The scope
            # line and the provenance footer are engine-generated metadata, and
            # warnings are operator-generated — none pass through a model.
            body = answer.narration
            permitted = {s.render() for s in answer.bundle.slots.values() if s.value is not None}
            residual = body
            for value in sorted(permitted, key=len, reverse=True):
                residual = residual.replace(value, " ")
            leftovers = [t for t in residual.split() if any(c.isdigit() for c in t)]
            assert not leftovers, f"{question!r} leaked {leftovers}"
