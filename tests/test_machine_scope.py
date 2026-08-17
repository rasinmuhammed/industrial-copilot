"""Asking about one machine must not answer about the whole plant.

This is the silent-substitution failure again - the class of bug this system
treats as its worst - in the place it does the most damage:

    "How often does L-03 fail?"  ->  3.39 % (339 of 10,000)

Every guarantee held. The plan validated, the arithmetic was exact, the numerals
traced to slots, the verifier passed, and the answer was about ten thousand
cycles from fifteen machines when the question was about one. An engineer
reading 3.39 % concludes something false about L-03 and has no way to see it.

The cause: the machine pattern required a noun in front of the id
(`machine L-03`), so a bare id never matched, and a filter that is never added
cannot be seen to be missing. The scoped and unscoped answers are both
well-formed sentences with correct numbers in them.

It matters more now than it used to. The operations console docks the question
box under a selected machine, so nearly every question asked through the
product is about one asset.
"""

from __future__ import annotations

import pytest

from copilot.engine import Engine
from copilot.planner.grammar import _extract_filters
from copilot.ir import OpName
from copilot.session import SessionState


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine.build()


def _filters(question: str, op: OpName = OpName.RATE) -> dict[str, object]:
    return {f.field: f.value for f in _extract_filters(question, None, op)}


class TestABareMachineIdScopes:
    @pytest.mark.parametrize("question", [
        "How often does L-03 fail?",
        "what are the main failure modes on L-03",
        "describe L-03",
        "L-03 failure rate",
        "is L-03 healthy",
    ])
    def test_the_id_reaches_the_plan(self, question):
        assert _filters(question).get("machine_id") == "L-03"

    def test_the_noun_form_still_works(self):
        for phrasing in ("machine H-01", "asset H-01", "unit H-01", "machine #H-01"):
            assert _filters(f"describe {phrasing}").get("machine_id") == "H-01"

    def test_the_answer_is_about_that_machine(self, engine):
        """The end-to-end property. A machine's cycles are a small fraction of
        the fleet's, so a scoped answer has a visibly smaller denominator."""
        scoped = engine.ask("How often does L-03 fail?", SessionState())
        fleet = engine.ask("How many failures are in this data?", SessionState())
        assert not scoped.refused and not fleet.refused
        assert "10,000" in fleet.narration
        assert "10,000" not in scoped.narration


class TestAMachineIsNotAVariant:
    """`L-03` contains a word-boundary `L`, which the bare-variant pattern
    matches. Left in place, "failure rate for L-03 variants" scoped to machine
    L-03 AND every L-variant machine - an intersection nobody asked for, which
    silently changes the denominator again."""

    def test_the_letter_is_consumed_by_the_machine_id(self):
        f = _filters("what is the failure rate for L-03 variants")
        assert f.get("machine_id") == "L-03"
        assert "product_type" not in f

    def test_a_real_variant_question_is_untouched(self):
        f = _filters("what is the failure rate for L variants")
        assert f.get("product_type") == "L"
        assert "machine_id" not in f

    def test_both_can_be_asked_when_both_are_named(self):
        """"H variants on machine L-03" is contradictory, but it is what was
        asked; the engine's job is to scope as asked, not to guess."""
        f = _filters("failure rate for H variants on machine L-03")
        assert f.get("machine_id") == "L-03"
        assert f.get("product_type") == "H"


class TestIdsInsideLongerTokensAreNotAssets:
    """Product ids look like `L47181`, and a part number may embed a hyphen.
    Reading one as an asset id would scope a fleet question to one machine -
    the same failure with the sign flipped."""

    @pytest.mark.parametrize("question", [
        "describe product L47181",
        "show me part L-0312-A",
    ])
    def test_no_machine_filter_is_invented(self, question):
        assert "machine_id" not in _filters(question)
