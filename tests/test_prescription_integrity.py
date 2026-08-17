"""The prescriptive path must answer about the setpoint that was asked about.

Two independent defects, on the one capability an operator would actually act
on, both of which produced verified answers to questions nobody asked:

1. THE PLAN CACHE INHERITED THE OPERATING POINT.
   `normalise()` erases numbers on purpose - "at 8 minutes of wear" and "at 200
   minutes of wear" are one cache key, which is what lets one shared cache serve
   a fleet. The value was supposed to hold only a plan SHAPE, with entities
   rebound from the current question. Filters were rebound. Params were not. So
   every envelope question in the process returned the operating point of
   whichever one was asked first.

2. THE ENVELOPE OP DISCARDED A LONE PARAMETER.
   It honoured explicit values only when at least three of five were given.
   "What torque should I run at 200 minutes of wear" names exactly one - the
   entire subject of the question - so it was dropped and the envelope was
   computed at the cohort's mean wear of 108 minutes.

Together they mean an engineer asking about a badly worn tool was told the safe
torque ceiling for an average one. The overstrain limit is a product of wear and
torque, so the ceiling at 240 minutes is 50 N·m and at 108 minutes it is power-
limited at 55.85 N·m: the advice was to run a worn tool above its limit.

554 tests passed throughout. Nothing was checking that the answer described the
operating point in the question, because every answer was internally consistent
and correctly computed - it was consistent about the wrong point.
"""

from __future__ import annotations

import re

import pytest

from copilot.engine import Engine
from copilot.session import SessionState


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine.build()


def _reported_wear(narration: str) -> float:
    m = re.search(r"At ([\d,.]+) min of wear", narration)
    assert m, f"no operating point in narration: {narration[:120]!r}"
    return float(m.group(1).replace(",", ""))


def _band(narration: str) -> tuple[float, float]:
    m = re.search(r"between ([\d.]+) N·m and ([\d.]+) N·m", narration)
    assert m, f"no torque band in narration: {narration[:120]!r}"
    return float(m.group(1)), float(m.group(2))


class TestTheAnswerDescribesTheQuestionsSetpoint:
    @pytest.mark.parametrize("wear", [8, 60, 150, 200, 240])
    def test_the_reported_wear_is_the_asked_wear(self, engine, wear):
        answer = engine.ask(
            f"What torque should I run at {wear} minutes of wear?", SessionState()
        )
        assert not answer.refused
        assert _reported_wear(answer.narration) == pytest.approx(wear, abs=0.5)

    def test_a_lone_parameter_is_enough(self, engine):
        """One named quantity used to be silently discarded - the op required
        three of five before it honoured any."""
        answer = engine.ask("What torque should I run at 240 minutes of wear?", SessionState())
        assert _reported_wear(answer.narration) == pytest.approx(240, abs=0.5)

    def test_unspecified_quantities_still_come_from_the_cohort(self, engine):
        """Overlay, not replace: what was not asked about is observed, and the
        answer says so rather than quietly inventing a nominal value."""
        answer = engine.ask("What torque should I run at 240 minutes of wear?", SessionState())
        assert any("MEAN" in w.message for w in answer.bundle.warnings)


class TestTheCacheDoesNotInheritOperatingPoints:
    def test_a_second_question_is_not_answered_with_the_first(self, engine):
        """The cache hit is desirable - the plan SHAPE is genuinely reusable.
        What must not survive the hit is the operating point."""
        first = engine.ask("What torque should I run at 8 minutes of wear?", SessionState())
        second = engine.ask("What torque should I run at 240 minutes of wear?", SessionState())

        assert second.tier == "cache", "expected a cache hit; this test is now checking nothing"
        assert _reported_wear(first.narration) == pytest.approx(8, abs=0.5)
        assert _reported_wear(second.narration) == pytest.approx(240, abs=0.5)

    def test_the_prescription_itself_changes(self, engine):
        """Wear and torque multiply into the overstrain limit, so a worn tool
        has a lower ceiling. Identical bands across wear levels was the visible
        symptom, and the one that makes this a safety defect rather than a
        cosmetic one."""
        low = engine.ask("What torque should I run at 8 minutes of wear?", SessionState())
        high = engine.ask("What torque should I run at 253 minutes of wear?", SessionState())
        assert _band(high.narration)[1] < _band(low.narration)[1]

    def test_a_question_naming_nothing_inherits_nothing(self, engine):
        """The dangerous direction: after a specific question is cached, a
        general one must not silently adopt its setpoint."""
        engine.ask("What torque should I run at 253 minutes of wear?", SessionState())
        general = engine.ask("What is the safe operating window?", SessionState())
        if not general.refused:
            assert _reported_wear(general.narration) != pytest.approx(253, abs=0.5)


class TestPlanShapesCarryNoEntities:
    """The module docstring in cache.py already stated this invariant: because
    the KEY erases entities, the VALUE must not contain them. It was true of
    filters and of `sql`, and untrue of everything else."""

    def test_the_stored_shape_omits_the_operating_point(self, engine):
        from copilot.planner.exemplars import plan_shape

        answer = engine.ask("What torque should I run at 150 minutes of wear?", SessionState())
        shape = plan_shape(answer.plan)
        assert "tool_wear_min" not in (shape.get("params") or {})

    def test_a_premise_survives_the_round_trip(self, engine):
        """Premise verification is the flagship capability and it lives in
        params, so dropping entity params must not disarm it."""
        first = engine.ask("why do H variants fail more", SessionState())
        second = engine.ask("why do H variants fail more", SessionState())
        for answer in (first, second):
            assert answer.plan.verify_premise
            assert (answer.plan.params or {}).get("premise", {}).get("value") == "H"
