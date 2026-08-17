"""The worst failure an Argus platform can have.

Not "I don't know". Not a wrong number. A **confident, verified, arithmetically
exact answer about a different sensor than the one the engineer asked about.**

    "show me the bearing temperature"       -> 20 arbitrary rows
    "how much does a replacement tool cost" -> air temperature statistics

Both were real. Every downstream guarantee held perfectly: the plan validated,
the arithmetic was exact, every numeral traced to a slot, the verifier passed.
The answers were still useless and misleading, because the questions were about
a bearing and a purchase order and the answers were about ambient air.

The cause was structural, not incidental: the grammar matched an intent verb
("show me", "how much") without resolving the subject, then fell back to
`_DEFAULT_DESCRIBE_METRICS` — every metric we do have.

The existing refusal test passed only by luck. It used "vibration signature",
which shares no token with any synonym, so nothing matched and the planner
declined for the right reason by accident.

Found by the risk-coverage harness, which is the only reason it was found at
all: no correctness gate could see it, because the answers were correct.
"""

from __future__ import annotations

import pytest

from copilot.engine import Engine
from copilot.planner.unknown import detect_unknown_quantity
from copilot.session import SessionState


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine.build()


class TestForeignQuantitiesAreNamed:
    """Refusing is necessary; saying WHAT is missing is what makes it useful."""

    @pytest.mark.parametrize("question,expected", [
        ("show me the bearing temperature", "bearing temperature"),
        ("what is the coolant temperature", "coolant"),   # named list wins
        ("what is the oil pressure", "pressure"),
        ("how loud is the machine", "acoustic"),
        ("what is the vibration signature", "vibration"),
        ("what is the spindle runout", "spindle runout"),
        ("how much does a replacement tool cost", "cost"),
        ("who is the operator on shift 3", "personnel"),
        ("what is the humidity in the plant", "humidity"),
        ("show me the flow rate", "flow"),
    ])
    def test_the_missing_quantity_is_identified(self, question, expected):
        found = detect_unknown_quantity(question)
        assert found is not None, f"{question!r} was not caught"
        assert found.quantity == expected

    def test_the_refusal_says_what_is_available(self, engine):
        answer = engine.ask("show me the bearing temperature", SessionState())
        assert answer.refused
        assert "no bearing temperature measurement" in answer.text
        assert "torque" in answer.text          # names what we DO have


class TestOurOwnChannelsStillWork:
    """A guard that refuses real questions is worse than the bug it fixes."""

    @pytest.mark.parametrize("question", [
        "what is the air temperature",
        "how hot does the process usually run",
        "what is the average temperature",
        "what is the process temperature",
        "what is the normal temperature differential",
        "describe the ambient temperature",
        "typical rpm?",
        "What is the average tool wear?",
    ])
    def test_known_channels_are_not_refused(self, question):
        assert detect_unknown_quantity(question) is None

    def test_the_engine_still_answers_them(self, engine):
        answer = engine.ask("what is the average tool wear?", SessionState())
        assert not answer.refused
        assert answer.bundle is not None and answer.bundle.slots


class TestNoSilentSubstitution:
    """The end-to-end property: never answer about a sensor we were not asked about."""

    def test_bearing_temperature_does_not_return_rows(self, engine):
        answer = engine.ask("show me the bearing temperature", SessionState())
        assert answer.refused
        assert answer.bundle is None

    def test_tool_cost_does_not_return_air_temperature(self, engine):
        """The sharpest instance: 'tool' matched tool_wear, and the fallback
        described every metric, so the narrator led with ambient air."""
        answer = engine.ask("how much does a replacement tool cost", SessionState())
        assert answer.refused
        assert "air temperature averages" not in answer.text


class TestCoverageDidNotRegress:
    """Fixing silent substitution must not turn the system mute.

    The risk-coverage harness exists because correctness gates cannot see this
    tradeoff: a system that answers nothing scores perfectly on every one of
    them. These are the phrasings that were being declined despite mapping onto
    operations the engine already supports.
    """

    @pytest.mark.parametrize("question,op", [
        ("what does normal look like here", "describe"),
        ("what proportion of cycles failed", "rate"),
        ("How many failures are in this data?", "rate"),
        ("Show failures grouped by machine.", "rate"),
        ("what are the main failure modes", "root_cause"),
        ("which failure modes are firing", "root_cause"),
        ("biggest factors behind breakdowns", "drivers"),
        ("are there problems with this dataset", "data_quality"),
        ("is the labelling reliable", "data_quality"),
        ("what torque should I run at 150 minutes of wear", "envelope"),
    ])
    def test_natural_phrasings_reach_the_right_operation(self, engine, question, op):
        answer = engine.ask(question, SessionState())
        assert not answer.refused, f"{question!r} was declined"
        assert answer.plan is not None and answer.plan.op.value == op

    def test_plural_failures_matches_singular_fail(self, engine):
        """The RATE pattern ended in `fail)\\b`, so "how many fail" matched and
        "how many failures" did not — every natural plural phrasing of the
        commonest question in the product was unreachable."""
        answer = engine.ask("How many failures are in this data?", SessionState())
        assert not answer.refused

    def test_a_causal_premise_about_a_metric_is_verified(self, engine):
        """Premise verification is the flagship capability and it was
        unreachable from the most natural way to assert a cause."""
        answer = engine.ask("why is torque causing so many failures", SessionState())
        assert not answer.refused
        assert any(w.code == "premise_refuted" for w in answer.bundle.warnings)
