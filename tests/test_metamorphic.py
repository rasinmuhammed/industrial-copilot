"""Relations that must hold, checked against generated inputs.

WHY THIS SUITE EXISTS
---------------------
Twenty-eight defects were found in this codebase by deliberate probing. Sorted
by root cause, the largest single category - nine of them, 32% - was the same
thing every time: **a guarantee asserted somewhere and enforced nowhere.**

    "the executor caps it"          the executor did not
    sigmas "measured"               they were invented; 94% false alarms
    unsourced numbers "unreachable" spelling them out reached
    session_id = "default"          implied an isolation that did not exist
    "exact"                         float split 128 rows on the boundary

Every one lived in a comment, a docstring, a README line or a default argument.
None lived anywhere a build could check.

The project already has four mechanisms that turn a claim into something
executable - the fail-closed verifier, `make verify`, `verify_readme`, and the
AST literal guard - and each has caught real errors, several of them mine. The
defects clustered exactly where no such mechanism existed.

So this is not another layer. The layering survived every attack: the closed
vocabulary defeated six injection strings, the IR rejected every malformed plan,
physics stayed finite at the extremes. What was missing is a way to make
BEHAVIOURAL claims executable, the way numeric ones already are.

WHY METAMORPHIC RATHER THAN MORE EXAMPLES
-----------------------------------------
Example-based tests need a known correct answer. For a generated operating
point nobody knows the right failure rate, which is the classic test-oracle
problem - and it is why the missing coverage sat undetected: every test fed the
system inputs its author had already thought about.

Metamorphic testing sidesteps the oracle. You may not know f(x), but you know
that f(x) and f(transform(x)) must be related in a specific way. Adding a filter
cannot increase a row count. Shuffling inputs cannot change a mean. Doubling
torque must exactly double power. Those hold for inputs nobody has imagined, so
they catch the defects that examples structurally cannot.

Generation is hand-rolled rather than Hypothesis: no new dependency, and the
distribution is visible in the file rather than behind a strategy DSL.
"""

from __future__ import annotations

import math
import random

import pytest

from copilot.engine import Engine
from copilot.ir import parse_plan
from copilot.ops.registry import execute
from copilot.physics import OperatingPoint, evaluate
from copilot.process_model import load_process_model
from copilot.session import SessionState

TRIALS = 200


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine.build()


def _points(n: int, seed: int = 20260817):
    """Operating points spanning the plausible envelope and past its edges."""
    rng = random.Random(seed)
    for _ in range(n):
        air = rng.uniform(295.0, 305.0)
        yield OperatingPoint(
            air_temp_k=air,
            process_temp_k=air + rng.uniform(5.0, 15.0),
            rotational_speed_rpm=rng.uniform(1100.0, 2900.0),
            torque_nm=rng.uniform(3.0, 80.0),
            tool_wear_min=rng.uniform(0.0, 260.0),
            product_type=rng.choice("LMH"),
        )


class TestPhysicsRelations:
    """Properties of the margin engine that hold for every operating point."""

    def test_a_mode_fires_exactly_when_its_margin_is_negative(self):
        """The sign convention is the whole basis of the system. If it can slip
        for some point, every downstream verdict is unsound for that point."""
        for point in _points(TRIALS):
            margins = evaluate(point)
            fired = set(margins.fired_modes())
            assert ("OSF" in fired) == (margins.osf_distance(point.osf_threshold) < 0)

    def test_power_is_exactly_linear_in_torque(self):
        """power = torque x omega. Scaling torque by k must scale power by k.

        A dimensional relation that holds only approximately is a unit bug
        waiting to surface at an extreme.
        """
        for point in _points(TRIALS // 2):
            for k in (0.5, 2.0, 10.0):
                scaled = OperatingPoint(
                    point.air_temp_k, point.process_temp_k,
                    point.rotational_speed_rpm, point.torque_nm * k,
                    point.tool_wear_min, point.product_type,
                )
                assert scaled.power_w == pytest.approx(point.power_w * k, rel=1e-12)

    def test_evaluation_is_pure(self):
        """Same point, same margins - no accumulated state anywhere."""
        for point in _points(50):
            first = evaluate(point)
            second = evaluate(point)
            assert first.hdf_distance == second.hdf_distance
            assert first.pwf_distance == second.pwf_distance

    def test_margins_are_always_finite(self):
        """No NaN, no inf, at any point in or beyond the envelope."""
        for point in _points(TRIALS):
            margins = evaluate(point)
            for value in (margins.hdf_distance, margins.pwf_distance,
                          margins.osf_distance(point.osf_threshold)):
                assert math.isfinite(value)

    def test_min_composition_is_associative_over_random_partitions(self):
        """The fleet rollup claim, checked against arbitrary splits rather than
        one hand-picked one."""
        rng = random.Random(7)
        values = [rng.uniform(-5000, 5000) for _ in range(2000)]
        for _ in range(50):
            cut = rng.randrange(1, len(values))
            assert min(min(values[:cut]), min(values[cut:])) == min(values)

    def test_the_config_driven_evaluator_agrees_on_generated_points(self):
        """Two independent implementations must agree on inputs neither author
        chose. The existing check runs over the 10,000 dataset rows, which are
        exactly the points both were written against."""
        model = load_process_model()
        for point in _points(TRIALS):
            reading = {
                "temp_delta_k": point.process_temp_k - point.air_temp_k,
                "rotational_speed_rpm": point.rotational_speed_rpm,
                "power_w": point.power_w,
                "overstrain_min_nm": point.tool_wear_min * point.torque_nm,
            }
            generic = set(model.fires(reading, point.product_type))
            fast = set(evaluate(point).fired_modes()) - {"TWF"}
            assert generic - {"TWF"} == fast


class TestQueryRelations:
    """Properties of an answer that hold whatever the question."""

    def test_narrowing_never_widens(self, engine):
        """Adding a conjunctive filter cannot increase the row count.

        A relation, not an expected value: it holds for filters nobody wrote a
        test for.
        """
        base = execute(parse_plan({"op": "rate"}), engine.ctx)
        rng = random.Random(3)
        for _ in range(30):
            field, op, value = rng.choice([
                ("torque_nm", ">", rng.uniform(10, 70)),
                ("tool_wear_min", "<", rng.uniform(20, 240)),
                ("rotational_speed_rpm", ">", rng.uniform(1200, 2500)),
                ("product_type", "=", rng.choice("LMH")),
            ])
            narrowed = execute(
                parse_plan({"op": "rate",
                            "filters": [{"field": field, "op": op, "value": value}]}),
                engine.ctx,
            )
            assert narrowed.provenance.row_count <= base.provenance.row_count

    def test_groups_partition_the_population(self, engine):
        """Counts across a grouping must sum to the ungrouped total. If they do
        not, rows are being dropped or double-counted somewhere in the SQL."""
        total = execute(parse_plan({"op": "rate"}), engine.ctx)
        for dimension in ("product_type", "shift", "machine_id"):
            grouped = execute(
                parse_plan({"op": "rate", "group_by": [dimension]}), engine.ctx
            )
            counts = [
                slot.value for key, slot in grouped.slots.items()
                if key.endswith(".n") and not key.startswith("overall")
            ]
            assert sum(counts) == total.provenance.row_count, dimension

    def test_a_question_is_answered_the_same_in_every_fresh_session(self, engine):
        """Session isolation, stated as a relation over arbitrary questions."""
        questions = [
            "What is the overall failure rate?",
            "Why did cycle 9016 fail?",
            "What causes failures?",
            "failure rate by product variant",
            "Compare failed and healthy cycles.",
        ]
        for question in questions:
            narrations = {
                engine.ask(question, SessionState()).narration for _ in range(3)
            }
            assert len(narrations) == 1, question

    def test_a_refusal_is_not_softened_by_conversational_context(self, engine):
        """A prior question must never make an unanswerable one answerable.

        This is the shape of the session-bleed defect: context that changes what
        the system is willing to claim.
        """
        unanswerable = [
            "what is the oil pressure",
            "how loud is the machine",
            "show me the bearing temperature",
        ]
        for question in unanswerable:
            state = SessionState()
            engine.ask("What is the overall failure rate?", state)
            engine.ask("What's the failure rate for L variants?", state)
            assert engine.ask(question, state).refused, question

    def test_every_numeral_in_every_answer_traces_to_a_slot(self, engine):
        """The flagship guarantee, asserted over a spread of questions rather
        than the golden set it was designed against."""
        questions = [
            "What are the typical operating conditions?",
            "What's the failure rate by product variant?",
            "How does failure rate vary with tool wear?",
            "Which variables best separate failures from healthy operation?",
            "What if we reduce torque by 5 Nm?",
            "Can I trust this data?",
            "Why are we seeing more failures at high rotational speeds?",
        ]
        for question in questions:
            answer = engine.ask(question, SessionState())
            if not answer.refused:
                assert answer.verified, question

    def test_a_null_counterfactual_changes_nothing(self, engine):
        """Asking "what if we change torque by 0" must reproduce the baseline.

        An identity transform is the cheapest possible check that the
        counterfactual path recomputes from base variables rather than mutating
        a stored column.
        """
        plan = parse_plan({"op": "counterfactual",
                           "params": {"changes": {"torque_nm": 0.0}}})
        bundle = execute(plan, engine.ctx)
        for mode in ("PWF", "OSF", "HDF"):
            before = bundle.slots.get(f"before.{mode}")
            after = bundle.slots.get(f"after.{mode}")
            if before is not None and after is not None:
                assert before.value == after.value, mode

    def test_the_physical_invariants_hold_over_the_whole_archive(self, engine):
        """Relations that must hold for every row, not just the ones a query
        touches.

        The streaming observer checks these per tick and catches an inverted
        thermocouple at 20 sigma. The historical path checks them too, but only
        when somebody asks "can I trust this data" - so a query whose margins
        depend on those channels is answered without consulting them. That gap
        is narrow and known; this test at least holds the relations themselves.
        """
        answer = engine.ask("Can I trust this data?", SessionState())
        slots = answer.bundle.slots
        assert slots["invariant.I1.violations"].value == 0     # process > air
        assert 9.0 < slots["invariant.I2.mean_temp_delta"].value < 11.0
        assert slots["invariant.I3.rpm_torque_corr"].value < -0.5

