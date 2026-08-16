"""Every way this could be a bad solution, probed deliberately.

Not "does it work" — the other suites cover that. This asks the questions a
hostile reviewer asks: can I make it execute my SQL, can I make it lie, can I
exhaust it, does it agree with itself, does it survive being used by more than
one person at once.

Three real defects were found here and each is now a test:

  * the fail-closed verifier could be bypassed by SPELLING THE NUMBER OUT
  * grouping by an identifier emitted 40,006 slots into an in-memory bundle
  * a comment claimed the executor capped open-valued dimensions; it did not
"""

from __future__ import annotations

import threading

import pytest

from copilot.engine import Engine
from copilot.ir import PlanError, ValidationStage, parse_plan
from copilot.ops.rate import MAX_REPORTABLE_GROUPS
from copilot.ops.registry import execute
from copilot.session import SessionState
from copilot.verify import verify


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine.build()


class TestInjection:
    """The closed vocabulary is the defence. Verify it actually holds."""

    @pytest.mark.parametrize("question", [
        "failure rate for L'; DROP TABLE cycles; --",
        "why did cycle 1 OR 1=1 fail",
        "describe torque for variant L' UNION SELECT * FROM cycles --",
        "why did cycle 9016; ATTACH 'evil.db' AS e fail",
        "failure rate by product_type; PRAGMA database_list",
        "what is the average torque\\x00; DELETE FROM cycles",
    ])
    def test_nothing_reaches_the_sql(self, engine, question):
        """A plan can only name vocabulary from the semantic layer, so hostile
        text has nowhere to land. This asserts the property rather than
        trusting it."""
        answer = engine.ask(question, SessionState())
        sql = (answer.bundle.provenance.sql if answer.bundle else "") or ""
        for danger in ("DROP", "UNION", "ATTACH", "PRAGMA", "DELETE", "--"):
            assert danger not in sql.upper(), f"{danger} reached the SQL"

    def test_a_hostile_question_cannot_crash_the_engine(self, engine):
        for question in ("\\x00\\x01\\x02", "𝕊𝕖𝕝𝕖𝕔𝕥", "a" * 10_000, "", "   "):
            engine.ask(question, SessionState())      # must not raise


class TestTheVerifierCannotBeTalkedAround:
    """The flagship guarantee: an unsourced figure must be unreachable.

    It was reachable. The verifier scanned for DIGITS, so a narration saying
    "three point three nine percent" passed the gate whose entire purpose is to
    stop exactly that. The template narrator never spells numbers out, but the
    MODEL narrator is the component the gate exists to contain, and language
    models spell numbers routinely.
    """

    @pytest.fixture(scope="class")
    def bundle(self, engine):
        return engine.ask("What is the overall failure rate?", SessionState()).bundle

    @pytest.mark.parametrize("draft", [
        "The rate is 3.39 percent.",
        "The rate is three point three nine percent.",
        "About three quarters of failures are explained.",
        "Twenty five machines are affected.",
        "Roughly ten thousand cycles were scanned.",
        "Forty-six tools failed.",
        "The rate is ٣.٣٩ percent.",
        "The rate is **3.39**%.",
        "The rate is 3.39e0 percent.",
    ])
    def test_an_unsourced_figure_is_refused(self, bundle, draft):
        assert not verify(draft, bundle, question="rate?").ok

    @pytest.mark.parametrize("draft", [
        "{{all.failure_rate}} of cycles failed.",
        "The first band is worst.",
        "one mode fired.",
        "Torque and speed are collinear.",
    ])
    def test_ordinary_english_is_not_false_flagged(self, bundle, draft):
        """A guard that blocks legitimate prose is worse than the hole it
        closes, because it pushes the narrator toward vaguer language."""
        assert verify(draft, bundle, question="rate?").ok


class TestResourceExhaustion:
    """An evidence bundle is held in memory and rendered in full."""

    def test_grouping_by_an_identifier_is_rejected(self):
        """`udi` addresses one record. Grouping by it produced 40,006 slots.

        Validation skipped it because it declares no value list, under a comment
        asserting that "the executor caps it". The executor did not. A comment
        claiming a guarantee that does not exist is the same defect as a
        constant described as measured when it was chosen.
        """
        with pytest.raises(PlanError) as exc:
            parse_plan({"op": "rate", "group_by": ["udi"]})
        assert exc.value.stage is ValidationStage.CARDINALITY

    def test_addressing_one_record_still_works(self):
        """The fix must not break the flagship 'why did cycle 9016 fail'."""
        plan = parse_plan(
            {"op": "root_cause",
             "filters": [{"field": "udi", "op": "=", "value": 9016}]}
        )
        assert plan.filters[0].field == "udi"

    def test_open_valued_dimensions_have_a_real_ceiling(self, engine):
        """machine_id is 15 here and thousands in a real fleet."""
        answer = engine.ask("failure rate by machine", SessionState())
        assert not answer.refused
        assert len(answer.bundle.slots) < MAX_REPORTABLE_GROUPS * 6

    def test_an_impossible_filter_returns_no_rows_rather_than_nan(self, engine):
        plan = parse_plan({
            "op": "describe", "metrics": ["torque_nm"],
            "filters": [{"field": "torque_nm", "op": ">", "value": 100},
                        {"field": "torque_nm", "op": "<", "value": 1}],
        })
        bundle = execute(plan, engine.ctx)
        assert bundle.provenance.row_count == 0
        for slot in bundle.slots.values():
            if isinstance(slot.value, float):
                assert slot.value == slot.value              # not NaN
                assert slot.value not in (float("inf"), float("-inf"))


class TestAgreementWithItself:
    """A system that answers differently on a retry cannot be audited."""

    def test_the_substance_is_deterministic(self, engine):
        """Only the provenance footer varies — wall-clock timing, and the tier
        once the cache warms. The narration, slot values and plan hash must be
        byte-identical."""
        answers = [
            engine.ask("What is the overall failure rate?", SessionState())
            for _ in range(5)
        ]
        assert len({a.narration for a in answers}) == 1
        assert len({a.bundle.provenance.plan_hash for a in answers}) == 1
        values = [
            tuple(sorted((k, str(s.value)) for k, s in a.bundle.slots.items()))
            for a in answers
        ]
        assert len(set(values)) == 1

    def test_concurrent_sessions_do_not_bleed(self, engine):
        """One engine, many users. Session state is per-request; if it were
        shared, one engineer's scope would silently narrow another's answer."""
        results: list[str] = []
        errors: list[str] = []

        def ask(question: str) -> None:
            try:
                results.append(engine.ask(question, SessionState()).narration)
            except Exception as exc:                     # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")

        threads = [
            threading.Thread(target=ask, args=("What is the overall failure rate?",))
            for _ in range(24)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors[:3]
        assert len(set(results)) == 1, "concurrent answers diverged"

    def test_the_embedding_is_stable_across_processes(self):
        """The retrieval hash must not be Python's randomised builtin.

        HashingEmbedder used hash(), which is seeded per interpreter. Two
        silent consequences: exemplar retrieval differed between runs (measured
        coverage moved 96.8% <-> 98.4% depending on PYTHONHASHSEED), and a
        PERSISTED store would have been quietly corrupt — vectors written by one
        process never matching those computed by the next, so the system would
        stop learning after a restart with no error at all.
        """
        import subprocess
        import sys

        probe = (
            "from copilot.planner.exemplars import HashingEmbedder;"
            "import numpy as np;"
            "v=HashingEmbedder().embed('impact of dropping torque 10 Nm');"
            "print(f'{float(v@np.arange(len(v))):.6f}')"
        )
        seen = set()
        for seed in ("0", "1", "7"):
            out = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True, text=True,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            )
            seen.add(out.stdout.strip())
        assert len(seen) == 1, f"embedding varies with hash seed: {seen}"

    def test_a_scoped_session_does_not_leak_into_a_fresh_one(self, engine):
        scoped = SessionState()
        engine.ask("What's the failure rate for L variants?", scoped)
        fresh = engine.ask("What's the overall failure rate?", SessionState())
        assert fresh.bundle.provenance.row_count == 10_000
