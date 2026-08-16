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


class TestTheHttpSurface:
    """The engine was tested; the API in front of it was not."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        import copilot.api as api

        api._sessions.clear()
        return TestClient(api.app)

    def test_two_callers_without_a_session_id_do_not_share_one(self, client):
        """session_id defaulted to the literal "default", so every caller who
        omitted it shared a conversation. Engineer A scoped to L variants;
        Engineer B then asked for the OVERALL rate and received A's scope — a
        confident, correctly computed answer to a question nobody asked.

        Follow-up context is acceptance criterion 4, and it was the mechanism
        that leaked.
        """
        a = client.post("/ask", json={
            "question": "What's the failure rate for L variants?"}).json()
        b = client.post("/ask", json={
            "question": "What's the overall failure rate?"}).json()
        assert a["scope"] != b["scope"]
        assert "all" in b["scope"].lower()

    def test_an_explicit_session_still_carries_context(self, client):
        """Isolation must not cost the follow-up capability it protects."""
        client.post("/ask", json={
            "question": "What's the failure rate for L variants?",
            "session_id": "eng-7"})
        follow = client.post("/ask", json={
            "question": "What about H?", "session_id": "eng-7"}).json()
        assert "H" in follow["scope"]

    def test_the_session_store_is_bounded(self, client):
        """Nothing was ever evicted: 5,000 ids retained 5,001 states forever."""
        import copilot.api as api

        for i in range(api._MAX_SESSIONS + 200):
            client.post("/ask", json={"question": "rate?", "session_id": f"u{i}"})
        assert len(api._sessions) <= api._MAX_SESSIONS

    @pytest.mark.parametrize("payload", [
        {}, {"question": None}, {"question": 12345},
        {"question": "a" * 200_000}, {"question": {"$ne": 1}},
        {"question": "rate?", "session_id": "../../etc/passwd"},
    ])
    def test_malformed_payloads_never_leak_internals(self, client, payload):
        response = client.post("/ask", json=payload)
        assert response.status_code in (200, 422)
        for marker in ("Traceback", "site-packages", 'File "'):
            assert marker not in response.text


class TestBoundaryArithmetic:
    """"Exact" is a strong word. Test what it actually covers."""

    def test_a_margin_within_representation_error_is_degenerate(self):
        """128 rows here have a thermal delta of exactly 8.6 K, and float
        subtraction puts 43 below the limit and 85 at or above it, decided by
        which decimal pair was subtracted:

            306.9 - 298.3 = 8.599999999999966  -> fires
            308.6 - 300.0 = 8.600000000000023  -> does not fire

        Both are 8.6 K. The rule was deciding on representation error.
        """
        from copilot.physics import is_degenerate

        for process, air in [(306.9, 298.3), (308.6, 300.0), (309.4, 300.8)]:
            margin = (process - air) - 8.6
            assert is_degenerate(margin, process, air, 8.6)

    def test_a_real_verdict_is_not_called_degenerate(self):
        """A tolerance that swallows genuine margins is worse than none."""
        from copilot.physics import is_degenerate

        assert not is_degenerate(2.5, 310.0, 300.0, 8.6)
        assert not is_degenerate(-1.2, 310.0, 300.0, 8.6)

    def test_the_tolerance_scales_with_magnitude(self):
        """Overstrain lives near 11,000 and temperature near 8.6. One constant
        cannot serve both, so the tolerance is derived in ULPs."""
        from copilot.physics import boundary_tolerance

        assert boundary_tolerance(11_000.0) > boundary_tolerance(8.6)

    def test_min_composition_is_exactly_associative(self):
        """The fleet rollup claim: min(min(a), min(b)) == min(a + b)."""
        import random

        random.seed(1)
        values = [random.uniform(-100, 100) for _ in range(10_000)]
        assert min(min(values[:5000]), min(values[5000:])) == min(values)

    def test_power_stays_finite_at_the_extremes(self):
        import math

        from copilot.physics import OperatingPoint

        for torque, rpm in [(0.0, 1500.0), (76.6, 2886.0), (1e-12, 1.0)]:
            point = OperatingPoint(300.0, 310.0, rpm, torque, 100.0, "L")
            assert math.isfinite(point.power_w)


class TestGuardsThatMustActuallyFire:
    """An unexercised guard is not a guard.

    The invariant warning was added, 443 tests passed, and it was broken — a
    missing import in a branch that never runs on a clean archive. The whole
    suite was green because the code path was unreachable with real data.

    That is the failure this project keeps rediscovering, so every guard whose
    trigger condition is absent from the dataset gets a test that forces the
    condition.
    """

    def test_a_violated_invariant_reaches_the_answers_that_depend_on_it(self):
        """The invariants are checked when somebody asks about data quality.
        They must also reach a question whose margins depend on those channels,
        because otherwise a query over a corrupt archive returns confident
        numbers with no hint that the physics is impossible.
        """
        engine = Engine.build()
        engine._invariants = {"I1": 137}          # simulate a corrupt archive
        for question in ("What are the typical operating conditions?",
                         "What is the overall failure rate?"):
            answer = engine.ask(question, SessionState())
            flagged = [w for w in answer.bundle.warnings if w.code == "data_quality"]
            assert flagged, question
            assert "137" in flagged[0].message

    def test_a_clean_archive_raises_nothing(self):
        """A guard that always fires is noise, not protection."""
        engine = Engine.build()
        assert engine._invariants == {"I1": 0}
        answer = engine.ask("What is the average torque?", SessionState())
        assert not [w for w in answer.bundle.warnings if w.code == "data_quality"]

    def test_an_intermittent_channel_is_caught(self):
        """Staleness counted CONSECUTIVE gaps and escalated after three, so a
        sensor dropping one reading in five reset the counter on every good
        sample and was reported healthy. Loose connectors fail exactly this way.
        """
        import duckdb

        import copilot.observer as ob

        con = duckdb.connect()
        con.execute("CREATE VIEW t AS SELECT * FROM read_csv_auto('data/ai4i2020.csv')")
        rows = con.execute(
            'SELECT "Air temperature [K]", "Process temperature [K]", '
            '"Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]" '
            "FROM t ORDER BY UDI LIMIT 400"
        ).fetchall()
        ready = ob.WARMUP + ob.CALIBRATION
        observer = ob.FleetObserver()
        caught = None
        for i, r in enumerate(rows):
            reading = {
                "machine_id": "M1", "air_temp_k": r[0], "process_temp_k": r[1],
                "rotational_speed_rpm": r[2], "torque_nm": r[3], "tool_wear_min": r[4],
            }
            if i >= ready and (i - ready) % 5 == 0:
                reading["torque_nm"] = float("nan")     # 20% loss, never 3 in a row
            report = observer.observe(reading)
            status = report.channels["torque_nm"].status
            if i >= ready and caught is None and status in (
                ob.NE107.FAILURE, ob.NE107.MAINTENANCE_REQUIRED
            ):
                caught = i - ready
        assert caught is not None and caught < 40

    def test_a_stream_terminates(self):
        """/stream/alerts defaulted to limit=None: 10,000 cycles at the default
        takt is 33 minutes per connection, with no ceiling on either the limit
        or the number of connections."""
        import itertools

        from fastapi.testclient import TestClient

        from copilot.api import MAX_STREAM_TICKS, app

        client = TestClient(app)
        with client.stream("GET", "/stream/alerts", params={"speed": 0}) as response:
            lines = sum(1 for _ in itertools.islice(response.iter_lines(), 50_000))
        assert 0 < lines < 50_000
        over = client.get("/stream/alerts",
                          params={"speed": 0, "limit": MAX_STREAM_TICKS + 1})
        assert over.status_code == 422

