"""The transport boundary and the ranking falsifier.

Two classes of failure that a CSV can never surface:

  * what arrives is not what was measured — duplicates, reordering, gaps,
    clocks that disagree
  * a ranked list of assets that is pure chance, printed as though it were a
    finding, because ranking N groups always produces an extreme
"""

from __future__ import annotations

import pytest

from copilot.connectors import CsvSource, JsonlSource, MqttSource, OpcUaSource, TagMap, build_source
from copilot.engine import Engine
from copilot.intake import Intake, IntakeConfig, Verdict
from copilot.session import SessionState
from copilot.stats import spread_vs_chance

NOW = 1_700_000_000.0


def _steady(intake: Intake, n: int = 20, period: float = 2.0, start: float = NOW):
    """Establish a cadence so gap detection has a period to compare against."""
    for i in range(n):
        intake.offer(
            {"machine_id": "M1", "ts": start + i * period, "seq": i},
            now=start + i * period,
        )
    return start + n * period


@pytest.fixture
def intake() -> Intake:
    return Intake(IntakeConfig(lateness_s=30, stale_s=300, gap_factor=3.0))


class TestAtLeastOnceDelivery:
    """MQTT QoS 1 and Kafka both guarantee 'more than once'. That is the
    contract, not a fault, and it must not reach the estimator twice."""

    def test_a_duplicate_is_rejected(self, intake):
        t = _steady(intake)
        first = intake.offer({"machine_id": "M1", "ts": t, "seq": 99}, now=t)
        again = intake.offer({"machine_id": "M1", "ts": t, "seq": 99}, now=t + 1)
        assert first.usable
        assert again.verdict is Verdict.DUPLICATE and not again.usable

    def test_duplicates_are_counted_not_silently_dropped(self, intake):
        """A silent drop is indistinguishable from a sensor that stopped, and
        those two demand opposite responses."""
        t = _steady(intake)
        for _ in range(3):                      # seq 700 is unused by _steady
            intake.offer({"machine_id": "M1", "ts": t, "seq": 700}, now=t)
        assert intake.stats.duplicate == 2      # first accepted, two rejected

    def test_the_same_reading_from_two_machines_is_not_a_duplicate(self, intake):
        _steady(intake)
        a = intake.offer({"machine_id": "A", "ts": NOW, "seq": 1}, now=NOW)
        b = intake.offer({"machine_id": "B", "ts": NOW, "seq": 1}, now=NOW)
        assert a.usable and b.usable


class TestOrdering:
    def test_slightly_late_is_accepted(self, intake):
        """Two gateways with different latency is normal, not a fault."""
        t = _steady(intake)
        env = intake.offer({"machine_id": "M1", "ts": t - 5, "seq": 500}, now=t)
        assert env.usable

    def test_far_too_late_is_refused(self, intake):
        """The estimate has moved past it; folding it in drags state backwards."""
        t = _steady(intake)
        env = intake.offer({"machine_id": "M1", "ts": t - 600, "seq": 501}, now=t)
        assert env.verdict is Verdict.TOO_LATE
        assert "lateness bound" in env.reason

    def test_one_machines_lateness_does_not_gate_another(self, intake):
        t = _steady(intake)
        intake.offer({"machine_id": "M1", "ts": t + 100, "seq": 1}, now=t + 100)
        other = intake.offer({"machine_id": "M2", "ts": t, "seq": 1}, now=t + 100)
        assert other.usable


class TestClocks:
    def test_an_event_from_the_future_is_refused(self, intake):
        """A future timestamp poisons any watermark that trusts it — every
        honest message after it then looks late."""
        t = _steady(intake)
        env = intake.offer({"machine_id": "M1", "ts": t + 9999, "seq": 3}, now=t)
        assert env.verdict is Verdict.FUTURE
        assert "clock is wrong" in env.reason

    @pytest.mark.parametrize("stamp", [
        1_700_000_500,                    # epoch seconds
        1_700_000_500_000,                # epoch milliseconds
        "2023-11-14T22:21:40Z",           # ISO with zone
        "2023-11-14T22:21:40",            # ISO naive
    ])
    def test_common_timestamp_shapes_are_understood(self, intake, stamp):
        env = intake.offer({"machine_id": "N", "ts": stamp, "seq": 1}, now=1_700_000_501)
        assert env.usable

    def test_an_unparseable_timestamp_does_not_raise(self, intake):
        """A malformed message from a field gateway is data, not an exception."""
        env = intake.offer({"machine_id": "M1", "ts": "yesterday-ish"}, now=NOW)
        assert env.usable          # arrival time substituted, and recorded

    def test_a_message_with_no_machine_is_malformed(self, intake):
        env = intake.offer({"ts": NOW}, now=NOW)
        assert env.verdict is Verdict.MALFORMED and not env.usable


class TestGapsAndStaleness:
    def test_a_hole_is_flagged_not_smoothed_over(self, intake):
        """A slope computed across a gap is a trend that never happened."""
        t = _steady(intake)
        env = intake.offer({"machine_id": "M1", "ts": t + 200, "seq": 900}, now=t + 200)
        assert env.verdict is Verdict.GAP_BEFORE
        assert env.usable                      # usable, but the hole is visible
        assert "samples are missing" in env.reason

    def test_an_old_reading_is_marked_stale(self, intake):
        t = _steady(intake)
        env = intake.offer({"machine_id": "M1", "ts": t, "seq": 901}, now=t + 5000)
        assert env.stale
        assert "not a current reading" in env.reason

    def test_transport_latency_is_measured(self, intake):
        t = _steady(intake)
        env = intake.offer({"machine_id": "M1", "ts": t, "seq": 902}, now=t + 12)
        assert env.transport_latency_s == pytest.approx(12.0, abs=0.1)


class TestConnectors:
    def test_csv_source_is_marked_verified(self, tmp_path):
        p = tmp_path / "x.csv"
        p.write_text("machine_id,torque_nm\nM1,40\nM1,41\n")
        src = CsvSource(path=p)
        assert src.info().verified
        rows = list(src.read())
        assert len(rows) == 2 and "ts" in rows[0]

    def test_protocol_adapters_declare_themselves_unverified(self):
        """They are written against the real client libraries but have never met
        a broker or a PLC. Claiming otherwise would be the same error as a
        constant described as measured when it was chosen."""
        for src in (MqttSource(host="h"), OpcUaSource(endpoint="opc.tcp://x")):
            assert src.info().verified is False
            assert src.info().requires

    def test_a_missing_client_library_says_how_to_fix_it(self):
        src = MqttSource(host="nowhere.invalid")
        try:
            import paho.mqtt.client  # noqa: F401
            pytest.skip("paho-mqtt is installed")
        except ImportError:
            pass
        with pytest.raises(RuntimeError, match="pip install"):
            next(src.read())

    def test_tag_mapping_is_explicit(self):
        """Guessing that SITE1.LINE3.SPINDLE.TQ_FB means torque_nm is how a
        system confidently reads the wrong sensor."""
        tm = TagMap(tags={"SITE1.SPINDLE.TQ_FB": "torque_nm"}, machine_from="ASSET")
        out = tm.apply({"SITE1.SPINDLE.TQ_FB": 41.2, "ASSET": "M7", "NOISE": 1})
        assert out == {"torque_nm": 41.2, "machine_id": "M7"}

    def test_a_corrupt_jsonl_line_does_not_stop_the_stream(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text('{"machine_id":"M1"}\n{oops\n{"machine_id":"M2"}\n')
        rows = list(JsonlSource(path=p).read())
        assert len(rows) == 3 and "_malformed" in rows[1]

    def test_unknown_source_kind_is_named(self):
        with pytest.raises(ValueError, match="unknown source kind"):
            build_source({"kind": "carrier-pigeon"})


class TestRankingFalsification:
    """The premise gates test the user's claims. This tests ours."""

    def test_a_ranking_that_is_pure_chance_is_identified(self):
        """These 12 machines are assigned round-robin, so no machine effect can
        exist. A reader still sees a 3.6 point spread and acts on it."""
        machines = ((7, 215), (2, 205), (3, 196), (4, 199), (5, 188), (48, 1191),
                    (50, 1202), (34, 1188), (48, 1213), (55, 1206), (9, 594), (21, 593))
        spread, p, median_null = spread_vs_chance(machines)
        assert p > 0.05
        assert median_null > 2.0            # chance alone produces most of it

    def test_a_real_effect_survives_the_null(self):
        """H 2.09%, M 2.97%, L 3.92% is documented and real. Do not cry wolf."""
        variants = ((21, 1003), (89, 2997), (235, 6000))
        _, p, _ = spread_vs_chance(variants)
        assert p < 0.05

    def test_the_bigger_spread_is_the_fake_one(self):
        """Why this cannot be left to judgement.

        Machines spread 3.58 points and are noise; variants spread 1.82 points
        and are real. The eye ranks them the wrong way round, because it cannot
        see that group sizes differ.
        """
        machines = ((7, 215), (2, 205), (3, 196), (4, 199), (5, 188), (48, 1191),
                    (50, 1202), (34, 1188), (48, 1213), (55, 1206), (9, 594), (21, 593))
        variants = ((21, 1003), (89, 2997), (235, 6000))
        m_spread, m_p, _ = spread_vs_chance(machines)
        v_spread, v_p, _ = spread_vs_chance(variants)
        assert m_spread > v_spread          # machines look worse
        assert m_p > 0.05 > v_p             # variants are the real finding

    def test_p_value_is_never_zero(self):
        """2,000 resamples cannot justify p = 0; the honest floor is 1/2001."""
        _, p, _ = spread_vs_chance(((0, 500), (400, 500)))
        assert p > 0

    def test_degenerate_inputs_do_not_raise(self):
        assert spread_vs_chance(())[1] == 1.0
        assert spread_vs_chance(((5, 100),))[1] == 1.0
        assert spread_vs_chance(((0, 100), (0, 100)))[1] == 1.0

    def test_it_is_fast_enough_for_the_query_path(self):
        """The obvious implementation shuffled every row per permutation and took
        the suite from 4.6s to 102s. Sampling group counts from the multivariate
        hypergeometric is the same null at a fraction of the work."""
        import time

        groups = tuple((i * 3, 500) for i in range(1, 16))
        start = time.perf_counter()
        spread_vs_chance(groups, seed=12345)
        assert (time.perf_counter() - start) < 0.25


class TestRankingWarningEndToEnd:
    @pytest.fixture(scope="class")
    def engine(self):
        return Engine.build()

    def test_the_machine_ranking_is_flagged_as_chance(self, engine):
        answer = engine.ask("failure rate by machine", SessionState())
        codes = {w.code for w in answer.bundle.warnings}
        assert "ranking_is_chance" in codes

    def test_the_variant_ranking_is_not_flagged(self, engine):
        answer = engine.ask(
            "What's the failure rate by product variant?", SessionState()
        )
        codes = {w.code for w in answer.bundle.warnings}
        assert "ranking_is_chance" not in codes

    def test_the_evidence_is_in_slots_so_the_verifier_can_check_it(self, engine):
        answer = engine.ask("failure rate by machine", SessionState())
        for slot in ("ranking.spread_points", "ranking.p_value",
                     "ranking.chance_spread_points"):
            assert slot in answer.bundle.slots

    def test_the_warning_reads_as_english(self, engine):
        answer = engine.ask("failure rate by machine", SessionState())
        message = next(
            w.message for w in answer.bundle.warnings if w.code == "ranking_is_chance"
        )
        assert "these machines" in message      # not "these machine"
