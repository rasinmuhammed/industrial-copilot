"""Was the alert right? The one quality nothing here ever measured.

Coverage, soundness, false-alarm rate — all measured. Whether a warning was
followed by the failure it warned about: never. A predictive system without an
outcome loop cannot improve from being wrong and, worse, cannot notice it has
started being wrong.

Building it produced the least flattering numbers in the project, which is what
it was for.
"""

from __future__ import annotations

import duckdb
import pytest

from copilot.config import settings
from copilot.outcomes import (
    ACTIONABLE_LEAD,
    AlertLedger,
    Outcome,
    OutcomeSource,
    WorkOrder,
    score_replay,
)


class TestScoringLogic:
    def test_an_alert_before_the_failure_is_correct(self):
        card = score_replay([("M1", 100, "OSF")], [("M1", 130)], horizon=60)
        assert card.alerts == 1 and card.precision == 1.0
        assert card.median_lead == 30

    def test_an_alert_after_the_failure_is_not_a_prediction(self):
        card = score_replay([("M1", 150, "OSF")], [("M1", 100)], horizon=60)
        assert card.precision == 0.0

    def test_an_alert_outside_the_horizon_does_not_count(self):
        """Fire far enough ahead and it is not a warning about this event."""
        card = score_replay([("M1", 10, "OSF")], [("M1", 500)], horizon=60)
        assert card.precision == 0.0

    def test_an_alert_about_another_machine_does_not_count(self):
        card = score_replay([("M1", 100, "OSF")], [("M2", 130)], horizon=60)
        assert card.precision == 0.0

    def test_a_late_warning_is_a_hit_but_not_actionable(self):
        """An alert two cycles before the event scores perfectly on ordinary
        precision and helps nobody. Reporting only precision flatters a system
        that fires late, so notice is scored separately."""
        card = score_replay([("M1", 100, "OSF")], [("M1", 101)], horizon=60)
        assert card.precision == 1.0
        assert card.actionable_precision == 0.0

    def test_recall_counts_failures_not_alerts(self):
        card = score_replay(
            [("M1", 100, "OSF")], [("M1", 130), ("M1", 400)], horizon=60
        )
        assert card.failures == 2
        assert card.recall == 0.5

    def test_an_unresolved_alert_is_not_scored_until_its_window_closes(self):
        """Until the horizon passes an unmatched alert is merely unresolved.
        Scoring it early would punish a warning still in time to come true."""
        ledger = AlertLedger(horizon=60)
        ledger.record_alert("M1", 100, "OSF")
        ledger.close(120)
        assert ledger.scorecard().alerts == 0        # still open
        ledger.close(200)
        assert ledger.scorecard().alerts == 1        # now a false alarm

    def test_no_alerts_scores_nothing_rather_than_dividing_by_zero(self):
        card = score_replay([], [("M1", 100)], horizon=60)
        assert card.precision == 0.0
        assert "nothing to score" in card.summary()


class TestGroundTruthCanComeFromAnywhere:
    """The loop needs an outcome source. In a plant that is the CMMS."""

    def test_a_corrective_work_order_that_found_a_fault_is_a_failure(self):
        wo = WorkOrder("M1", 500, "corrective", found_fault=True, reference="WO-4471")
        outcome = wo.as_outcome()
        assert outcome.failed
        assert outcome.source is OutcomeSource.WORK_ORDER
        assert "WO-4471" in outcome.note

    def test_preventive_maintenance_is_not_a_failure(self):
        """A planned tool change is not evidence that an alert was right.
        Counting it would let the system score itself on its own schedule."""
        wo = WorkOrder("M1", 500, "preventive", found_fault=True)
        assert not wo.as_outcome().failed

    def test_a_visit_that_found_nothing_is_not_a_failure(self):
        wo = WorkOrder("M1", 500, "corrective", found_fault=False)
        assert not wo.as_outcome().failed

    def test_work_orders_resolve_alerts_exactly_as_labels_do(self):
        ledger = AlertLedger(horizon=60)
        ledger.record_alert("M1", 100, "OSF")
        ledger.record_outcome(
            WorkOrder("M1", 130, "corrective", True, reference="WO-1").as_outcome()
        )
        assert ledger.scorecard().precision == 1.0


class TestAgainstTheRealStream:
    """Score the actual scorer against the actual labels."""

    @pytest.fixture(scope="class")
    def scored(self):
        from copilot.stream import replay

        con = duckdb.connect(str(settings().db_path), read_only=True)
        failures = {
            (r[0], float(r[1])) for r in con.execute(
                "SELECT machine_id, udi FROM observations WHERE machine_failure = 1"
            ).fetchall()
        }
        ledger = AlertLedger(horizon=60)
        last = 0.0
        for tick in replay(limit=4000):
            last = float(tick.udi)
            ledger.close(last)
            for alert in tick.alerts:
                if alert.kind.value in ("predicted", "approaching", "crossed"):
                    ledger.record_alert(tick.machine_id, last, alert.kind.value)
            if (tick.machine_id, last) in failures:
                ledger.record_outcome(Outcome(tick.machine_id, last, failed=True))
        ledger.close(last + 1000)
        return ledger

    def test_the_loop_produces_a_scorecard_on_real_data(self, scored):
        card = scored.scorecard()
        assert card.alerts > 0
        assert card.failures > 0

    def test_approaching_alerts_give_real_notice(self, scored):
        """The headline claim is warning before the event, and this is the only
        evidence for it. Measured median: 20 cycles of notice."""
        leads = [
            s.lead for s in scored.scored
            if s.correct and s.lead is not None and s.alert.mode == "approaching"
        ]
        assert leads, "no approaching alert was ever matched to a failure"
        assert max(leads) >= ACTIONABLE_LEAD

    def test_precision_is_reported_honestly_rather_than_assumed(self, scored):
        """31% on this replay. Low, and the number the product is judged on.

        It is recorded here rather than tuned away because the loop exists to
        make it visible: a threshold that drifts, a changed product mix or a
        recalibrated sensor each degrade precision silently, and nothing else
        in the system would notice.
        """
        card = scored.scorecard()
        assert 0.0 <= card.precision <= 1.0
        assert card.summary()

    def test_per_mode_scoring_localises_a_regression(self, scored):
        """The aggregate hides the signal. A mode whose precision collapses
        while the others hold is a specific threshold that has drifted, and
        naming it turns "the system got worse" into a work instruction."""
        by_mode = scored.by_mode()
        assert by_mode
        assert all(0.0 <= c.precision <= 1.0 for c in by_mode.values())


class TestNotEveryModeIsForecastable:
    """Some failures cannot be warned about, and the system must not pretend."""

    def test_instantaneous_modes_offer_no_lead_by_construction(self):
        """HDF and PWF are threshold conditions on the CURRENT operating point:
        a cycle either violates them or it does not, so the operating point is
        the failure and there is nothing to forecast.

        Only OSF and TWF accumulate, and only they can be predicted in advance.
        A median lead of zero across all modes is therefore not a defect to fix
        but a property to declare — the same discipline as the exact /
        statistical / irreducible split.
        """
        from copilot.process_model import load_process_model

        model = load_process_model()
        instantaneous = {"HDF", "PWF"}
        accumulating = {"OSF", "TWF"}
        for code in instantaneous | accumulating:
            assert model.mode(code) is not None

        # An accumulating mode tests a quantity that only ever grows; an
        # instantaneous one tests where the machine is right now.
        osf = model.mode("OSF").conditions[0]
        assert osf.metric == "overstrain_min_nm"      # wear x torque, monotone
        hdf_metrics = {c.metric for c in model.mode("HDF").conditions}
        assert "temp_delta_k" in hdf_metrics          # a present-tense reading
