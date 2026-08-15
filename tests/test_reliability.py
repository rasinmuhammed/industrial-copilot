"""Phase 5: the reliability gates.

Adversarial by construction. Each test injects a fault and asserts the system
reaches the right verdict — these are the mechanisms that address the documented
reason 80% of industrial AI projects never leave pilot, which is data quality
and trust rather than algorithm accuracy.
"""

from __future__ import annotations

import pytest

from copilot.ingest import connect
from copilot.reliability import (
    UNKNOWN_PROVENANCE,
    Direction,
    Uncertainty,
    DriftVerdict,
    Verdict,
    audit_calibration,
    check_invariants,
    diagnose_drift,
    estimate_threshold,
    evaluate_interval,
)

BASE = dict(
    air_temp_k=300.0,
    process_temp_k=310.0,
    rotational_speed_rpm=1500.0,
    torque_nm=45.0,
    tool_wear_min=210.0,
    product_type="L",
)


@pytest.fixture(scope="module")
def con():
    """Read-only. Temporary views work fine on a read-only DuckDB connection, so
    the drift-injection tests need no write lock — which means the suite runs
    while the API server is up, and mirrors production, where every reader is
    read-only and only ingest writes."""
    return connect(read_only=True)


def _drifted(con, column: str, delta: float) -> str:
    con.execute(
        f"""CREATE OR REPLACE TEMP VIEW drifted AS
            SELECT * EXCLUDE (temp_delta_k, {column}),
                   {column} + {delta} AS {column},
                   process_temperature_k - (CASE WHEN '{column}' = 'air_temperature_k'
                       THEN air_temperature_k + {delta} ELSE air_temperature_k END) AS temp_delta_k
            FROM observations"""
    )
    return "drifted"


# --------------------------------------------------------------------------
# Interval margins — the three-state decision
# --------------------------------------------------------------------------


class TestIntervalMargins:
    def test_trusted_reading_gives_a_definite_verdict(self):
        m = evaluate_interval(**BASE, uncertainty=Uncertainty())
        assert m.verdict() is Verdict.SAFE
        assert m.overstrain.is_point

    def test_uncertain_torque_abstains_rather_than_alerting(self):
        """Outliers are the most frequent sensor error. A binary alerter turns
        them straight into false alarms."""
        m = evaluate_interval(**BASE, uncertainty=Uncertainty(torque_nm=8.0))
        assert m.verdict() is Verdict.ABSTAIN
        assert "OSF" in m.abstaining_rules()
        assert not m.firing_rules()

    def test_unknown_provenance_abstains_on_everything(self):
        m = evaluate_interval(**BASE, uncertainty=UNKNOWN_PROVENANCE)
        assert m.verdict() is Verdict.ABSTAIN
        assert set(m.abstaining_rules()) == {"HDF", "PWF", "OSF"}

    def test_a_genuine_violation_still_alerts_under_uncertainty(self):
        """Abstention must not swallow real failures."""
        point = {**BASE, "tool_wear_min": 300.0, "torque_nm": 60.0}
        m = evaluate_interval(**point, uncertainty=Uncertainty(torque_nm=1.0))
        assert m.verdict() is Verdict.ALERT
        assert "OSF" in m.firing_rules()

    def test_hdf_is_conjunctive_under_uncertainty(self):
        """One condition certainly satisfied makes the rule certainly safe,
        however uncertain the other is."""
        point = {**BASE, "process_temp_k": 330.0, "rotational_speed_rpm": 1200.0}
        m = evaluate_interval(**point, uncertainty=Uncertainty(rotational_speed_rpm=100.0))
        assert m.rule_verdicts()["HDF"] is Verdict.SAFE

    def test_interval_width_tracks_input_uncertainty(self):
        narrow = evaluate_interval(**BASE, uncertainty=Uncertainty(torque_nm=1.0))
        wide = evaluate_interval(**BASE, uncertainty=Uncertainty(torque_nm=10.0))
        assert wide.overstrain.width > narrow.overstrain.width


# --------------------------------------------------------------------------
# Gate 2 — instrument honesty
# --------------------------------------------------------------------------


class TestInvariants:
    def test_all_invariants_hold_on_the_published_data(self, con):
        for result in check_invariants(con):
            assert result.holds, f"{result.code}: {result.observed} vs {result.expected}"

    def test_baseline_shows_no_drift(self, con):
        report = diagnose_drift(con, window_where="TRUE", baseline_table="observations")
        assert report.verdict is DriftVerdict.OK

    def test_air_sensor_drift_is_diagnosed_as_an_instrument_fault(self, con):
        """The headline case: a 0.4 K drift HALVES heat-dissipation alerts, which
        a conventional copilot reports as an improvement."""
        table = _drifted(con, "air_temperature_k", -0.4)
        report = diagnose_drift(con, window_where="TRUE", table=table,
                                baseline_table="observations")
        assert report.verdict is DriftVerdict.SENSOR
        assert abs(report.z_temp_delta) > 5
        assert abs(report.z_rotational_speed) < 5
        assert "instrument fault" in report.explanation

        fires = con.execute(
            f"SELECT count(*) FROM {table} "
            "WHERE temp_delta_k < 8.6 AND rotational_speed_rpm < 1380"
        ).fetchone()[0]
        assert fires < 115 * 0.6  # alerts collapse; the plant would look healthier

    def test_process_slowdown_is_diagnosed_as_operations(self, con):
        """Same symptom class, opposite cause."""
        table = _drifted(con, "rotational_speed_rpm", -40.0)
        report = diagnose_drift(con, window_where="TRUE", table=table,
                                baseline_table="observations")
        assert report.verdict is DriftVerdict.PROCESS
        assert abs(report.z_rotational_speed) > 5

    def test_thermodynamic_invariant_catches_impossible_readings(self, con):
        con.execute(
            """CREATE OR REPLACE TEMP VIEW impossible AS
               SELECT * EXCLUDE (process_temperature_k),
                      air_temperature_k - 1.0 AS process_temperature_k
               FROM observations"""
        )
        i1 = next(r for r in check_invariants(con, table="impossible") if r.code == "I1")
        assert not i1.holds


# --------------------------------------------------------------------------
# Gate 3 — knowledge-base staleness
# --------------------------------------------------------------------------


class TestCalibrationMonitor:
    def test_published_rules_are_perfectly_calibrated(self, con):
        report = audit_calibration(con)
        assert report.healthy
        for rule in report.rules:
            assert rule.total_signal == 0

    @pytest.mark.parametrize("error,expected", [(0.02, Direction.TOO_LOOSE),
                                                (-0.02, Direction.TOO_TIGHT)])
    def test_a_perturbed_threshold_is_detected_with_direction(self, con, error, expected):
        """Which counter fires tells you which way to move the threshold."""
        con.execute(
            f"""CREATE OR REPLACE TEMP VIEW perturbed AS
                SELECT * EXCLUDE (osf_rule),
                       (overstrain_min_nm > osf_threshold_min_nm * (1 + {error})) AS osf_rule
                FROM observations"""
        )
        report = audit_calibration(con, table="perturbed")
        osf = next(r for r in report.rules if r.mode == "OSF")
        assert osf.direction is expected
        assert not report.healthy

    def test_signal_is_monotone_in_the_size_of_the_error(self, con):
        signals = []
        for error in (0.01, 0.02, 0.05):
            con.execute(
                f"""CREATE OR REPLACE TEMP VIEW perturbed AS
                    SELECT * EXCLUDE (osf_rule),
                           (overstrain_min_nm > osf_threshold_min_nm * (1 + {error})) AS osf_rule
                    FROM observations"""
            )
            report = audit_calibration(con, table="perturbed")
            signals.append(next(r for r in report.rules if r.mode == "OSF").total_signal)
        assert signals == sorted(signals)

    def test_advice_names_the_corrective_direction(self, con):
        con.execute(
            """CREATE OR REPLACE TEMP VIEW perturbed AS
               SELECT * EXCLUDE (osf_rule),
                      (overstrain_min_nm > osf_threshold_min_nm * 1.05) AS osf_rule
               FROM observations"""
        )
        osf = next(r for r in audit_calibration(con, table="perturbed").rules if r.mode == "OSF")
        assert "tightened" in osf.advice()


class TestThresholdDiscovery:
    @pytest.mark.parametrize("variant,documented,tolerance_pct", [
        ("L", 11000, 0.1),
        ("M", 12000, 2.0),
        ("H", 13000, 4.0),
    ])
    def test_brackets_the_documented_threshold(self, con, variant, documented, tolerance_pct):
        """Recovers the rule from outcomes alone. Tolerance widens with variant
        rarity, which is the point: the bracket IS the uncertainty."""
        lower, upper, midpoint, support = estimate_threshold(
            con, metric_column="overstrain_min_nm", label_column="osf", product_type=variant
        )
        assert lower < documented < upper
        assert abs(midpoint - documented) / documented * 100 < tolerance_pct
        assert support > 0

    def test_bracket_narrows_with_more_support(self, con):
        """L has 87 supporting failures, H has 2. The widths must reflect that."""
        widths = {}
        for variant in ("L", "H"):
            lower, upper, _, _ = estimate_threshold(
                con, metric_column="overstrain_min_nm", label_column="osf", product_type=variant
            )
            widths[variant] = upper - lower
        assert widths["L"] < widths["H"]
