"""Foundation tests: units, plan validation, and the first two operators.

These lock in the guarantees the architecture rests on. A failure here is not a
cosmetic regression — it means a stated property no longer holds.
"""

from __future__ import annotations

import pytest

from copilot.evidence import Interval, Quality, Slot
from copilot.ingest import connect
from copilot.ir import PlanError, ValidationStage, parse_plan
from copilot.ops import ExecutionContext, data_fingerprint, execute, kb_version
from copilot.stats import is_reportable, wilson_interval
from copilot.units import UnitError, assert_compatible, convert, derived, unit


@pytest.fixture(scope="module")
def ctx() -> ExecutionContext:
    con = connect()
    return ExecutionContext(con=con, kb_version=kb_version(), data_version=data_fingerprint(con))


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------


class TestUnits:
    def test_absolute_conversion_applies_offset(self):
        assert convert(25, "degC", "K") == pytest.approx(298.15)

    def test_difference_conversion_skips_offset(self):
        assert convert(10, "ΔdegC", "ΔK") == pytest.approx(10.0)

    def test_rpm_to_rad_per_second(self):
        assert convert(1380, "rpm", "rad/s") == pytest.approx(144.5133, abs=1e-3)

    @pytest.mark.parametrize(
        "alias,canonical",
        [("Nm", "N·m"), ("°C", "degC"), ("rev/min", "rpm"), ("watts", "W"), ("pct", "%")],
    )
    def test_aliases(self, alias, canonical):
        assert unit(alias).symbol == canonical

    def test_torque_times_angular_velocity_is_power(self):
        """Holds precisely because radians are dimensionless."""
        assert derived(("N·m", 1), ("rad/s", 1)) == unit("W").dimension
        assert derived(("N·m", 1), ("rpm", 1)) == unit("W").dimension

    def test_wear_times_torque_is_strain(self):
        assert derived(("min", 1), ("N·m", 1)) == unit("min·N·m").dimension

    @pytest.mark.parametrize("a,b", [("K", "N·m"), ("rpm", "W"), ("min", "N·m")])
    def test_incompatible_dimensions_rejected(self, a, b):
        with pytest.raises(UnitError):
            assert_compatible(a, b)

    def test_absolute_versus_difference_rejected(self):
        """A process temperature and a temperature rise are different kinds."""
        with pytest.raises(UnitError, match="absolute quantity with a difference"):
            assert_compatible("K", "ΔK")

    def test_unknown_unit_is_loud(self):
        with pytest.raises(UnitError):
            unit("furlongs")


# --------------------------------------------------------------------------
# Plan validation — the hallucination barrier
# --------------------------------------------------------------------------


class TestPlanValidation:
    def test_valid_compare_plan(self):
        plan = parse_plan(
            {
                "op": "compare",
                "cohorts": [
                    {"name": "failed", "filters": [{"field": "failure", "op": "=", "value": 1}]},
                    {"name": "healthy", "filters": [{"field": "failure", "op": "=", "value": 0}]},
                ],
                "metrics": ["torque_nm"],
            }
        )
        assert plan.hash

    def test_plan_hash_is_stable(self):
        payload = {"op": "rate", "group_by": ["product_type"]}
        assert parse_plan(payload).hash == parse_plan(dict(payload)).hash

    @pytest.mark.parametrize(
        "payload,stage",
        [
            ({"op": "describe", "metrics": ["vibration_rms"]}, ValidationStage.VOCABULARY),
            ({"op": "rate", "group_by": ["operator_name"]}, ValidationStage.VOCABULARY),
            ({"op": "describe", "metrics": ["torque_Nm"]}, ValidationStage.VOCABULARY),
            ({"op": "describe"}, ValidationStage.VIABILITY),
            (
                {"op": "compare", "metrics": ["torque_nm"], "cohorts": [{"name": "a"}]},
                ValidationStage.VIABILITY,
            ),
            (
                {"op": "compare", "metrics": ["torque_nm"],
                 "cohorts": [{"name": "a"}, {"name": "a"}]},
                ValidationStage.VIABILITY,
            ),
            (
                {"op": "counterfactual", "params": {"changes": {"pressure": -5}}},
                ValidationStage.OP_SPECIFIC,
            ),
            (
                {"op": "sql_explore", "params": {"sql": "DROP TABLE observations"}},
                ValidationStage.OP_SPECIFIC,
            ),
        ],
    )
    def test_rejected_at_expected_stage(self, payload, stage):
        with pytest.raises(PlanError) as exc:
            parse_plan(payload)
        assert exc.value.stage is stage

    def test_invented_op_rejected_structurally(self):
        with pytest.raises(PlanError) as exc:
            parse_plan({"op": "predict_failure"})
        assert exc.value.stage is ValidationStage.STRUCTURAL

    def test_delta_metric_cannot_take_absolute_unit(self):
        """temp_delta_k is a difference; comparing it to an absolute K is a bug."""
        with pytest.raises(PlanError) as exc:
            parse_plan(
                {
                    "op": "describe",
                    "metrics": ["temp_delta_k"],
                    "filters": [{"field": "temp_delta_k", "op": "<", "value": 8.6, "unit": "K"}],
                }
            )
        assert exc.value.stage is ValidationStage.DIMENSIONAL

    def test_delta_metric_accepts_delta_unit(self):
        parse_plan(
            {
                "op": "describe",
                "metrics": ["temp_delta_k"],
                "filters": [{"field": "temp_delta_k", "op": "<", "value": 8.6, "unit": "ΔK"}],
            }
        )

    def test_synthetic_dimensions_are_tracked(self):
        plan = parse_plan({"op": "rate", "group_by": ["shift"], "time_grain": "day"})
        assert plan.synthetic_used() == ["shift", "ts"]

    def test_repair_prompt_names_the_stage(self):
        with pytest.raises(PlanError) as exc:
            parse_plan({"op": "describe", "metrics": ["vibration_rms"]})
        assert "vocabulary" in exc.value.repair_prompt()


# --------------------------------------------------------------------------
# Evidence and statistics
# --------------------------------------------------------------------------


class TestEvidence:
    def test_slot_id_encodes_cohort(self):
        assert Slot(id="failed.torque_nm.mean", value=50.2, unit="N·m").cohort == "failed"

    def test_abstained_slot_renders_as_words_not_a_number(self):
        slot = Slot(id="all.torque_nm.mean", value=None, quality=Quality.ABSTAIN)
        assert slot.render() == "not determined"
        assert not any(ch.isdigit() for ch in slot.render())

    def test_slot_rejects_unknown_unit(self):
        with pytest.raises(ValueError):
            Slot(id="a.b.c", value=1.0, unit="furlongs")

    def test_slot_must_have_value_unless_abstaining(self):
        with pytest.raises(ValueError):
            Slot(id="a.b.c", value=None)

    @pytest.mark.parametrize(
        "lo,hi,expected",
        [(-100, -10, "negative"), (10, 100, "positive"), (-10, 100, "straddles")],
    )
    def test_interval_verdict_drives_abstention(self, lo, hi, expected):
        assert Interval(lo=lo, hi=hi).verdict() == expected


class TestStats:
    def test_wilson_interval_stays_in_bounds_at_zero(self):
        ci = wilson_interval(0, 50)
        assert ci.lo == 0.0 and 0 < ci.hi < 1

    def test_wilson_interval_brackets_the_estimate(self):
        ci = wilson_interval(339, 10000)
        assert ci.lo < 0.0339 < ci.hi

    def test_low_sample_is_not_reportable(self):
        ok, reason = is_reportable(0.042, Interval(lo=0.01, hi=0.09), n=12)
        assert not ok and reason == "low_sample"

    def test_wide_interval_is_not_reportable(self):
        ok, reason = is_reportable(0.05, Interval(lo=0.001, hi=0.20), n=500)
        assert not ok and reason == "wide_interval"


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------


class TestDescribe:
    def test_cohort_means_are_namespaced_and_differ(self, ctx):
        bundle = execute(
            parse_plan(
                {
                    "op": "describe",
                    "cohorts": [
                        {"name": "failed", "filters": [{"field": "failure", "op": "=", "value": 1}]},
                        {"name": "healthy", "filters": [{"field": "failure", "op": "=", "value": 0}]},
                    ],
                    "metrics": ["torque_nm"],
                }
            ),
            ctx,
        )
        failed = bundle.slots["failed.torque_nm.mean"]
        healthy = bundle.slots["healthy.torque_nm.mean"]
        assert failed.n == 339 and healthy.n == 9661
        assert failed.value > healthy.value
        assert bundle.cohorts() == {"failed", "healthy"}

    def test_empty_cohort_abstains(self, ctx):
        bundle = execute(
            parse_plan(
                {
                    "op": "describe",
                    "metrics": ["torque_nm"],
                    "filters": [{"field": "tool_wear_min", "op": ">", "value": 999}],
                }
            ),
            ctx,
        )
        assert bundle.slots["all.torque_nm.mean"].quality is Quality.ABSTAIN
        assert bundle.abstained == ["all.torque_nm.mean"]

    def test_provenance_is_complete(self, ctx):
        bundle = execute(parse_plan({"op": "describe", "metrics": ["torque_nm"]}), ctx)
        p = bundle.provenance
        assert p.plan_hash and p.kb_version and p.data_version
        assert p.row_count == 10000 and p.elapsed_ms > 0


class TestRate:
    def test_overall_failure_rate_matches_dataset(self, ctx):
        bundle = execute(parse_plan({"op": "rate"}), ctx)
        assert bundle.slots["all.failures"].value == 339
        assert bundle.slots["all.n"].value == 10000
        assert bundle.slots["all.failure_rate"].value == pytest.approx(3.39, abs=0.01)

    def test_grouped_rates_carry_intervals(self, ctx):
        bundle = execute(parse_plan({"op": "rate", "group_by": ["product_type"]}), ctx)
        for variant in ("l", "m", "h"):
            slot = bundle.slots[f"{variant}.failure_rate"]
            assert slot.ci is not None and slot.ci.contains(slot.value)

    def test_premise_refutation_on_the_briefs_example_question(self, ctx):
        """'Why more failures at high rpm?' is false: the curve is U-shaped."""
        bundle = execute(
            parse_plan(
                {
                    "op": "rate",
                    "bin": {"field": "rotational_speed_rpm", "method": "quantile", "bins": 5},
                }
            ),
            ctx,
        )
        assert bundle.slots["premise.shape"].value == "U-shaped"
        assert bundle.slots["premise.first_group_rate"].value == pytest.approx(12.17, abs=0.01)
        assert bundle.slots["premise.last_group_rate"].value == pytest.approx(2.24, abs=0.01)
        assert bundle.slots["premise.low_high_ratio"].value == pytest.approx(5.4, abs=0.1)
        assert any(w.code == "premise_refuted" for w in bundle.warnings)

    def test_synthetic_dimension_forces_a_disclosure(self, ctx):
        bundle = execute(parse_plan({"op": "rate", "group_by": ["shift"]}), ctx)
        assert bundle.provenance.synthetic_dimensions == ["shift"]
        assert any(w.code == "synthetic_dimension" for w in bundle.warnings)


class TestCompare:
    def test_answers_the_briefs_third_example_question(self, ctx):
        bundle = execute(
            parse_plan(
                {
                    "op": "compare",
                    "cohorts": [
                        {"name": "failed", "filters": [{"field": "failure", "op": "=", "value": 1}]},
                        {"name": "healthy", "filters": [{"field": "failure", "op": "=", "value": 0}]},
                    ],
                    "metrics": ["torque_nm", "tool_wear_min"],
                }
            ),
            ctx,
        )
        assert bundle.slots["failed.n"].value == 339
        assert bundle.slots["healthy.n"].value == 9661
        # Failed cycles run hotter and more worn.
        assert bundle.slots["delta.torque_nm"].value > 0
        assert bundle.slots["delta.tool_wear_min"].value > 0
        # Torque separates the cohorts strongly.
        assert bundle.slots["effect.torque_nm.cohens_d"].value > 0.8
        assert bundle.slots["effect.torque_nm.magnitude"].value == "large"

    def test_delta_slots_carry_difference_units(self, ctx):
        """A delta must not be comparable against an absolute threshold."""
        bundle = execute(
            parse_plan(
                {
                    "op": "compare",
                    "cohorts": [
                        {"name": "a", "filters": [{"field": "product_type", "op": "=", "value": "L"}]},
                        {"name": "b", "filters": [{"field": "product_type", "op": "=", "value": "H"}]},
                    ],
                    "metrics": ["torque_nm"],
                }
            ),
            ctx,
        )
        assert bundle.slots["delta.torque_nm"].unit == "ΔN·m"

    def test_collinearity_is_detected_automatically(self, ctx):
        """r(rpm, torque) = -0.875: every rpm analysis here is confounded."""
        bundle = execute(
            parse_plan(
                {
                    "op": "compare",
                    "cohorts": [
                        {"name": "failed", "filters": [{"field": "failure", "op": "=", "value": 1}]},
                        {"name": "healthy", "filters": [{"field": "failure", "op": "=", "value": 0}]},
                    ],
                    "metrics": ["rotational_speed_rpm", "torque_nm"],
                }
            ),
            ctx,
        )
        r = bundle.slots["corr.rotational_speed_rpm__torque_nm"].value
        assert r == pytest.approx(-0.875, abs=0.005)
        assert any(w.code == "collinearity" for w in bundle.warnings)

    def test_empty_cohort_abstains_rather_than_comparing(self, ctx):
        bundle = execute(
            parse_plan(
                {
                    "op": "compare",
                    "cohorts": [
                        {"name": "real", "filters": [{"field": "product_type", "op": "=", "value": "L"}]},
                        {"name": "empty", "filters": [{"field": "tool_wear_min", "op": ">", "value": 999}]},
                    ],
                    "metrics": ["torque_nm"],
                }
            ),
            ctx,
        )
        assert bundle.slots["delta.torque_nm"].quality is Quality.ABSTAIN


def _by_udi(ctx, udi: int):
    return execute(
        parse_plan({"op": "root_cause", "filters": [{"field": "udi", "op": "=", "value": udi}]}),
        ctx,
    )


class TestRootCause:
    def test_orphan_failure_returns_undetermined(self, ctx):
        """9 rows are labelled failures with no documented mode. Tested for
        structure and found to have none — 'undetermined' is verified correct."""
        bundle = _by_udi(ctx, 9016)
        assert bundle.slots["cycle.failed"].value == "yes"
        assert bundle.slots["cause.verdict"].value == "cause_undetermined"
        assert any("cannot be determined" in w.message for w in bundle.warnings)

    def test_multi_mode_failure_reports_every_mode(self, ctx):
        """A single-label classifier must pick one and is wrong on the rest."""
        udi = ctx.con.execute(
            "SELECT udi FROM observations WHERE pwf=1 AND osf=1 LIMIT 1"
        ).fetchone()[0]
        bundle = _by_udi(ctx, udi)
        assert bundle.slots["cause.mode_count"].value == 2
        assert set(bundle.slots["cause.verdict"].value.split(" + ")) == {"PWF", "OSF"}
        assert bundle.slots["PWF.fired"].value == "yes"
        assert bundle.slots["OSF.fired"].value == "yes"

    def test_osf_reports_the_crossing_point(self, ctx):
        udi = ctx.con.execute(
            "SELECT udi FROM observations WHERE osf=1 AND pwf=0 AND hdf=0 LIMIT 1"
        ).fetchone()[0]
        bundle = _by_udi(ctx, udi)
        crossing = bundle.slots["OSF.crossing_wear_min"].value
        wear = bundle.slots["TWF.tool_wear"].value
        assert wear > crossing  # it crossed before the observed wear
        assert bundle.slots["OSF.exceeded_by_min"].value == pytest.approx(wear - crossing, abs=1e-6)

    def test_twf_is_a_probability_never_a_certainty(self, ctx):
        udi = ctx.con.execute(
            "SELECT udi FROM observations WHERE tool_wear_min BETWEEN 200 AND 240 LIMIT 1"
        ).fetchone()[0]
        bundle = _by_udi(ctx, udi)
        assert bundle.slots["TWF.in_window"].value == "yes"
        assert bundle.slots["TWF.failure_probability"].value == pytest.approx(5.4, abs=0.1)

    def test_healthy_row_explains_the_non_event_at_rule_level(self, ctx):
        """HDF is conjunctive: being below 1380 rpm alone is not an approach to
        failure. Distance must be per-rule, never per-condition."""
        udi = ctx.con.execute(
            "SELECT udi FROM observations WHERE machine_failure=0 "
            "ORDER BY worst_normalised_margin DESC LIMIT 1"
        ).fetchone()[0]
        bundle = _by_udi(ctx, udi)
        assert bundle.slots["cause.verdict"].value == "no failure mode triggered"
        assert bundle.slots["closest.normalised_distance"].value >= 0

    def test_no_healthy_row_has_a_negative_rule_distance(self, ctx):
        """Self-validation of the margin definition itself."""
        bad = ctx.con.execute(
            "SELECT count(*) FROM observations WHERE machine_failure=0 "
            "AND least(hdf_distance, pwf_distance, osf_distance) < 0"
        ).fetchone()[0]
        assert bad == 0

    def test_exactly_the_deterministic_failures_have_negative_distance(self, ctx):
        negative = ctx.con.execute(
            "SELECT count(*) FROM observations WHERE machine_failure=1 "
            "AND least(hdf_distance, pwf_distance, osf_distance) < 0"
        ).fetchone()[0]
        assert negative == 287

    def test_cohort_attribution_totals(self, ctx):
        bundle = execute(parse_plan({"op": "root_cause"}), ctx)
        assert bundle.slots["cohort.n"].value == 10000
        assert bundle.slots["cohort.failures"].value == 339
        assert bundle.slots["HDF.count"].value == 115
        assert bundle.slots["PWF.count"].value == 95
        assert bundle.slots["OSF.count"].value == 98
        assert bundle.slots["orphans.count"].value == 9
        assert bundle.slots["multi_mode.count"].value == 23

    def test_rnf_is_never_attributed(self, ctx):
        udi = ctx.con.execute("SELECT udi FROM observations WHERE rnf=1 LIMIT 1").fetchone()[0]
        bundle = _by_udi(ctx, udi)
        assert any("RNF" in w.message and "cannot be attributed" in w.message
                   for w in bundle.warnings)
