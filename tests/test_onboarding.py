"""Onboarding a process is an algorithm, not a consulting engagement.

The moat around Argus platforms is integration labour: every new plant is a
project in which somebody interviews engineers and hand-builds a config. These
tests assert that the labour is automatable AND that the automation is honest
about what it could not establish.

The second half matters more than the first. A discovery tool that quietly
presents guesses as physics is worse than no tool, because it launders a
hypothesis into an authority.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "onboard", Path(__file__).resolve().parent.parent / "scripts" / "onboard.py"
)
onboard_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["onboard"] = onboard_mod
_SPEC.loader.exec_module(onboard_mod)

CSV = Path("data/ai4i2020.csv")
LABEL = "Machine failure"


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    out = tmp_path_factory.mktemp("onboard") / "discovered.yaml"
    return onboard_mod.onboard(CSV, LABEL, out), out


class TestChannelProfiling:
    """Channel character is worked out from the signal, not declared."""

    def test_counters_and_levels_are_distinguished(self, report):
        rep, _ = report
        kinds = {onboard_mod._slug(c.name): c.kind for c in rep.channels}
        assert kinds["tool_wear"] == "counter"
        assert kinds["torque"] == "level"

    def test_noise_identification_recovers_the_documented_torque_sigma(self, report):
        """AI4I documents torque as N(40, 10). Nothing here is told that."""
        rep, _ = report
        torque = next(c for c in rep.channels if onboard_mod._slug(c.name) == "torque")
        assert torque.instrument_sd == pytest.approx(10.0, rel=0.10)


class TestTargetLeakage:
    """The label decomposed is not a cause of the label."""

    def test_mode_flags_are_excluded(self, report):
        """AI4I ships TWF/HDF/PWF/OSF beside the failure flag.

        Handed those, a learner "discovers" that failure occurs when the failure
        flag is set. Leakage is among the most common ways an industrial model
        looks excellent in evaluation and useless in production, so it is caught
        by construction rather than by noticing the numbers are too good.
        """
        rep, _ = report
        assert set(rep.leaked) >= {"twf", "hdf", "pwf", "osf"}

    def test_leaked_columns_are_reported_not_silently_dropped(self, report):
        rep, out = report
        assert "target leakage" in out.read_text()


class TestRuleDiscovery:
    """Boundaries come back from the data, at documented accuracy."""

    def test_recovers_the_documented_boundaries(self, report):
        """The headline claim. Nothing upstream of the audit reads the docs."""
        rep, _ = report
        matches = onboard_mod.audit(rep)
        by_metric = {}
        for metric, got, want, err in matches:
            by_metric.setdefault(metric, []).append(err)

        assert min(by_metric["rotational_speed_rpm"]) < 0.01   # exact
        assert min(by_metric["overstrain_min_nm"]) < 0.1       # 11,000
        assert min(by_metric["power_w"]) < 0.5                 # 3,500 / 9,000
        assert min(by_metric["temp_delta_k"]) < 1.0            # 8.6 K

    def test_conjunctive_modes_are_found(self, report):
        """HDF is hot AND slow. A single cut cannot express it.

        The first implementation scored single conditions against the whole
        label and found nothing at all, on a dataset whose rules are fully
        recoverable — recall is structurally capped when the label is a union of
        modes and the rule explains only one of them.
        """
        rep, _ = report
        assert any(len(r.terms) >= 2 for r in rep.rules)

    def test_separate_and_conquer_beats_a_single_tree(self, report):
        """One tree partitions the space, so modes compete for splits.

        A single depth-3 fit reached 8.3% coverage. Removing each mode's
        failures and refitting surfaces the next one.
        """
        rep, _ = report
        assert rep.coverage > 0.75, f"coverage regressed to {rep.coverage:.1%}"

    def test_per_variant_limits_are_reachable(self, report):
        """The overstrain boundary differs by product type.

        No global cut expresses that, so the categoricals must be one-hot
        encoded or the largest documented mode is invisible.
        """
        rep, _ = report
        assert any(
            any(t.metric.startswith("type_is_") for t in r.terms) for r in rep.rules
        )


class TestHonestGrading:
    """What it cannot establish, it must say."""

    def test_verified_rules_have_no_false_alarms(self, report):
        rep, _ = report
        for rule in rep.rules:
            if rule.confidence == "verified":
                assert rule.false_alarms == 0

    def test_imperfect_rules_are_marked_for_review_not_asserted(self, report):
        rep, _ = report
        for rule in rep.rules:
            if rule.false_alarms:
                assert rule.confidence == "candidate"
                assert "engineer" in rule.note

    def test_stochastic_modes_are_left_uncovered_rather_than_faked(self, report):
        """TWF and RNF are random by construction.

        No threshold rule can find them, and a tool that claimed 100% coverage
        here would be lying. The gap is the honest answer.
        """
        rep, _ = report
        assert rep.coverage < 0.95

    def test_the_config_states_its_own_provenance(self, report):
        rep, out = report
        text = out.read_text()
        assert "provenance: discovered" in text
        assert "not documented, not reviewed" in text.lower()
        assert "confidence:" in text

    def test_every_rule_carries_its_evidence(self, report):
        rep, out = report
        text = out.read_text()
        assert "precision:" in text and "failures_explained:" in text


class TestEmittedConfigIsLoadable:
    """The output has to be a process definition, not a report."""

    def test_the_discovered_config_parses_as_a_process_model(self, report):
        from copilot.process_model import load_process_model

        _, out = report
        model = load_process_model(out)
        assert model.deterministic_modes

    def test_the_discovered_config_evaluates(self, report):
        """Closing the loop: discovered config -> the same engine that runs AI4I."""
        from copilot.process_model import load_process_model

        _, out = report
        model = load_process_model(out)
        reading = {
            "tool_wear_x_torque": 20_000.0,
            "type_is_l": 1.0,
            "power_from_torque_rotational_speed": 6_000.0,
            "rotational_speed": 1_500.0,
            "tool_wear": 150.0,
            "process_temperature_minus_air_temperature": 10.0,
            "air_temperature": 300.0,
        }
        fired = model.fires(reading)
        assert isinstance(fired, list)
