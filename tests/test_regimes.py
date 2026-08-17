"""Three regimes, and the system always says which one it is in.

An adversarial technical review found, at high confidence, that the margin
paradigm assumes a discrete failure boundary and that only 15-20% of real
industrial failure modes have one. The finding was correct.

The remedy tested here is not the reviewer's - they proposed discarding margins
entirely for conformal survival prediction, which over-corrects. Where a
documented boundary exists an exact margin is strictly better than a
probabilistic interval, because it inverts into a setpoint. The error was never
having margins; it was applying them everywhere while letting the reader assume
one confidence level throughout.
"""

from __future__ import annotations

import duckdb
import pytest

from copilot.process_model import load_process_model
from copilot.regimes import (
    ConformalInterval,
    Regime,
    classify_modes,
    conformal_interval,
    empirical_coverage,
    regime_for,
)


@pytest.fixture(scope="module")
def twf_wear() -> list[float]:
    con = duckdb.connect()
    con.execute("CREATE VIEW t AS SELECT * FROM read_csv_auto('data/ai4i2020.csv')")
    return [r[0] for r in con.execute(
        'SELECT "Tool wear [min]" FROM t WHERE TWF = 1'
    ).fetchall()]


class TestTheRegimeWasAlreadyDeclared:
    """The split has been in failure_modes.yaml since the first commit.

    HDF/PWF/OSF are marked `deterministic`, TWF/RNF `stochastic`, and RNF
    carries no predicate at all. The answer path simply never read it.
    """

    def test_documented_boundaries_are_exact(self):
        model = load_process_model()
        for code in ("HDF", "PWF", "OSF"):
            assert regime_for(model.mode(code)).regime is Regime.EXACT

    def test_a_window_without_a_crisp_edge_is_statistical(self):
        """TWF fails somewhere inside a wear window. There is no line to measure
        distance to, so a margin would be a fabricated precision."""
        model = load_process_model()
        verdict = regime_for(model.mode("TWF"))
        assert verdict.regime is Regime.STATISTICAL
        assert "no crisp edge" in verdict.why

    def test_a_mode_with_no_predicate_is_irreducible(self):
        """RNF is parameter-independent by the dataset's own construction. Not
        hard to predict - impossible. A system that offers a number here is
        fabricating one."""
        model = load_process_model()
        verdict = regime_for(model.mode("RNF"))
        assert verdict.regime is Regime.IRREDUCIBLE
        assert "no predicate" in verdict.why

    def test_every_mode_is_classified(self):
        verdicts = classify_modes()
        assert {v.mode for v in verdicts} == {"HDF", "PWF", "OSF", "TWF", "RNF"}
        assert all(v.why for v in verdicts)

    def test_a_mode_flag_is_not_a_failure(self):
        """19 rows carry the RNF flag; exactly ONE of them failed.

        An earlier version of this module claimed the non-exact share was "TWF
        (46) plus RNF (19) = 19.2%", counting flag-rows as failures and
        inflating the irreducible share nineteen-fold. Precisely the error this
        project exists to prevent, made by its author, caught by measuring.
        """
        con = duckdb.connect()
        con.execute("CREATE VIEW t AS SELECT * FROM read_csv_auto('data/ai4i2020.csv')")
        flagged = con.execute("SELECT count(*) FROM t WHERE RNF = 1").fetchone()[0]
        actual = con.execute(
            'SELECT count(*) FROM t WHERE RNF = 1 AND "Machine failure" = 1'
        ).fetchone()[0]
        assert flagged == 19 and actual == 1

    def test_the_statistical_regime_dominates_the_discovery_gap(self):
        """Rule discovery reaches 80.8%; the remainder is mostly TWF at 13.6%.

        That gap was being read as a shortfall in the discovery algorithm. It is
        the algorithm correctly declining to invent a threshold for failures
        that do not have one.
        """
        con = duckdb.connect()
        con.execute("CREATE VIEW t AS SELECT * FROM read_csv_auto('data/ai4i2020.csv')")
        twf, total = con.execute(
            'SELECT sum(TWF), sum("Machine failure") FROM t'
        ).fetchone()
        assert 0.13 < twf / total < 0.14


class TestConformalGuarantee:
    """Distribution-free and finite-sample exact, or it is not a guarantee."""

    @pytest.mark.parametrize("alpha", [0.20, 0.10, 0.05])
    def test_coverage_holds_on_real_data(self, twf_wear, alpha):
        """Leave-one-out. A guarantee that is never tested is a claim."""
        rate, _covered, n = empirical_coverage(twf_wear, alpha)
        assert n == 46
        assert rate >= 1 - alpha, f"achieved {rate:.3f}, promised {1 - alpha:.3f}"

    def test_conformal_is_conservative_not_tight(self, twf_wear):
        """Conformal over-covers by construction. Achieving exactly 1-alpha
        would suggest the implementation is wrong, not that it is efficient."""
        rate, _, _ = empirical_coverage(twf_wear, 0.10)
        assert 0.90 <= rate <= 1.0

    def test_it_refuses_a_level_the_sample_cannot_support(self, twf_wear):
        """46 observations cannot support a 99% two-sided interval. The honest
        response is to say so, not to return a narrower interval than the
        evidence permits."""
        interval = conformal_interval(twf_wear, 0.01)
        assert not interval.valid
        assert conformal_interval(twf_wear, 0.10).valid

    def test_the_interval_brackets_the_documented_window(self, twf_wear):
        """The dataset documents TWF between 200 and 240 minutes. Nothing here
        reads that; it is recovered from 46 failures."""
        interval = conformal_interval(twf_wear, 0.10)
        assert interval.lower <= 200
        assert interval.upper >= 240

    def test_no_distributional_assumption_is_made(self):
        """The guarantee must survive a shape no parametric model would fit."""
        bimodal = [1.0] * 30 + [100.0] * 30
        rate, _, _ = empirical_coverage(bimodal, 0.10)
        assert rate >= 0.90

    def test_a_wider_interval_is_returned_for_a_stronger_claim(self, twf_wear):
        loose = conformal_interval(twf_wear, 0.20)
        tight = conformal_interval(twf_wear, 0.05)
        assert (tight.upper - tight.lower) >= (loose.upper - loose.lower)

    def test_degenerate_inputs_fail_loudly(self):
        with pytest.raises(ValueError, match="at least one"):
            conformal_interval([])
        with pytest.raises(ValueError, match="alpha"):
            conformal_interval([1.0, 2.0], alpha=1.5)

    def test_a_single_observation_reports_what_it_can(self):
        interval = conformal_interval([42.0])
        assert interval.lower == interval.upper == 42.0
        assert not interval.valid


class TestTheClaimIsStated:
    """An interval without its epistemic status is just two numbers."""

    def test_each_regime_names_what_it_entitles_us_to_claim(self):
        assert "exact" in Regime.EXACT.claim
        assert "coverage" in Regime.STATISTICAL.claim
        assert "not predictable" in Regime.IRREDUCIBLE.claim

    def test_the_sentence_states_the_guarantee_and_its_assumption(self, twf_wear):
        text = conformal_interval(twf_wear, 0.10, unit="min").sentence("Tool failure")
        assert "at least 90%" in text
        assert "distribution-free" in text
        assert "future tools resemble past ones" in text   # the one assumption
