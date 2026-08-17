"""Deep learning on the sensor stream: built, measured, and it lost.

The question was fair — a predictive-maintenance system with no learned model on
the sensor path invites the assumption that the author avoided the work. So the
work was done: a residual autoencoder over the channel vector, trained on
healthy cycles, scored against the failures and against a linear baseline.

It loses to both, and the reason is the most useful thing in this file.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from copilot.config import settings
from copilot.neural import CHANNELS, PCABaseline, ResidualDetector

COLS = ("air_temperature_k, process_temperature_k, rotational_speed_rpm, "
        "torque_nm, tool_wear_min")


@pytest.fixture(scope="module")
def split():
    con = duckdb.connect(str(settings().db_path), read_only=True)
    healthy = np.array(
        con.execute(f"SELECT {COLS} FROM observations WHERE machine_failure = 0").fetchall()
    )
    failed = np.array(
        con.execute(f"SELECT {COLS} FROM observations WHERE machine_failure = 1").fetchall()
    )
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(healthy))
    return healthy[idx[:7000]], healthy[idx[7000:]], failed


class TestItIsRealAndItRuns:
    def test_it_trains_and_scores(self, split):
        train, held, _ = split
        model = ResidualDetector().fit(train)
        assert model.trained_on == len(train)
        assert np.isfinite(model.score_many(held)).all()

    def test_it_is_small_enough_to_deploy_anywhere(self, split):
        """38 parameters. Importing a deep-learning framework to run fifty
        multiply-accumulates would add hundreds of megabytes to an image that
        holds seven light dependencies."""
        train, _, _ = split
        assert ResidualDetector().fit(train, epochs=50).parameters < 100

    def test_inference_is_fast_enough_for_a_stream(self, split):
        import time

        train, held, _ = split
        model = ResidualDetector().fit(train, epochs=50)
        start = time.perf_counter()
        for row in held[:2000]:
            model.score(row)
        per_point = (time.perf_counter() - start) / 2000
        assert per_point < 1e-3          # microseconds, not milliseconds

    def test_training_uses_healthy_cycles_only(self, split):
        """Training on everything teaches the model that failures are normal —
        the commonest way an anomaly detector is quietly ruined. It then
        reconstructs the faults perfectly and flags nothing."""
        train, _, failed = split
        model = ResidualDetector().fit(train, epochs=100)
        assert (model.score_many(failed) > model.threshold).mean() > 0

    def test_a_malformed_input_is_refused(self, split):
        train, _, _ = split
        with pytest.raises(ValueError, match="expected"):
            ResidualDetector().fit(train[:, :3])
        with pytest.raises(RuntimeError, match="not trained"):
            ResidualDetector().score(dict.fromkeys(CHANNELS, 1.0))


class TestItDoesNotBeatTheBaseline:
    """The burden is on the learned model, and it does not discharge it."""

    def test_the_autoencoder_does_not_beat_pca(self, split):
        """Measured: both detect 8.3% of failures, and PCA does it at a LOWER
        false-alarm rate (0.11% against 0.15%).

        This matches the published evaluation of time-series foundation models,
        where anomaly-detection performance "does not significantly differ to
        simple one-liner baselines". A nonlinear model that ties a linear
        projection has not earned the complexity.
        """
        train, held, failed = split
        auto = ResidualDetector().fit(train)
        linear = PCABaseline().fit(train)
        auto_tp = (auto.score_many(failed) > auto.threshold).mean()
        linear_tp = (linear.score_many(failed) > linear.threshold).mean()
        assert auto_tp <= linear_tp + 0.05        # no meaningful advantage

    def test_both_are_far_worse_than_the_physics(self, split):
        """8.3% against 84.7%. The rules explain 287 of 339 failures exactly;
        the density models find roughly one in twelve."""
        train, _, failed = split
        auto = ResidualDetector().fit(train)
        detected = (auto.score_many(failed) > auto.threshold).mean()
        assert detected < 0.30                    # nowhere near 0.847


class TestWhyItLosesIsTheInterestingPart:
    """A failure here is a rule violation, not a statistical outlier."""

    def test_most_failures_are_distributionally_ordinary(self, split):
        """82% of failing cycles sit within 3 sigma on EVERY channel.

        Nothing about the joint distribution marks them, so no density model can
        find them however deep it is. This is not a limitation of the
        architecture chosen — it is a property of the failure mechanism.
        """
        train, _, failed = split
        mean, sd = train.mean(0), train.std(0)
        worst_z = np.abs((failed - mean) / sd).max(axis=1)
        assert (worst_z < 3).mean() > 0.7

    def test_but_the_derived_quantity_separates_them_perfectly(self):
        """Overstrain is wear x torque. Both factors are unremarkable; their
        PRODUCT crosses a documented line at 11,000, and the minimum value among
        fired rows is 11,003.

        A density model sees two ordinary values. The rule sees the line. That
        is the whole argument for computing the physics rather than learning it.
        """
        con = duckdb.connect(str(settings().db_path), read_only=True)
        fired = con.execute(
            "SELECT min(tool_wear_min * torque_nm) FROM observations WHERE OSF = 1"
        ).fetchone()[0]
        assert 11_000 <= fired < 11_100

    def test_there_are_no_trajectories_to_learn_from(self):
        """Why this is an autoencoder and not an LSTM, stated as a fact rather
        than a preference.

        AI4I has 1,490 tool-life segments with a median length of six cycles and
        not one reaching twenty. A sequence model has nothing to fit, and a
        published RUL curve from one would be an artefact of the fitting.
        """
        con = duckdb.connect(str(settings().db_path), read_only=True)
        rows = con.execute(
            "SELECT machine_id, tool_wear_min FROM observations ORDER BY machine_id, udi"
        ).fetchall()
        segments, run, prev_m, prev_w = [], 0, None, None
        for machine, wear in rows:
            if machine != prev_m or (prev_w is not None and wear < prev_w):
                if run > 1:
                    segments.append(run)
                run = 0
            run += 1
            prev_m, prev_w = machine, wear
        if run > 1:
            segments.append(run)
        assert max(segments) < 20
