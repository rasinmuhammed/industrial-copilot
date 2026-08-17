"""Regime detection: learn which mode of operation a machine is in.

This is the one place machine learning belongs in this system. Physics computes
the numbers and they must be exact; but "which of several recipes is running
right now" has no ground truth, nobody to label it, and a need to discover modes
never seen before. That is unsupervised segmentation, and it decides which
baseline applies rather than supplying any figure.

The design took three attempts and each failure is recorded here, because the
failures are the reason the final shape is what it is.
"""

from __future__ import annotations

import duckdb
import pytest

from copilot.config import settings
from copilot.regime import (
    CONFIRM_CYCLES,
    RADIUS,
    REGIME_AXES,
    Regime,
    RegimeStatus,
    RegimeTracker,
)


@pytest.fixture(scope="module")
def scale() -> list[float]:
    """Per-axis variance from CHANNEL noise, which is what sets the radius."""
    con = duckdb.connect(str(settings().db_path), read_only=True)
    sd = con.execute(
        "SELECT stddev_samp(rotational_speed_rpm), stddev_samp(torque_nm), "
        "stddev_samp(temp_delta_k) FROM observations"
    ).fetchone()
    return [v * v for v in sd]


@pytest.fixture(scope="module")
def cycles() -> list[tuple]:
    con = duckdb.connect(str(settings().db_path), read_only=True)
    return con.execute(
        "SELECT rotational_speed_rpm, torque_nm, temp_delta_k "
        "FROM observations ORDER BY udi LIMIT 2000"
    ).fetchall()


def _run(cycles, scale, mutate=None, at=900):
    tracker = RegimeTracker(scale=scale)
    first_new = None
    for i, row in enumerate(cycles):
        reading = dict(zip(REGIME_AXES, row))
        if mutate and i >= at:
            mutate(reading)
        verdict = tracker.observe(reading)
        if i >= at and verdict.status is RegimeStatus.NEW and first_new is None:
            first_new = i - at
    return tracker, first_new


class TestTheRadiusIsNotLearned:
    """Two earlier designs tried to learn it. Both were unstable."""

    def test_the_radius_comes_from_the_error_budget(self):
        """A chi-square quantile at a stated false-new rate over three axes -
        not a number anybody picked."""
        assert 15.0 < RADIUS < 18.0

    def test_a_regimes_spread_is_supplied_not_accumulated(self):
        """Accumulating the cluster's own variance is unstable in both
        directions. Let it grow and a regime that absorbs a few neighbouring
        points widens, absorbs more, and swallows the space - measured: five
        regimes found in two-regime data, flapping every few cycles. Freeze it
        early instead and a variance from forty samples is too tight, so the
        regime rejects its own future points and shatters into eleven.

        The radius is a property of the CHANNEL, not of the cluster.
        """
        regime = Regime(label="R1", scale=[100.0, 4.0, 1.0])
        for value in ([1500, 40, 10], [1520, 42, 11], [1480, 38, 9]):
            regime.update(value)
        assert regime.variance() == [100.0, 4.0, 1.0]      # unchanged by data

    def test_the_mean_still_tracks_slow_movement(self):
        regime = Regime(label="R1", scale=[100.0, 4.0, 1.0])
        for _ in range(50):
            regime.update([1000, 40, 10])
        assert regime.mean[0] == pytest.approx(1000, rel=1e-6)


class TestSteadyProductionIsOneMode:
    """A detector that finds five regimes in single-regime data is broken."""

    def test_unchanging_production_yields_exactly_one_regime(self, cycles, scale):
        """AI4I is one recipe. rpm and torque are coupled at r = -0.875, so the
        data lies on a curved manifold with large natural spread - precisely the
        shape that a naive clustering carves into arbitrary chunks.
        """
        tracker, _ = _run(cycles, scale)
        assert len(tracker.regimes) == 1

    def test_ordinary_scatter_stays_well_inside_the_radius(self, cycles, scale):
        tracker, _ = _run(cycles[:500], scale)
        regime = next(iter(tracker.regimes.values()))
        distances = [regime.distance(list(row)) for row in cycles[500:900]]
        typical = sorted(distances)[len(distances) // 2]
        assert typical < RADIUS / 2


class TestAChangeoverIsFound:
    def test_a_recipe_change_creates_a_second_regime(self, cycles, scale):
        tracker, _ = _run(
            cycles, scale,
            lambda r: r.update(rotational_speed_rpm=r["rotational_speed_rpm"] * 0.55,
                               torque_nm=r["torque_nm"] * 1.8),
        )
        assert len(tracker.regimes) == 2
        speeds = sorted(r.mean[0] for r in tracker.regimes.values())
        assert speeds[0] < 1000 < speeds[1]      # two genuinely distinct modes

    def test_detection_needs_sustained_evidence(self, cycles, scale):
        """A single wild point is a transient; a run of them is a recipe.

        Requiring persistence is what stops a changeover spawning a phantom
        regime for the seconds it takes the line to settle.
        """
        tracker, _ = _run(cycles, scale)          # unchanged production
        assert len(tracker.regimes) == 1
        assert CONFIRM_CYCLES >= 5

    def test_a_lone_outlier_does_not_create_a_regime(self, cycles, scale):
        tracker = RegimeTracker(scale=scale)
        for i, row in enumerate(cycles[:600]):
            reading = dict(zip(REGIME_AXES, row))
            if i == 400:
                reading["rotational_speed_rpm"] = 20_000     # one wild sample
            tracker.observe(reading)
        assert len(tracker.regimes) == 1

    def test_overlapping_modes_take_longer_and_that_is_honest(self, cycles, scale):
        """Detection lag is a function of separation, not a tunable.

        The simulated change here sits at d^2 = 24.5 against a radius of 16.27
        - distinct, but the two balls overlap, so post-change scatter keeps
        landing inside the old mode and resetting the counter. Measured lag was
        282 cycles rather than the 12 a clean break would give.

        Shortening the confirmation window would speed this up and reintroduce
        phantom regimes. The right response is to state the property.
        """
        _, lag = _run(
            cycles, scale,
            lambda r: r.update(rotational_speed_rpm=r["rotational_speed_rpm"] * 0.55,
                               torque_nm=r["torque_nm"] * 1.8),
        )
        assert lag is not None


class TestItSaysWhatItDoesNotKnow:
    """An unknown mode has no baseline, so it must not be judged against one."""

    def test_a_new_mode_reports_that_it_is_learning(self, scale):
        tracker = RegimeTracker(scale=scale)
        verdict = tracker.observe(dict(zip(REGIME_AXES, (1500, 40, 10))))
        assert verdict.status is RegimeStatus.NEW
        assert not verdict.usable
        assert "no baseline" in verdict.reason

    def test_a_missing_axis_yields_a_transition_not_a_guess(self, scale):
        tracker = RegimeTracker(scale=scale)
        verdict = tracker.observe(
            {"rotational_speed_rpm": 1500, "torque_nm": None, "temp_delta_k": 10}
        )
        assert verdict.status is RegimeStatus.TRANSITION
        assert not verdict.usable

    def test_a_known_changeover_is_announced_as_planned(self, cycles, scale):
        """Entering a mode we already understand is a normal event. Reporting it
        as a fault is how a control room learns to ignore the system."""
        tracker, _ = _run(
            cycles, scale,
            lambda r: r.update(rotational_speed_rpm=r["rotational_speed_rpm"] * 0.55,
                               torque_nm=r["torque_nm"] * 1.8),
        )
        assert len(tracker.regimes) == 2
        # Returning to the original setpoints must not invent a third mode.
        for row in cycles[:200]:
            tracker.observe(dict(zip(REGIME_AXES, row)))
        assert len(tracker.regimes) == 2

    def test_it_refuses_to_shard_without_limit(self, scale):
        """More modes than a plant plausibly runs means the axes are wrong."""
        from copilot.regime import MAX_REGIMES

        tracker = RegimeTracker(scale=[1.0, 1.0, 1.0])
        for i in range(MAX_REGIMES * CONFIRM_CYCLES * 3):
            tracker.observe(dict(zip(REGIME_AXES, (i * 500.0, i * 500.0, i * 500.0))))
        assert len(tracker.regimes) <= MAX_REGIMES


class TestTheObserverUsesIt:
    """The payoff: a changeover must read as planned, not as a fleet of faults."""

    def _stream(self, scale, change_at=1200):
        import duckdb

        from copilot.config import settings
        from copilot.observer import FaultKind, FleetObserver

        con = duckdb.connect(str(settings().db_path), read_only=True)
        rows = con.execute(
            "SELECT air_temperature_k, process_temperature_k, rotational_speed_rpm, "
            "torque_nm, tool_wear_min FROM observations ORDER BY udi LIMIT 2000"
        ).fetchall()
        observer = FleetObserver(regime_scale=scale)
        faults, announced = 0, None
        for i, r in enumerate(rows):
            reading = {
                "machine_id": "M1", "air_temp_k": r[0], "process_temp_k": r[1],
                "rotational_speed_rpm": r[2], "torque_nm": r[3],
                "tool_wear_min": r[4], "temp_delta_k": r[1] - r[0],
            }
            if i >= change_at:
                reading["rotational_speed_rpm"] *= 0.55
                reading["torque_nm"] *= 1.8
            report = observer.observe(reading)
            if i >= change_at and report.fault_kind is FaultKind.INSTRUMENT:
                faults += 1
            if i >= change_at and announced is None and "changeover" in report.explanation:
                announced = i - change_at
        return faults, announced

    def test_a_changeover_stops_looking_like_a_fleet_of_sensor_faults(self, scale):
        """Without regime tracking a recipe change moves every channel at once
        and the observer reports each as an instrument fault - a screen of
        alarms for a planned event, which is the fastest way to lose a control
        room. Measured: 212 fault ticks without, 58 with."""
        without, _ = self._stream(None)
        with_regimes, _ = self._stream(scale)
        assert with_regimes < without / 2

    def test_the_changeover_is_announced_in_words(self, scale):
        _, announced = self._stream(scale)
        assert announced is not None

    def test_channels_relearn_for_the_new_mode(self, scale):
        """Noise identified in one recipe does not describe another. Carrying it
        across judges the new mode against the old one's spread."""
        from copilot.observer import MachineObserver

        observer = MachineObserver(
            machine_id="M1", regimes=RegimeTracker(scale=scale)
        )
        channel = observer.channels["torque_nm"]
        channel.calibrated = True
        channel.learning.extend([1.0, 2.0, 3.0])
        channel.reset_for_new_regime()
        assert not channel.calibrated
        assert not channel.learning

