"""The instrument layer, tested against the faults it exists to catch.

Every case here corresponds to a bug found while building the module, or to a
published sensor-fault class. Several of these tests exist because the first
implementation got the answer confidently wrong — those are marked, because a
test whose provenance is a real mistake is worth more than one written from
imagination.
"""

from __future__ import annotations

import math

import duckdb
import pytest

from copilot.observer import (
    CALIBRATION,
    CUSUM_H,
    FALSE_STUCK_RATE,
    REDUNDANT_CHANNELS,
    WARMUP,
    ChannelKind,
    FaultKind,
    FleetObserver,
    MachineObserver,
    NE107,
    cusum_threshold_for_arl,
)

READY = WARMUP + CALIBRATION


@pytest.fixture(scope="module")
def cycles() -> list[dict]:
    con = duckdb.connect()
    con.execute("CREATE VIEW t AS SELECT * FROM read_csv_auto('data/ai4i2020.csv')")
    rows = con.execute(
        'SELECT "Air temperature [K]", "Process temperature [K]", '
        '"Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]" '
        "FROM t ORDER BY UDI"
    ).fetchall()
    return [
        {
            "machine_id": "M1",
            "air_temp_k": r[0],
            "process_temp_k": r[1],
            "rotational_speed_rpm": r[2],
            "torque_nm": r[3],
            "tool_wear_min": r[4],
        }
        for r in rows
    ]


def _run(cycles, mutate=None, limit=2200, start=800):
    obs = FleetObserver()
    reports = []
    for i, base in enumerate(cycles[:limit]):
        row = dict(base)
        if mutate is not None and i >= start:
            mutate(row, i - start)
        reports.append(obs.observe(row))
    return obs, reports


def _first_failure(reports, channel, start=800):
    for i, rep in enumerate(reports):
        if i < start:
            continue
        if rep.channels[channel].status is NE107.FAILURE:
            return i - start
        if any(channel in p.channels for p in rep.parity if p.violated):
            return i - start
    return None


class TestDerivedThresholds:
    """No constant in the module may be chosen by taste."""

    def test_cusum_threshold_matches_the_published_arl(self):
        """k=0.5, h=5 is 465 cycles in every SPC text. Ours must agree.

        The first implementation had the sign of the in-control drift wrong,
        which made the inversion run into its bisection ceiling and return
        h = 50 — a chart that would never alarm.
        """
        h = cusum_threshold_for_arl(465.0, k=0.5)
        assert 4.8 < h < 5.2

    def test_cusum_threshold_is_monotone_in_the_budget(self):
        assert cusum_threshold_for_arl(500) < cusum_threshold_for_arl(5000)

    def test_threshold_is_a_consequence_of_the_budget_not_a_constant(self):
        assert CUSUM_H == pytest.approx(cusum_threshold_for_arl(2000.0), rel=1e-9)


class TestNoiseIdentification:
    """The noise model is measured from the signal, never declared."""

    def test_recovers_the_documented_torque_sigma_from_data_alone(self, cycles):
        """AI4I documents torque as N(40, 10). Nothing tells the observer that.

        The first version hard-coded a torque sigma of 0.30 and a temperature
        sigma six times too large, with a comment claiming both were measured.
        That single misdescription flagged 94% of healthy cycles as frozen
        sensors — a constant chosen but described as derived is exactly the
        failure this project exists to prevent, and it still got in.
        """
        obs, _ = _run(cycles, limit=WARMUP + 1)
        torque = obs.observers["M1"].channels["torque_nm"]
        # 9.44 against a documented 10.0 — within 6% from 200 samples, with no
        # knowledge of the generating process.
        assert math.sqrt(torque.r) == pytest.approx(10.0, rel=0.10)

    def test_white_noise_channels_identify_as_zero_process_noise(self, cycles):
        """Torque is iid about a constant, not a random walk. q = 0 is correct."""
        obs, _ = _run(cycles, limit=WARMUP + 1)
        torque = obs.observers["M1"].channels["torque_nm"]
        assert torque.q < torque.r


class TestColdStart:
    """A new asset is abstained on, not guessed at."""

    def test_channel_is_untrusted_until_calibrated(self, cycles):
        _, reports = _run(cycles, limit=READY)
        assert not reports[0].trusted
        assert not reports[WARMUP + 10].trusted
        assert "learning" in reports[5].channels["torque_nm"].reason

    def test_becomes_trusted_after_calibration(self, cycles):
        _, reports = _run(cycles, limit=READY + 60)
        assert reports[-1].trusted


class TestFaultsDetectableFromOneChannel:
    """Freeze, dropout and invalid values violate the noise model itself."""

    def test_hard_freeze_is_caught_within_a_handful_of_cycles(self, cycles):
        held: dict[str, float] = {}

        def freeze(row, _):
            held.setdefault("v", row["torque_nm"])
            row["torque_nm"] = held["v"]

        _, reports = _run(cycles, freeze)
        delay = _first_failure(reports, "torque_nm")
        assert delay is not None and delay <= 8

    def test_a_frozen_sensor_is_an_instrument_fault_not_a_process_fault(self, cycles):
        """The distinction the whole module exists to make."""
        held: dict[str, float] = {}

        def freeze(row, _):
            held.setdefault("v", row["torque_nm"])
            row["torque_nm"] = held["v"]

        _, reports = _run(cycles, freeze)
        fired = [r for r in reports[810:] if not r.trusted]
        assert fired and fired[0].fault_kind is FaultKind.INSTRUMENT
        assert "dispatch to the sensor" in fired[0].explanation

    def test_dropout_is_detected_and_does_not_crash(self, cycles):
        _, reports = _run(cycles, lambda row, _: row.update(torque_nm=None))
        assert _first_failure(reports, "torque_nm") is not None

    def test_nan_is_treated_as_missing_rather_than_raising(self, cycles):
        """float('nan') used to propagate straight into the margin arithmetic."""
        _, reports = _run(cycles, lambda row, _: row.update(torque_nm=float("nan")))
        assert _first_failure(reports, "torque_nm") is not None

    def test_a_garbage_string_is_missing_data_not_an_exception(self, cycles):
        _, reports = _run(cycles, lambda row, _: row.update(torque_nm="n/a"))
        assert _first_failure(reports, "torque_nm") is not None

    def test_staleness_widens_uncertainty_instead_of_needing_a_flag(self, cycles):
        _, reports = _run(cycles, lambda row, _: row.update(torque_nm=None))
        before = reports[799].uncertainty.torque_nm
        after = reports[900].uncertainty.torque_nm
        assert after > before


class TestFaultsRequiringRedundancy:
    """A sensor cannot detect its own bias. The module must not pretend."""

    def test_thermal_decoupling_is_caught_immediately(self, cycles):
        """Two thermocouples guard each other, so bias IS detectable here."""
        _, reports = _run(
            cycles, lambda row, _: row.update(process_temp_k=row["process_temp_k"] + 6)
        )
        broken = [p for p in reports[801].parity if p.violated]
        assert broken and broken[0].name == "thermal_coupling"
        assert abs(broken[0].z) > 5

    def test_unguarded_channels_are_declared_not_silently_uncovered(self, cycles):
        """The honest output: name what cannot be protected.

        A local level estimator absorbs a bias step as a genuine process change,
        so torque bias is undetectable in principle from torque alone. An
        earlier version of the injection harness appeared to catch it — but that
        was a coincident baseline false alarm on a different channel, which is
        precisely how this class of mistake survives into production.
        """
        _, reports = _run(cycles, limit=READY + 10)
        note = reports[-1].detectability_note()
        assert "torque_nm" in reports[-1].unguarded_channels
        assert "not detectable" in note
        assert set(REDUNDANT_CHANNELS) == {"air_temp_k", "process_temp_k"}


class TestCounters:
    """A counter is not a level, and testing it as one is a category error."""

    def test_wear_is_classified_as_a_counter(self):
        obs = MachineObserver(machine_id="X")
        assert obs.channels["tool_wear_min"].kind is ChannelKind.COUNTER
        assert obs.channels["torque_nm"].kind is ChannelKind.LEVEL

    def test_an_idle_counter_is_not_a_frozen_sensor(self, cycles):
        """Wear holding steady means the machine is idle, not that it is broken.

        Applying the level-model stuck test to wear produced a second wave of
        false alarms: its sawtooth resets inflate process noise to sd 21 min, so
        ordinary 2 min increments then read as a collapse.
        """
        _, reports = _run(cycles, lambda row, _: row.update(tool_wear_min=150.0))
        wear = [r.channels["tool_wear_min"] for r in reports[900:]]
        assert all(c.status is not NE107.FAILURE for c in wear)

    def test_wear_running_backwards_is_a_violation(self, cycles):
        _, reports = _run(
            cycles, lambda row, i: row.update(tool_wear_min=120.0 if i % 2 else 60.0)
        )
        assert any(
            p.name == "wear_monotonicity" and p.violated
            for r in reports[801:820]
            for p in r.parity
        )

    def test_a_reset_to_zero_is_a_tool_change_not_corruption(self, cycles):
        """Without this the system forecasts a crossing for a tool that is gone,
        and every legitimate maintenance action reads as a data fault."""
        _, reports = _run(
            cycles, lambda row, i: row.update(tool_wear_min=0.0 if i < 3 else 5.0 + i)
        )
        assert not any(
            p.name == "wear_monotonicity" for r in reports[800:830] for p in r.parity
        )


class TestFalseAlarmBudget:
    """The observer must run at its stated rate, not a tuned one."""

    def test_healthy_data_stays_within_the_declared_budget(self, cycles):
        _, reports = _run(cycles, limit=len(cycles))
        episodes, prev = 0, False
        for i, rep in enumerate(reports):
            untrusted = i >= READY and not rep.trusted
            if untrusted and not prev:
                episodes += 1
            prev = untrusted
        n = len(reports) - READY
        # Union bound across the three tests on four level channels.
        budget = (FALSE_STUCK_RATE * n + 1.0 / 2000.0 * n) * 4
        assert episodes <= budget, f"{episodes} episodes over {n} cycles"

    def test_a_single_gated_outlier_is_robustness_not_a_fault(self, cycles):
        """Gating one spike and carrying on is the filter working.

        Treating it as a fault raised the false-alarm rate from 16 episodes to
        226 on identical data.
        """
        spike_at = READY + 20
        obs, reports = _run(
            cycles,
            lambda row, i: row.update(torque_nm=row["torque_nm"] + 400) if i == 0 else None,
            limit=READY + 120,
            start=spike_at,
        )
        # The spike itself may be gated; the very next cycles must recover.
        assert any(r.trusted for r in reports[spike_at + 1:spike_at + 5])


class TestUncertaintyHandoff:
    """The instrument layer meets the arithmetic layer here."""

    def test_untrusted_channel_yields_a_width_that_forces_abstention(self, cycles):
        held: dict[str, float] = {}

        def freeze(row, _):
            held.setdefault("v", row["torque_nm"])
            row["torque_nm"] = held["v"]

        _, reports = _run(cycles, freeze)
        healthy = reports[799].uncertainty.torque_nm
        frozen = reports[-1].uncertainty.torque_nm
        assert frozen > healthy

    def test_healthy_uncertainty_is_finite_and_small(self, cycles):
        _, reports = _run(cycles, limit=READY + 100)
        u = reports[-1].uncertainty
        assert 0 < u.torque_nm < 50
        assert math.isfinite(u.air_temp_k)
