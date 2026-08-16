"""The instrument layer: decide whether a reading deserves to be believed.

Everything else in this system computes on numbers. This module is the only
part that interrogates them.

WHY THIS EXISTS
---------------
`stream.score()` used to open with six `float()` calls on a raw dict. Downstream
sat signed margins, interval arithmetic, three-state verdicts and a fail-closed
renderer — rigorous arithmetic performed on unexamined inputs. That inverts the
value of the rigour: the more careful the downstream, the more confidently the
system asserts a conclusion drawn from a dead sensor.

The abstention rule made it worse. ABSTAIN fired when a margin's uncertainty
interval straddled zero, which is a statement about sensor *precision*. A stuck
sensor has zero noise, so its interval is tight and it never abstains. It
reports SAFE, with maximum confidence, forever. We had built the one abstention
rule that inverts on the most common sensor fault in industry.

WHAT IT DOES
------------
Per channel, per machine, every tick:

  1. A scalar Kalman filter (local level model) predicts the next value and
     produces an *innovation* — the surprise. Under a healthy sensor the
     normalised innovation is N(0, 1). This is observer-based FDI, the same
     construction used for GNSS/INS integrity monitoring.
  2. A chi-square test on realised variance detects a *stuck* channel. Stuck
     faults are hard precisely because their signature is the absence of noise,
     which every uncertainty model reads as confidence. Inverting the test
     turns that weakness into the detector.
  3. A CUSUM on the innovation sequence detects slow bias and drift. CUSUM is
     not a heuristic here: Moustakides proved it exactly optimal under Lorden's
     minimax criterion, so this is the provably earliest detection available
     for a given false-alarm budget.
  4. Missing data is handled by predicting without updating. The covariance
     grows, so staleness becomes a *quantity* that widens margins rather than a
     flag somebody has to remember to check.

Then, across channels, algebraic redundancy relations that must hold when every
instrument is honest.

NO THRESHOLD IN THIS FILE WAS CHOSEN
------------------------------------
Every constant is derived from a declared error budget:

  * the stuck threshold is a chi-square quantile at a stated false-positive rate
  * the CUSUM threshold is inverted from a target average run length via
    Siegmund's approximation — you specify "one false alarm per N cycles" and
    the threshold follows
  * the innovation gate is a normal quantile
  * the thermal parity sigma is measured from data (1.001 K), not assumed

This is the same discipline as threshold discovery: the constant comes from the
physics or from the error budget, never from the author's taste.

WHAT IT BUYS
------------
The residuals separate *instrument* faults from *process* faults — the
distinction that decides whether an operator ever trusts the system again.
Conflating the two is the origin of alarm fatigue. Splitting them lets the
copilot say "machine 7's torque channel is dead, dispatch to the sensor, and I
will not tell you whether the machine is safe" instead of inventing a number.

Statuses are emitted as NAMUR NE 107 categories, the vocabulary process plants
already use for device health, so the output routes into existing asset
management without translation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from scipy.stats import chi2, norm

from copilot.reliability.intervals import Uncertainty

__all__ = [
    "NE107",
    "FaultKind",
    "ChannelHealth",
    "ParityResidual",
    "TrustReport",
    "MachineObserver",
    "FleetObserver",
    "CHANNELS",
    "cusum_threshold_for_arl",
]

# ── Error budget. These are the only free parameters, and each is a stated
# rate rather than a magic number. Everything else is derived from them. ──────
FALSE_STUCK_RATE = 1e-4      # P(healthy channel declared stuck) per test
INNOVATION_ALPHA = 1e-3      # P(healthy sample gated as an outlier)
TARGET_ARL0 = 2000.0         # cycles between CUSUM false alarms, per channel
CUSUM_SHIFT_SIGMA = 1.0      # the drift size we want detected quickly, in sigma
STUCK_WINDOW = 24            # innovations in the quietness test
VAR_WINDOW = 64              # samples retained for dispersion
# A channel must not only be present, it must be present ENOUGH.
#
# Staleness was tracked as consecutive missing samples, escalating after three.
# An intermittently failing sensor never reaches three: dropping one reading in
# five resets the counter on every good sample, so a channel losing 20% of its
# data was reported healthy. Loose connectors, EMI and flaky links fail exactly
# this way, and they are far more common in a plant than a clean hard failure.
#
# Stated as an availability budget rather than a tuned constant: a channel
# delivering below this fraction of its expected samples is degrading, and well
# below it is unusable.
AVAILABILITY_FLOOR = 0.98
AVAILABILITY_FAILED = 0.80

WARMUP = 60                  # samples to identify the noise model
TOOL_CHANGE_RESET_MIN = 1.0  # wear at or below this after a drop = tool change
# P(k consecutive healthy readings all beyond the gate) = INNOVATION_ALPHA**k.
# At alpha = 1e-3, three in a row is 1e-9 — not chance.
GATED_RUN_LIMIT = 3
CALIBRATION = 40             # further samples to measure innovation energy

# The channels this observer tracks. Note what is NOT here: any noise constant.
#
# An earlier version declared a process and measurement sigma per channel,
# with a comment claiming they were measured. They were not — they were
# invented, and the temperature figures were about six times too large, which
# made healthy innovations tiny and flagged 94% of good cycles as frozen
# sensors. A constant described as derived but actually chosen is the exact
# failure this project exists to prevent, and it slipped in anyway.
#
# Noise is now identified from the signal itself. See _Channel.identify.
#
# Channels do carry a *kind*, because not every signal is the same sort of
# object and applying one validity test to all of them is a category error.
# A LEVEL is a physical quantity observed through noise: it should always be
# moving, so stillness is a fault. A COUNTER is a monotone accumulator —
# tool wear, run hours, a flow totaliser: it may legitimately sit unchanged for
# hours while a machine is idle, so the stuck test is meaningless on it and its
# validity comes from kinematics instead (it must never run backwards).
#
# Fitting a level model to a counter is what produced the second wave of false
# alarms here: wear's sawtooth resets inflate the process noise to sd 21 min,
# so ordinary 2 min increments then look like a frozen channel.


class ChannelKind(StrEnum):
    LEVEL = "level"
    COUNTER = "counter"


CHANNELS: dict[str, ChannelKind] = {
    "air_temp_k": ChannelKind.LEVEL,
    "process_temp_k": ChannelKind.LEVEL,
    "rotational_speed_rpm": ChannelKind.LEVEL,
    "torque_nm": ChannelKind.LEVEL,
    "tool_wear_min": ChannelKind.COUNTER,
}


class NE107(StrEnum):
    """NAMUR NE 107 device status. The vocabulary a plant already reads."""

    OK = "ok"
    FAILURE = "failure"                    # invalid signal; do not use
    FUNCTION_CHECK = "function_check"      # temporarily invalid, known cause
    OUT_OF_SPEC = "out_of_spec"            # working, but outside stated limits
    MAINTENANCE_REQUIRED = "maintenance"   # valid now, degrading


class FaultKind(StrEnum):
    """Which layer the problem is on. This is the whole point of the module."""

    NONE = "none"
    INSTRUMENT = "instrument"   # a channel is lying; dispatch to the sensor
    PROCESS = "process"         # signals agree; the machine is in trouble
    MODEL = "model"             # signals disagree with each other; model invalid


def cusum_threshold_for_arl(arl0: float, k: float = CUSUM_SHIFT_SIGMA / 2) -> float:
    """Invert Siegmund's ARL approximation to get the decision interval h.

    Siegmund (1985) for a one-sided CUSUM with drift delta:

        ARL ~= (exp(-2 d (h + 1.166)) + 2 d (h + 1.166) - 1) / (2 d^2)

    In control the drift is d = -k. We run a two-sided chart, and two
    independent one-sided charts alarm twice as often, so the two-sided ARL is
    half the one-sided value.

    Sanity: k = 0.5, h = 5 gives 465 cycles here, which is the textbook value.

    Monotone in h, so bisection is exact. The caller states a false-alarm
    budget; the threshold is a consequence of it, never a knob.
    """
    def two_sided_arl(h: float) -> float:
        d = -k
        b = 2 * d * (h + 1.166)
        return (math.exp(-b) + b - 1) / (2 * d * d) / 2.0

    lo, hi = 0.0, 200.0
    for _ in range(300):
        mid = (lo + hi) / 2
        if two_sided_arl(mid) < arl0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# Derived once, at import, from the budget above.
CUSUM_K = CUSUM_SHIFT_SIGMA / 2
CUSUM_H = cusum_threshold_for_arl(TARGET_ARL0, CUSUM_K)
INNOVATION_GATE = float(norm.ppf(1 - INNOVATION_ALPHA / 2))
STUCK_CHI2 = float(chi2.ppf(FALSE_STUCK_RATE, STUCK_WINDOW))


@dataclass(frozen=True, slots=True)
class ChannelHealth:
    """One sensor's epistemic state this tick."""

    name: str
    status: NE107
    value: float | None
    estimate: float
    posterior_sd: float
    innovation_z: float
    cusum: float
    stuck_score: float          # chi-square statistic; small means frozen
    age_cycles: int
    reason: str = ""

    @property
    def trusted(self) -> bool:
        """Is the *estimate* usable?

        OUT_OF_SPEC counts as trusted: the reading was rejected but the estimate
        survived precisely because the filter refused it. That is the gate
        working, not a fault. A sustained run of rejections escalates to FAILURE
        separately, and that is the condition that matters.
        """
        return self.status not in (NE107.FAILURE, NE107.FUNCTION_CHECK)

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.name,
            "status": self.status.value,
            "value": None if self.value is None else round(self.value, 3),
            "estimate": round(self.estimate, 3),
            "sd": round(self.posterior_sd, 4),
            "innovation_z": round(self.innovation_z, 2),
            "cusum": round(self.cusum, 2),
            "age_cycles": self.age_cycles,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ParityResidual:
    """An algebraic relation that must hold when every instrument is honest."""

    name: str
    residual: float
    sigma: float
    unit: str
    channels: tuple[str, ...]
    description: str

    @property
    def z(self) -> float:
        return self.residual / self.sigma if self.sigma > 0 else 0.0

    @property
    def violated(self) -> bool:
        return abs(self.z) > INNOVATION_GATE

    def as_dict(self) -> dict[str, Any]:
        return {
            "relation": self.name,
            "residual": round(self.residual, 3),
            "z": round(self.z, 2),
            "unit": self.unit,
            "violated": self.violated,
            "channels": list(self.channels),
        }


# Which fault classes a channel can actually be protected against.
#
# This is not a limitation to be buried; it is the most useful thing the module
# knows. A local level filter ADAPTS to a bias step, because for a genuine
# process a level change is exactly what it should track. So "the sensor shifted
# by 25 N.m" and "the process shifted by 25 N.m" produce identical innovations
# and are formally indistinguishable from that channel alone — a standard
# identifiability result in FDI, not an implementation shortcoming.
#
# Only redundancy breaks the tie. A channel that participates in a parity
# relation can be checked against its partner; one that does not, cannot. In
# this dataset the two thermocouples guard each other (a 6 K decoupling is
# caught on the first sample at 82 sigma) while torque and speed have no
# partner, so their bias is undetectable in principle.
#
# Saying so is the honest engineering output: if a plant wants bias protection
# on torque, it needs a second measurement or a physical relation involving it.
# That is a procurement decision, and this is the module that can name it.
REDUNDANT_CHANNELS: frozenset[str] = frozenset({"air_temp_k", "process_temp_k"})

ALWAYS_DETECTABLE = ("freeze", "dropout", "invalid value", "persistent gating")
NEEDS_REDUNDANCY = ("bias", "slow drift", "gain error")


@dataclass(frozen=True, slots=True)
class TrustReport:
    """The verdict on whether this tick's numbers deserve to be believed."""

    machine_id: str
    channels: dict[str, ChannelHealth]
    parity: list[ParityResidual]
    fault_kind: FaultKind
    uncertainty: Uncertainty
    explanation: str

    @property
    def trusted(self) -> bool:
        return self.fault_kind is FaultKind.NONE

    @property
    def calibrating(self) -> bool:
        """Still learning this asset, as opposed to having found a fault.

        These are different states and conflating them is a design error. A
        fault means stop and dispatch. Calibration means proceed, but with the
        wide bounds that reflect not yet knowing the instrument — which the
        interval machinery already knows how to turn into ABSTAIN where it
        matters and SAFE where the margin is comfortable regardless.

        Hard-blocking during calibration made every new asset silent for its
        first several hundred cycles and produced 650 alerts per 1,000 in
        replay, which is alarm fatigue manufactured by the very module meant to
        prevent it.
        """
        return any(
            c.status is NE107.FUNCTION_CHECK for c in self.channels.values()
        ) and not any(c.status is NE107.FAILURE for c in self.channels.values())

    @property
    def suspect_channels(self) -> list[str]:
        return [n for n, c in self.channels.items() if not c.trusted]

    @property
    def unguarded_channels(self) -> list[str]:
        """Channels with no redundancy partner, so no bias protection.

        Reported alongside every answer that depends on them. The system does
        not get to imply coverage it cannot deliver.
        """
        return [n for n in self.channels if n not in REDUNDANT_CHANNELS]

    def detectability_note(self) -> str:
        unguarded = self.unguarded_channels
        if not unguarded:
            return ""
        return (
            f"Bias and slow drift on {', '.join(unguarded)} are not detectable: "
            f"these channels have no redundancy partner, and a local level "
            f"estimator absorbs a step as a genuine process change. Freezes, "
            f"dropouts and invalid values on them are still caught."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "machine": self.machine_id,
            "fault_kind": self.fault_kind.value,
            "trusted": self.trusted,
            "suspect": self.suspect_channels,
            "channels": {n: c.as_dict() for n, c in self.channels.items()},
            "parity": [p.as_dict() for p in self.parity],
            "explanation": self.explanation,
        }


@dataclass(slots=True)
class _Channel:
    """Scalar local-level Kalman filter plus its fault statistics.

    The local level model — random walk state, additive measurement noise — is
    the standard estimator for "a slowly varying true value observed through a
    noisy instrument". It is chosen over a full multivariate filter deliberately:
    the cross-channel information here is algebraic (the parity relations), and
    keeping it algebraic means every alarm names one channel rather than
    implicating a state vector nobody can inspect.

    Three phases, because a channel must be understood before it can be judged:

      LEARNING     collect values, identify q and r by method of moments
      CALIBRATING  run the filter, measure the actual innovation energy
      RUNNING      detect faults

    The second phase exists because the fault tests are only as well calibrated
    as the noise identification. Normalising innovations by their *theoretical*
    variance leaves any model error in the statistic. Normalising by the energy
    the channel actually produces makes the chi-square exact whether or not the
    local level model was the right one — which matters, because for real
    signals it often is not.
    """

    name: str
    kind: ChannelKind = ChannelKind.LEVEL
    q: float = 0.0                # process variance   (identified, not declared)
    r: float = 1.0                # measurement variance (identified, not declared)
    x: float = 0.0                # posterior estimate
    p: float = 0.0                # posterior variance
    initialised: bool = False
    age: int = 0                  # cycles since last accepted measurement
    cusum_hi: float = 0.0
    cusum_lo: float = 0.0
    # Rolling window of squared normalised innovations, with its sum carried
    # incrementally. Rescanning the window each tick cost ~4 us per channel and
    # took the scorer under its throughput budget for no analytic benefit.
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=STUCK_WINDOW))
    energy_sum: float = 0.0
    last_value: float | None = None
    identical_run: int = 0
    gated_run: int = 0
    #: Rolling record of delivery: True for a usable sample, False for a gap.
    delivery: deque[bool] = field(default_factory=lambda: deque(maxlen=VAR_WINDOW))
    learning: list[float] = field(default_factory=list)
    calibrating: list[float] = field(default_factory=list)
    calibrated: bool = False
    energy: float = 1.0           # mean normalised innovation energy, measured
    repeat_limit: int = 5         # identical readings that imply a freeze

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    def identify(self) -> None:
        """Fit the local level model to the learning window by method of moments.

        For  x_t = x_{t-1} + w_t  and  z_t = x_t + v_t,  the first differences
        satisfy

            Var(dz)             = q + 2r
            Cov(dz_t, dz_{t-1}) = -r

        so both variances follow from two sample moments. This is the standard
        structural-time-series identification, and it means no channel needs a
        hand-set noise figure — which is the point, because the ones that were
        hand-set here were wrong by a factor of six.

        Validated on this dataset: it recovers torque's documented sd of
        10 N.m as 10.039 from the data alone, and returns q = 0 for channels
        that are white noise about a constant rather than random walks.
        """
        v = self.learning
        d = [v[i + 1] - v[i] for i in range(len(v) - 1)]
        if len(d) < 8:
            self.q, self.r = 0.0, 1.0
            return
        m = sum(d) / len(d)
        g0 = sum((x - m) ** 2 for x in d) / len(d)
        g1 = sum((d[i] - m) * (d[i + 1] - m) for i in range(len(d) - 1)) / (len(d) - 1)

        if g1 < 0:
            r, q = -g1, g0 + 2 * g1
        else:
            # Positive lag-1 autocovariance means the signal is not
            # "random walk + noise". Fall back to pure observation noise about a
            # slowly moving level: the conservative reading, since it assumes
            # more noise and therefore yields wider intervals.
            r, q = g0 / 2.0, 0.0

        # q = 0 is a legitimate boundary solution — white noise about a constant
        # level is exactly what this dataset's torque and speed are — but a
        # filter with literally zero process noise stops adapting forever. Floor
        # it so the estimate can still track a genuine slow shift.
        self.r = max(r, 1e-12)
        self.q = max(q, self.r * 1e-3)

        # How often does this channel legitimately repeat a value? That depends
        # on its quantisation and its noise, both of which are properties of the
        # instrument rather than things we can know in advance. Measure it, then
        # pick the run length whose probability under "healthy" is below the
        # stated false-positive budget.
        repeats = sum(1 for i in range(len(v) - 1) if v[i + 1] == v[i])
        p_rep = repeats / max(len(v) - 1, 1)
        if p_rep <= 0.0:
            self.repeat_limit = 5
        elif p_rep >= 1.0:
            self.repeat_limit = 10 ** 6      # a constant channel; never call it stuck
        else:
            self.repeat_limit = max(5, math.ceil(
                math.log(FALSE_STUCK_RATE) / math.log(p_rep)) + 1)

    def observe(self, z: float | None) -> ChannelHealth:
        # ── Missing measurement: predict only. Covariance grows, so staleness
        # widens the margin automatically instead of needing a separate flag.
        if z is None or not math.isfinite(z):
            self.delivery.append(False)
            self.age += 1
            if self.initialised:
                self.p += self.q
            return ChannelHealth(
                name=self.name,
                status=NE107.FAILURE if self.age > 3 else NE107.FUNCTION_CHECK,
                value=None,
                estimate=self.x,
                posterior_sd=math.sqrt(self.p) if self.initialised else float("inf"),
                innovation_z=0.0,
                cusum=max(self.cusum_hi, self.cusum_lo),
                stuck_score=float("nan"),
                age_cycles=self.age,
                reason=f"no reading for {self.age} cycle(s)",
            )

        self.delivery.append(True)

        # ── Phase 1: learn the channel before judging it.
        #
        # This is the honest behaviour for a new asset. The system does not know
        # this sensor's noise yet, so it says so and abstains, rather than
        # importing a default and pretending to be calibrated. FUNCTION_CHECK is
        # the NE 107 category for exactly this: temporarily invalid, known cause.
        if not self.initialised:
            self.learning.append(z)
            self.last_value = z
            if len(self.learning) < WARMUP:
                return self._provisional(z, f"learning ({len(self.learning)}/{WARMUP})")
            self.identify()
            self.x, self.p, self.initialised = z, self.r, True
            return self._provisional(z, "identified; measuring innovation energy")

        # ── Predict, then innovate. Every fault statistic below is a functional
        # of the innovation sequence.
        p_pred = self.p + self.q * (1 + self.age)
        innovation = z - self.x
        s = p_pred + self.r
        z_score = innovation / math.sqrt(s)

        self.identical_run = self.identical_run + 1 if z == self.last_value else 0
        self.last_value = z

        # ── Phase 2: measure the energy this channel actually produces.
        if not self.calibrated:
            self.calibrating.append(z_score * z_score)
            self._update(z_score, p_pred, s, innovation, gated=False)
            if len(self.calibrating) >= CALIBRATION:
                mean_energy = sum(self.calibrating) / len(self.calibrating)
                self.energy = max(mean_energy, 1e-12)
                self.calibrated = True
                return self._provisional(
                    z, f"calibrated: innovation energy {self.energy:.3g}")
            return self._provisional(
                z, f"calibrating ({len(self.calibrating)}/{CALIBRATION})")

        # ── Phase 3: fault detection.
        #
        # Normalise by measured energy, so the chi-square is exact even where
        # the local level model is an imperfect description of the signal.
        unit = z_score / math.sqrt(self.energy)
        sq = unit * unit
        if len(self.recent) == STUCK_WINDOW:
            self.energy_sum -= self.recent[0]
        self.recent.append(sq)
        self.energy_sum += sq
        stuck_score = (
            self.energy_sum if len(self.recent) == STUCK_WINDOW else float("nan")
        )

        # ── Drift: two-sided CUSUM on the standardised innovation. Optimal
        # under Lorden's criterion (Moustakides 1986), threshold from ARL0.
        self.cusum_hi = max(0.0, self.cusum_hi + unit - CUSUM_K)
        self.cusum_lo = max(0.0, self.cusum_lo - unit - CUSUM_K)
        cusum = max(self.cusum_hi, self.cusum_lo)

        counter = self.kind is ChannelKind.COUNTER
        status, reason = NE107.OK, ""

        if counter:
            # A counter's validity is kinematic, not statistical. Stillness is
            # idleness and a rising mean is just accumulation; neither is a
            # fault. Monotonicity is enforced as a parity relation instead.
            stuck_score = float("nan")
        elif self.identical_run + 1 >= self.repeat_limit:
            status = NE107.FAILURE
            reason = (
                f"frozen: {self.identical_run + 1} identical readings, against a "
                f"limit of {self.repeat_limit} derived from this channel's own "
                f"measured repeat rate"
            )
        elif self.gated_run >= GATED_RUN_LIMIT:
            status = NE107.FAILURE
            reason = (
                f"not tracking the process: {self.gated_run} consecutive readings "
                f"beyond the {INNOVATION_GATE:.1f} sigma gate. A single outlier is "
                f"a spike; a run of them means the channel has moved and the "
                f"estimate can no longer follow it"
            )
        elif self.gated_run == 0 and not math.isnan(stuck_score) and stuck_score < STUCK_CHI2:
            status = NE107.FAILURE
            reason = (
                f"frozen: innovation energy over {STUCK_WINDOW} cycles collapsed "
                f"to {stuck_score:.3g}, below the chi-square floor "
                f"{STUCK_CHI2:.3g} at a {FALSE_STUCK_RATE:g} false-positive rate"
            )
        elif (availability := self._availability()) is not None and (
            availability < AVAILABILITY_FLOOR
        ):
            failed = availability < AVAILABILITY_FAILED
            status = NE107.FAILURE if failed else NE107.MAINTENANCE_REQUIRED
            reason = (
                f"intermittent: {availability:.0%} of the last {len(self.delivery)} "
                f"samples arrived, against a {AVAILABILITY_FLOOR:.0%} floor. No "
                f"single gap is long enough to look like a dropout, which is how "
                f"a failing connector hides"
            )
        elif abs(unit) > INNOVATION_GATE:
            status = NE107.OUT_OF_SPEC
            reason = f"innovation {unit:+.1f} sigma beyond the {INNOVATION_GATE:.1f} gate"
        elif cusum > CUSUM_H:
            status = NE107.MAINTENANCE_REQUIRED
            reason = (
                f"CUSUM {cusum:.1f} over {CUSUM_H:.1f}; a sustained bias is "
                f"present but readings remain individually plausible"
            )

        # An out-of-spec sample is gated out of the state so one wild reading
        # cannot drag the estimate, but it still counts against staleness.
        gated = status is NE107.OUT_OF_SPEC
        self.gated_run = self.gated_run + 1 if gated else 0
        self._update(unit, p_pred, s, innovation, gated=gated)

        return ChannelHealth(
            name=self.name, status=status, value=z, estimate=self.x,
            posterior_sd=math.sqrt(self.p), innovation_z=unit, cusum=cusum,
            stuck_score=stuck_score, age_cycles=self.age, reason=reason,
        )

    def _update(self, unit: float, p_pred: float, s: float,
                innovation: float, gated: bool) -> None:
        if gated:
            self.age += 1
            self.p = p_pred
            return
        gain = p_pred / s
        self.x += gain * innovation
        self.p = (1 - gain) * p_pred
        self.age = 0

    def _availability(self) -> float | None:
        """Fraction of recent samples that actually arrived, or None if the
        window is too short to say anything."""
        if len(self.delivery) < VAR_WINDOW:
            return None
        return sum(self.delivery) / len(self.delivery)

    def _provisional(self, z: float, reason: str) -> ChannelHealth:
        return ChannelHealth(
            name=self.name, status=NE107.FUNCTION_CHECK, value=z,
            estimate=self.x if self.initialised else z,
            posterior_sd=math.sqrt(self.p) if self.initialised else float("inf"),
            innovation_z=0.0, cusum=0.0, stuck_score=float("nan"),
            age_cycles=0, reason=reason,
        )


# ── Analytical redundancy ─────────────────────────────────────────────────────
#
# A relation is only useful if it links *independently measured* quantities.
#
# An earlier draft of this design proposed power = torque x omega as a parity
# equation. That is wrong, and the error is worth recording: power is *derived*
# from torque and speed, so its residual is identically zero and it carries no
# information whatsoever. A derived quantity provides no redundancy. Only two
# genuine relations survive in this dataset, and both were verified against it.
THERMAL_COUPLING_K = 10.001   # measured mean of (process - air) over 10,000 rows
THERMAL_SIGMA_K = 1.001       # measured sd; matches the documented 1 K


@dataclass(slots=True)
class MachineObserver:
    """One machine's instrument layer."""

    machine_id: str
    channels: dict[str, _Channel] = field(default_factory=dict)
    last_wear: float | None = None
    tool_changes: int = 0

    def __post_init__(self) -> None:
        if not self.channels:
            self.channels = {
                name: _Channel(name=name, kind=kind) for name, kind in CHANNELS.items()
            }

    def observe(self, reading: dict[str, Any]) -> TrustReport:
        health = {
            name: ch.observe(_as_float(reading.get(name)))
            for name, ch in self.channels.items()
        }
        parity = self._parity(health, reading)
        kind, why = self._classify(health, parity)
        return TrustReport(
            machine_id=self.machine_id,
            channels=health,
            parity=parity,
            fault_kind=kind,
            uncertainty=self._uncertainty(health),
            explanation=why,
        )

    def _parity(
        self, health: dict[str, ChannelHealth], reading: dict[str, Any]
    ) -> list[ParityResidual]:
        out: list[ParityResidual] = []

        air = _as_float(reading.get("air_temp_k"))
        proc = _as_float(reading.get("process_temp_k"))
        if air is not None and proc is not None:
            out.append(ParityResidual(
                name="thermal_coupling",
                residual=(proc - air) - THERMAL_COUPLING_K,
                sigma=THERMAL_SIGMA_K,
                unit="K",
                channels=("air_temp_k", "process_temp_k"),
                description=(
                    "Process temperature tracks air temperature at a fixed offset. "
                    "Two independently instrumented points, so a violation means "
                    "one of the two thermocouples is lying."
                ),
            ))

        # Wear is a kinematic accumulator: it cannot run backwards within a tool
        # life. A decrease is either a tool change (legitimate, and it must be
        # declared) or a corrupt reading. Treated as a hard constraint scaled to
        # one sigma so it enters the same z-score arithmetic as everything else.
        # A wear counter that returns to zero is a tool change, not corruption.
        # This distinction matters more than it looks: without it the system
        # forecasts a crossing for a tool that no longer exists, and every
        # legitimate maintenance action reads as a data fault. A *partial* drop
        # has no such explanation and stays a violation.
        wear = _as_float(reading.get("tool_wear_min"))
        if wear is not None and self.last_wear is not None and wear < self.last_wear:
            drop = self.last_wear - wear
            if wear <= TOOL_CHANGE_RESET_MIN:
                self.tool_changes += 1
                self.last_wear = wear
                return out
            out.append(ParityResidual(
                name="wear_monotonicity",
                residual=-drop,
                sigma=max(math.sqrt(self.channels["tool_wear_min"].r), 1e-6),
                unit="min",
                channels=("tool_wear_min",),
                description=(
                    "Cumulative wear cannot decrease within a tool life. A drop is "
                    "an undeclared tool change or a corrupt counter; either way the "
                    "degradation forecast built on it is void."
                ),
            ))
        if wear is not None:
            self.last_wear = wear
        return out

    def _classify(
        self, health: dict[str, ChannelHealth], parity: list[ParityResidual]
    ) -> tuple[FaultKind, str]:
        """The distinction the whole module exists to make."""
        # An observer that has not finished calibrating does not get to say it
        # trusts anything. This was a real bug: _classify escalated only on
        # FAILURE, so during cold start every channel sat at FUNCTION_CHECK and
        # the report still came back trusted — an uncalibrated instrument layer
        # vouching for its own inputs.
        warming: list[ChannelHealth] = []
        dead: list[ChannelHealth] = []
        degrading: list[ChannelHealth] = []
        for c in health.values():
            if c.status is NE107.FUNCTION_CHECK:
                warming.append(c)
            elif c.status is NE107.FAILURE:
                dead.append(c)
            elif c.status is NE107.MAINTENANCE_REQUIRED:
                degrading.append(c)

        if warming:
            names = ", ".join(c.name for c in warming)
            return FaultKind.INSTRUMENT, (
                f"{names} not yet characterised ({warming[0].reason}). No margin "
                f"depending on these channels is trustworthy until calibration "
                f"completes."
            )

        if dead:
            names = ", ".join(c.name for c in dead)
            return FaultKind.INSTRUMENT, (
                f"{names} unusable ({dead[0].reason}). This is an instrument "
                f"fault, not a process fault — dispatch to the sensor."
            )

        broken = [p for p in parity if p.violated]
        if broken:
            p = broken[0]
            return FaultKind.MODEL, (
                f"{p.name} residual is {p.z:+.1f} sigma "
                f"({p.residual:+.2f} {p.unit}). {p.description} Margins that "
                f"depend on these channels cannot be trusted this cycle."
            )

        if degrading:
            names = ", ".join(c.name for c in degrading)
            return FaultKind.NONE, (
                f"Readings usable. {names} shows a sustained bias and should be "
                f"recalibrated, but individual samples remain plausible."
            )
        return FaultKind.NONE, ""

    def _uncertainty(self, health: dict[str, ChannelHealth]) -> Uncertainty:
        """Posterior sd becomes the half-width the interval machinery consumes.

        This is where the instrument layer meets the arithmetic layer. A stale
        or suspect channel has a large posterior sd, which widens the margin
        interval, which makes it straddle zero, which yields ABSTAIN — with no
        special-casing anywhere. Doubt propagates as covariance.
        """
        def half(name: str) -> float:
            c = health[name]
            chan = self.channels[name]

            if c.status is NE107.FAILURE:
                # The channel is dead. There is no defensible bound, so use one
                # wide enough to force abstention rather than invent precision.
                return 10.0 * math.sqrt(chan.r) if chan.calibrated else _WIDE[name]

            if c.status is NE107.FUNCTION_CHECK:
                # Still learning. We have readings, we just do not yet know the
                # instrument's accuracy, so bound by the dispersion actually
                # observed. An earlier version used a fixed wide default here,
                # which made 84% of a replay abstain — a system that abstains on
                # everything is as useless as one that never does.
                seen = chan.learning or [c.value or 0.0]
                if len(seen) >= 4:
                    mean = sum(seen) / len(seen)
                    sd = (sum((v - mean) ** 2 for v in seen) / (len(seen) - 1)) ** 0.5
                    return max(2.0 * sd, 1e-9)
                return _WIDE[name]

            # Healthy: the operating point IS the measured value, so the
            # relevant uncertainty is the instrument's, plus whatever staleness
            # the filter has accumulated since the last accepted reading.
            declared = DECLARED_ACCURACY.get(name) or math.sqrt(chan.r)
            if not c.age_cycles:
                return 2.0 * declared
            return 2.0 * math.hypot(declared, math.sqrt(chan.q * c.age_cycles))

        return Uncertainty(
            air_temp_k=half("air_temp_k"),
            process_temp_k=half("process_temp_k"),
            rotational_speed_rpm=half("rotational_speed_rpm"),
            torque_nm=half("torque_nm"),
            tool_wear_min=half("tool_wear_min"),
        )


# Declared instrument accuracy, in native units — the half-width of the sensor's
# own error, NOT the variability of the process it observes.
#
# This distinction cost a rewrite. The identification returns a torque
# observation sd of ~10 N.m, and feeding that into the margin made 73% of a
# replay abstain. But 10 N.m is how much the *process* genuinely varies cycle to
# cycle; the transducer measuring it is accurate to a fraction of that. From a
# single channel the two are not separable — the same identifiability wall that
# makes bias undetectable — so the split has to come from outside the signal.
#
# It comes from tag metadata, which is what a real historian carries and what
# the Uncertainty docstring has always named as its first source. These are
# typical class-A accuracies for the instrument types AI4I implies, and are
# recorded as assumptions in the README rather than presented as measurements.
#
# The observer's own identification is used for FAULT DETECTION, where total
# variance is the right quantity, and never for margin width, where it is not.
DECLARED_ACCURACY: dict[str, float] = {
    "air_temp_k": 0.5,               # type-K thermocouple, class 1
    "process_temp_k": 0.5,
    "rotational_speed_rpm": 1.0,     # incremental encoder
    "torque_nm": 0.25,               # rotary torque transducer, 0.5% of 50 N.m
    "tool_wear_min": 0.5,            # cycle-time accumulator
}

_WIDE = {
    "air_temp_k": 5.0,
    "process_temp_k": 5.0,
    "rotational_speed_rpm": 200.0,
    "torque_nm": 15.0,
    "tool_wear_min": 10.0,
}


@dataclass(slots=True)
class FleetObserver:
    """One observer per machine, created on first sight."""

    observers: dict[str, MachineObserver] = field(default_factory=dict)

    def observe(self, reading: dict[str, Any]) -> TrustReport:
        machine = str(reading.get("machine_id", "unknown"))
        obs = self.observers.get(machine)
        if obs is None:
            obs = self.observers[machine] = MachineObserver(machine_id=machine)
        return obs.observe(reading)


def _as_float(value: Any) -> float | None:
    """Coerce without raising. A bad value is missing data, not a crash."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
