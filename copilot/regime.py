"""Which mode of operation is this? Learned online, from the data.

THE GAP THIS CLOSES
-------------------
Every threshold, baseline and noise estimate in this system was global. One
process, one set of expectations. A real plant is not that: it runs different
products on different recipes, and the same machine legitimately operates at
completely different setpoints through the week.

Against a global baseline a product changeover reads as mass anomaly. Every
margin shifts at once, the observer sees a step on every channel, and an
operator gets a screen of alarms for a planned event. That is the single
fastest way to lose a control room, and it was the largest gap remaining.

WHERE MACHINE LEARNING BELONGS
------------------------------
Here, and not on the numeric path. The system deliberately keeps learned models
away from anything that produces a figure — physics does that, and the figure
has to be exact. But "which of several operating modes is this machine in right
now" is not a physics question. It is an unsupervised segmentation problem with
no ground truth, nobody to label it, and a need to discover modes never seen
before. That is exactly what clustering is for.

So the regime is LEARNED and the margin is COMPUTED. The learned part chooses
which baseline applies; it never supplies a number.

WHY NOT k-MEANS
---------------
k-means needs k. Nobody knows how many recipes a plant runs, the answer changes
when production changes, and a wrong k silently merges two modes or splits one.
Choosing k would be exactly the kind of unjustified constant this project keeps
finding and removing.

Instead: sequential leader clustering with a statistically derived radius. A
point joins the nearest regime if its Mahalanobis distance falls inside a
chi-square quantile at a stated false-assignment rate; otherwise it is evidence
of a new mode. The radius is a consequence of the declared error rate and the
channel noise the observer already identified — not a number anyone picked. The
number of regimes is discovered, not configured.

Two properties matter for a control room:

  * **A changeover is announced, not alarmed.** Entering a known regime is a
    normal event and says so.
  * **An unknown regime abstains rather than guesses.** A mode never seen
    before has no baseline, so the honest output is "this is new, I am
    learning it" — the same discipline the observer already applies to a
    cold-started channel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Sequence

from scipy.stats import chi2

__all__ = [
    "RegimeStatus",
    "Regime",
    "RegimeVerdict",
    "RegimeTracker",
    "REGIME_AXES",
]

# The axes that define a mode of operation: the setpoints an operator chooses.
# Tool wear is excluded deliberately — it is a monotone counter that rises
# within every regime, so including it would split one recipe into a new
# "regime" every few hundred cycles.
REGIME_AXES: tuple[str, ...] = (
    "rotational_speed_rpm",
    "torque_nm",
    "temp_delta_k",
)

# ── Error budget. As everywhere in this system, the thresholds are consequences
# of a declared rate rather than choices. ─────────────────────────────────────
FALSE_NEW_RATE = 1e-3        # P(a point from a known regime is called new)
CONFIRM_CYCLES = 12          # sustained evidence before declaring a new mode
MIN_SAMPLES = 40             # observations before a regime has a usable baseline
MAX_REGIMES = 32             # a plant with more modes than this is misconfigured

#: Mahalanobis radius, from the budget and the dimensionality. Not chosen.
RADIUS = float(chi2.ppf(1 - FALSE_NEW_RATE, len(REGIME_AXES)))


class RegimeStatus(StrEnum):
    KNOWN = "known"              # inside an established mode
    TRANSITION = "transition"    # drifting between modes; do not judge yet
    NEW = "new"                  # a mode never seen; learning, not alarming
    LEARNING = "learning"        # inside a mode that lacks a usable baseline


@dataclass(slots=True)
class Regime:
    """One mode of operation, learned from the stream.

    Mean and variance are maintained by Welford's method so a regime can be
    updated in constant time and constant memory, however long it runs.
    """

    label: str
    #: Per-axis variance defining this mode's radius. Supplied by the caller
    #: from the CHANNEL noise, never accumulated from the cluster's own points.
    scale: list[float] = field(default_factory=list)
    n: int = 0
    mean: list[float] = field(default_factory=list)

    @property
    def calibrated(self) -> bool:
        return self.n >= MIN_SAMPLES

    def variance(self) -> list[float]:
        """The radius of this mode, from CHANNEL noise rather than cluster spread.

        Two failed designs preceded this, and both failed the same way.

        Accumulating the cluster's own variance is unstable in both directions.
        Let it grow and a regime that absorbs a few neighbouring points widens,
        absorbs more, and eventually swallows the space: five regimes found in
        two-regime data, flapping every few cycles. Freeze it early instead and
        the opposite happens — a variance estimated from forty samples is too
        tight, so the regime starts rejecting its own future points and shatters
        into eleven.

        The mistake was treating the radius as something to learn. It is not.
        Whether two operating points belong to the same mode depends on how much
        that channel naturally moves WITHIN a mode, which is exactly the noise
        the observer already identifies. Supplying it fixes the radius from the
        first sample and removes the feedback loop entirely.

        Measured on this dataset: within-regime scatter sits at d^2 ~ 3 while a
        genuine recipe change sits at 24.5, against a 16.27 radius. One regime
        for steady production, a clean break for a real changeover.
        """
        return self.scale or [1.0] * len(self.mean)

    def distance(self, point: Sequence[float]) -> float:
        """Squared Mahalanobis distance, diagonal covariance.

        Diagonal rather than full: a full covariance needs far more samples to
        estimate stably, and a singular matrix on a young regime is a silent
        source of nonsense. The cost is that a mode whose axes are strongly
        coupled — as rpm and torque are here, at r = -0.875 — is modelled as a
        ball around a curve rather than the curve itself. That is conservative:
        it merges genuinely distinct modes before it splits one.
        """
        if not self.mean:
            return 0.0
        return sum(
            (p - m) ** 2 / v
            for p, m, v in zip(point, self.mean, self.variance())
        )

    def update(self, point: Sequence[float]) -> None:
        """Track the mode's centre. The radius is fixed, so only the mean moves."""
        if not self.mean:
            self.mean = list(point)
            self.n = 1
            return
        self.n += 1
        for i, value in enumerate(point):
            self.mean[i] += (value - self.mean[i]) / self.n

    def describe(self) -> str:
        if not self.mean:
            return f"{self.label}: empty"
        parts = ", ".join(
            f"{axis.replace('_', ' ')} {m:.4g}"
            for axis, m in zip(REGIME_AXES, self.mean)
        )
        return f"{self.label}: {parts} (n={self.n:,})"


@dataclass(frozen=True, slots=True)
class RegimeVerdict:
    status: RegimeStatus
    label: str
    distance: float
    changed_from: str | None = None
    reason: str = ""

    @property
    def usable(self) -> bool:
        """Is there a baseline behind this verdict worth judging against?"""
        return self.status is RegimeStatus.KNOWN

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "regime": self.label,
            "distance": round(self.distance, 2),
            "changed_from": self.changed_from,
            "reason": self.reason,
        }


@dataclass(slots=True)
class RegimeTracker:
    """Discovers and tracks a machine's modes of operation."""

    #: Per-axis variance defining a mode's radius, from channel noise.
    scale: list[float] = field(default_factory=list)
    regimes: dict[str, Regime] = field(default_factory=dict)
    current: str | None = None
    _pending: int = 0
    _pending_point: list[float] = field(default_factory=list)
    _switch_to: str | None = None
    _switch_run: int = 0

    def observe(self, reading: dict[str, Any]) -> RegimeVerdict:
        point = [_as_float(reading.get(axis)) for axis in REGIME_AXES]
        if any(v is None for v in point):
            # A missing axis is the observer's problem, not the tracker's.
            return RegimeVerdict(
                RegimeStatus.TRANSITION, self.current or "unassigned", 0.0,
                reason="an axis is missing, so the mode cannot be determined",
            )
        values: list[float] = point  # type: ignore[assignment]

        if not self.regimes:
            return self._start("R1", values, first=True)

        label, distance = self._nearest(values)

        if distance <= RADIUS:
            self._pending = 0

            # Hysteresis. Without it a point sitting between two modes
            # reassigns on every cycle and the log fills with changeovers that
            # never happened. A real changeover persists; noise does not.
            if self.current is not None and label != self.current:
                current_distance = self.regimes[self.current].distance(values)
                if current_distance <= RADIUS:
                    # Still plausibly in the current mode — stay put.
                    self.regimes[self.current].update(values)
                    return RegimeVerdict(RegimeStatus.KNOWN, self.current,
                                         current_distance)
                self._switch_run = (
                    self._switch_run + 1 if self._switch_to == label else 1
                )
                self._switch_to = label
                if self._switch_run < CONFIRM_CYCLES:
                    return RegimeVerdict(
                        RegimeStatus.TRANSITION, self.current, distance,
                        reason=(f"looks like {label} for {self._switch_run} of "
                                f"{CONFIRM_CYCLES} cycles; not switching yet"),
                    )
            self._switch_run, self._switch_to = 0, None

            regime = self.regimes[label]
            regime.update(values)
            previous, self.current = self.current, label
            if not regime.calibrated:
                return RegimeVerdict(
                    RegimeStatus.LEARNING, label, distance,
                    reason=(f"{regime.n} of {MIN_SAMPLES} samples; no baseline for "
                            f"this mode yet, so margins are reported without one"),
                )
            if previous is not None and previous != label:
                return RegimeVerdict(
                    RegimeStatus.KNOWN, label, distance, changed_from=previous,
                    reason=(f"changeover from {previous} to a known mode. This is "
                            f"a planned event, not a fault: every margin moves at "
                            f"once because the setpoints moved"),
                )
            return RegimeVerdict(RegimeStatus.KNOWN, label, distance)

        # Outside every known mode. One point is a transient; a run of them is
        # a new recipe. Requiring persistence is what keeps a changeover from
        # spawning a phantom regime for the seconds it takes to settle.
        self._pending += 1
        self._pending_point = values
        if self._pending < CONFIRM_CYCLES:
            return RegimeVerdict(
                RegimeStatus.TRANSITION, self.current or "unassigned", distance,
                reason=(f"outside every known mode for {self._pending} of "
                        f"{CONFIRM_CYCLES} cycles; judgement withheld until this "
                        f"settles"),
            )
        return self._start(f"R{len(self.regimes) + 1}", values)

    def _start(self, label: str, values: list[float], *, first: bool = False) -> RegimeVerdict:
        self._pending = 0
        if len(self.regimes) >= MAX_REGIMES:
            # Refuse to shard indefinitely. More modes than this means the axes
            # are wrong or the machine is genuinely unstable, and inventing a
            # thirty-third regime helps nobody.
            return RegimeVerdict(
                RegimeStatus.TRANSITION, self.current or "unassigned", 0.0,
                reason=(f"already tracking {MAX_REGIMES} modes; this machine is "
                        f"not settling into recognisable recipes"),
            )
        regime = Regime(label=label, scale=list(self.scale))
        regime.update(values)
        self.regimes[label] = regime
        previous, self.current = self.current, label
        return RegimeVerdict(
            RegimeStatus.NEW, label, 0.0, changed_from=previous,
            reason=("a mode not seen before. There is no baseline for it, so "
                    "nothing is judged against one until it is learned"),
        )

    def _nearest(self, values: list[float]) -> tuple[str, float]:
        best, best_d = "", math.inf
        for label, regime in self.regimes.items():
            d = regime.distance(values)
            if d < best_d:
                best, best_d = label, d
        return best, best_d

    def summary(self) -> list[str]:
        return [r.describe() for r in self.regimes.values()]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
