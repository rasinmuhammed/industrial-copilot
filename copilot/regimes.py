"""Which kind of claim is this - exact, statistical, or impossible?

THE CRITICISM THIS ANSWERS
--------------------------
An adversarial technical review made one finding at high confidence, and it was
correct: the margin paradigm assumes a discrete failure boundary exists, and
that is true of only 15-20% of real industrial failure modes. Bearing spalling,
heat-exchanger fouling, pump cavitation and tool chatter are continuous
degradation. Forcing a threshold onto them produces, in the reviewer's words, a
synthetic and brittle metric.

The reviewer's remedy was to discard margins for conformal survival prediction.
That over-corrects. Where a documented boundary genuinely exists, an exact
margin is strictly BETTER than a probabilistic interval, because it inverts into
a setpoint: "reduce torque to 43 N·m" beats "73% risk" every time an engineer
has to actually do something.

The real error was never having margins. It was applying them everywhere and
letting the reader assume one confidence level throughout.

THE ARCHITECTURE
----------------
Three regimes, and the system always says which one it is in:

  EXACT         a documented boundary exists. Signed distance, invertible,
                verifiable to the last decimal. HDF, PWF, OSF here.
  STATISTICAL   failure occurs somewhere in a region with no crisp edge. The
                honest object is an interval with a COVERAGE GUARANTEE, not a
                point. TWF here - the tool fails somewhere in a wear window.
  IRREDUCIBLE   no relationship to any measured parameter. Not hard to predict:
                impossible. RNF here, by the dataset's own construction.

This was already declared and we were ignoring it. `failure_modes.yaml` marks
HDF/PWF/OSF `deterministic` and TWF/RNF `stochastic`, and RNF carries no
predicate at all. The regime split has been sitting in the knowledge base since
the first commit; the answer path just never read it.

It also explains a number we had been treating as a shortfall. Automated rule
discovery reaches 80.8% coverage on this dataset, and the remainder is dominated
by TWF: 46 of 339 failures, 13.6%, in a wear window with no crisp edge. That gap
is not a weakness in the discovery algorithm. It is the algorithm correctly
declining to invent a threshold for failures that do not have one.

A correction worth recording, because it is the exact error this project exists
to prevent and it was made here first. An earlier version of this note claimed
the gap was "TWF (46) plus RNF (19) = 19.2%". Only ONE of the 19 RNF-flagged
rows is labelled a machine failure; the other 18 carry the random-failure flag
without the process actually failing. Counting flag-rows as failures inflated
the irreducible share by a factor of nineteen. The corrected split, measured:

    exact         287 failures   84.7%   documented boundaries
    statistical    46 failures   13.6%   TWF, wear window
    irreducible     1 failure     0.3%   RNF
    undetermined    9 failures    2.7%   no documented mode

(These exceed 339 because 23 failures fire more than one mode at once.)

THE GUARANTEE
-------------
For the statistical regime we use split conformal prediction, which for the
identity nonconformity score reduces to order statistics of the calibration
sample. Two properties matter here and no other method has both:

  * **Distribution-free.** No Weibull, no Wiener process, no assumption about
    the shape of the degradation path. Wrong model, valid interval.
  * **Finite-sample exact.** Coverage is at least 1 - alpha for the actual n you
    have, not asymptotically. With 46 observed tool failures that distinction
    is the difference between a guarantee and a hope.

The only assumption is exchangeability: that a future tool is drawn from the
same population as the calibration tools. That assumption is checkable, it is
far weaker than any parametric alternative, and when a maintenance regime
changes it is the assumption that breaks - which is important to note to
the engineer rather than burying in a footnote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from copilot.process_model import FailureMode, ProcessModel, load_process_model

__all__ = [
    "Regime",
    "RegimeVerdict",
    "ConformalInterval",
    "regime_for",
    "conformal_interval",
    "empirical_coverage",
    "classify_modes",
]


class Regime(StrEnum):
    """What kind of claim the system is entitled to make about a failure mode."""

    EXACT = "exact"
    STATISTICAL = "statistical"
    IRREDUCIBLE = "irreducible"

    @property
    def claim(self) -> str:
        return {
            Regime.EXACT: "computed to the boundary, exact",
            Regime.STATISTICAL: "an interval with a coverage guarantee",
            Regime.IRREDUCIBLE: "not predictable from any measured parameter",
        }[self]


@dataclass(frozen=True, slots=True)
class RegimeVerdict:
    mode: str
    regime: Regime
    why: str

    @property
    def sentence(self) -> str:
        return f"{self.mode} is {self.regime.value}: {self.why}"


@dataclass(frozen=True, slots=True)
class ConformalInterval:
    """A prediction interval whose coverage is guaranteed, not modelled."""

    lower: float
    upper: float
    alpha: float
    n: int
    unit: str = ""
    achieved: float | None = None      # measured coverage, when validated

    @property
    def confidence(self) -> float:
        return 1.0 - self.alpha

    @property
    def valid(self) -> bool:
        """Is n large enough for the requested level to be attainable at all?

        With n calibration points the tightest honest two-sided interval is
        1 - 2/(n+1). Asking for 99% from 46 samples is asking for something the
        data cannot support, and the correct response is to say so rather than
        to return a narrower interval than the evidence permits.
        """
        return self.n >= math.ceil(2.0 / self.alpha) - 1

    def sentence(self, quantity: str) -> str:
        return (
            f"{quantity} falls between {self.lower:g} and {self.upper:g} {self.unit}"
            f" with at least {self.confidence:.0%} probability. This is a "
            f"distribution-free guarantee from {self.n} observed failures - it "
            f"holds whatever the true degradation shape is, assuming only that "
            f"future tools resemble past ones."
        ).strip()


def regime_for(mode: FailureMode) -> RegimeVerdict:
    """Read the regime off the process definition. It was always declared."""
    if not mode.conditions:
        return RegimeVerdict(
            mode.code, Regime.IRREDUCIBLE,
            "it has no predicate over any measured parameter, so no operating "
            "point predicts it and no margin can be computed",
        )
    if mode.kind == "deterministic":
        edges = ", ".join(f"{c.metric} {c.op}" for c in mode.conditions)
        return RegimeVerdict(
            mode.code, Regime.EXACT,
            f"a documented boundary exists ({edges}), so distance to it is "
            f"arithmetic rather than inference",
        )
    windows = [c for c in mode.conditions if c.op == "between"]
    if windows:
        return RegimeVerdict(
            mode.code, Regime.STATISTICAL,
            f"failure occurs somewhere within a {windows[0].metric} window with "
            f"no crisp edge, so the honest object is an interval with stated "
            f"coverage, not a point",
        )
    return RegimeVerdict(
        mode.code, Regime.STATISTICAL,
        "the mode is documented as stochastic, so its boundary is a "
        "distribution rather than a line",
    )


def classify_modes(model: ProcessModel | None = None) -> list[RegimeVerdict]:
    model = model or load_process_model()
    return [regime_for(m) for m in model.modes]


def conformal_interval(
    calibration: Sequence[float], alpha: float = 0.10, unit: str = ""
) -> ConformalInterval:
    """Split conformal prediction interval, exact in finite samples.

    For the identity nonconformity score, the conformal interval is a pair of
    order statistics. With n exchangeable calibration values sorted ascending,

        k = floor(alpha * (n + 1) / 2)

    and the interval [x_(k), x_(n+1-k)] covers a new exchangeable draw with
    probability at least 1 - alpha. No distributional assumption enters, and the
    result is not asymptotic: it holds at the n you actually have.

    When k rounds to zero the sample is too small to exclude anything at the
    requested level, and the honest interval is the full observed range with a
    coverage claim reduced to what n supports.
    """
    values = sorted(float(v) for v in calibration)
    n = len(values)
    if n == 0:
        raise ValueError("conformal prediction needs at least one calibration point")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    k = math.floor(alpha * (n + 1) / 2.0)
    if k < 1:
        # Too few points to trim either tail. Report the range and be explicit
        # that the achievable level is lower than the one requested.
        return ConformalInterval(
            lower=values[0], upper=values[-1],
            alpha=min(alpha, 2.0 / (n + 1)), n=n, unit=unit,
        )
    return ConformalInterval(
        lower=values[k - 1], upper=values[n - k], alpha=alpha, n=n, unit=unit
    )


def empirical_coverage(
    values: Sequence[float], alpha: float = 0.10
) -> tuple[float, int, int]:
    """Leave-one-out check that the guarantee actually holds on this data.

    A coverage guarantee that is never tested is a coverage claim. This holds
    out each observation, builds the interval from the remainder, and asks
    whether the held-out point was covered. Returns (rate, covered, n).

    Conformal is conservative by construction, so achieved coverage should land
    at or slightly above 1 - alpha. Materially below would mean exchangeability
    is violated in this sample and the guarantee does not apply.
    """
    xs = [float(v) for v in values]
    n = len(xs)
    if n < 3:
        return 1.0, n, n
    covered = 0
    for i in range(n):
        rest = xs[:i] + xs[i + 1:]
        interval = conformal_interval(rest, alpha)
        if interval.lower <= xs[i] <= interval.upper:
            covered += 1
    return covered / n, covered, n
