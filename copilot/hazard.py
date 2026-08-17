"""A hazard that rises with wear, learned under a physical constraint.

WHAT WAS WRONG
--------------
The tool-wear failure mode was reported as a single rate: 3.6% inside the
documented window. That number is the average of a curve, and the curve is not
remotely flat:

    190-200 min   0.21%
    200-210       4.18%
    210-220       4.74%
    220-230       9.38%
    230-240      16.13%

Telling an operator at 235 minutes the same thing as one at 195 is the
difference between a warning and a shrug. The information was in the data and
the reporting threw it away.

WHY ISOTONIC REGRESSION AND NOT A NEURAL NETWORK
------------------------------------------------
This is a wearout mode, so the hazard is monotone non-decreasing in wear: a tool
does not become less likely to fail by being used more. That is a fact about the
physics, and the right model is one that CANNOT violate it.

Isotonic regression fits the best monotone step function to the observed rates —
best in least squares, by the pool-adjacent-violators algorithm, exactly and in
linear time. No learning rate, no epochs, no seed, no local minimum. Fit the
same data twice and get the same curve, which matters for something an engineer
is asked to act on.

A neural hazard model could fit this too, and would sometimes emit a curve that
dips — a tool getting safer as it wears — which is not a possibility to be
smoothed away but a statement that is false. Constraining the hypothesis space
to the physically possible is strictly better than fitting freely and hoping.

WHAT IT DOES NOT CLAIM
----------------------
The hazard is estimated from 46 observed failures. The top bands hold a handful
of cycles each, so their rates are wide, and every point carries a Wilson
interval saying so. Where a band is too thin to support a claim the estimate is
still reported — with an interval that makes the thinness obvious rather than a
point that hides it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from copilot.stats import MIN_REPORTABLE_N, wilson_interval

__all__ = ["HazardPoint", "HazardCurve", "fit_hazard", "isotonic"]


def isotonic(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    """Best non-decreasing fit, by pool-adjacent-violators.

    Exact, linear time, deterministic. Wherever the observed sequence dips, the
    offending block is replaced by its weighted mean and the merge cascades
    backwards until the whole sequence is monotone.
    """
    # Each block carries its weighted mean, its total weight, AND how many
    # original positions it spans. The span is what maps the fit back onto the
    # input; an earlier version replicated each block by its WEIGHT instead,
    # which for exposure weights in the hundreds produced a list thousands long
    # and then truncated it to the first block. Every fitted value came back as
    # that block's mean — a flat line, which passed the monotonicity check
    # because a constant is monotone. The output looked plausible and was
    # entirely wrong.
    blocks: list[tuple[float, float, int]] = []     # (mean, weight, span)
    for value, weight in zip(values, weights):
        if weight <= 0:
            blocks.append((float(value), 0.0, 1))
            continue
        current_sum, current_w, span = value * weight, weight, 1
        while blocks and blocks[-1][1] > 0 and blocks[-1][0] > current_sum / current_w:
            prev_mean, prev_w, prev_span = blocks.pop()
            current_sum += prev_mean * prev_w
            current_w += prev_w
            span += prev_span
        blocks.append((current_sum / current_w, current_w, span))

    out: list[float] = []
    for mean, _weight, span in blocks:
        out.extend([mean] * span)
    return out[: len(values)]


@dataclass(frozen=True, slots=True)
class HazardPoint:
    lower_edge: float
    upper_edge: float
    n: int
    failures: int
    observed: float
    fitted: float
    ci_low: float
    ci_high: float

    @property
    def reportable(self) -> bool:
        return self.n >= MIN_REPORTABLE_N

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low


@dataclass(frozen=True, slots=True)
class HazardCurve:
    """A monotone hazard over an accumulating quantity."""

    metric: str
    unit: str
    points: tuple[HazardPoint, ...] = field(default_factory=tuple)

    def at(self, value: float) -> HazardPoint | None:
        """The hazard band containing this value."""
        for point in self.points:
            if point.lower_edge <= value < point.upper_edge:
                return point
        return self.points[-1] if self.points and value >= self.points[-1].upper_edge else None

    def sentence(self, value: float) -> str:
        point = self.at(value)
        if point is None:
            return f"No hazard estimate covers {value:g} {self.unit}."
        if not point.reportable:
            return (
                f"At {value:g} {self.unit} the estimated hazard is "
                f"{point.fitted:.1%}, but this band holds only {point.n} cycles "
                f"({point.ci_low:.1%} to {point.ci_high:.1%}). Treat it as a "
                f"direction, not a figure."
            )
        return (
            f"At {value:g} {self.unit} the per-cycle hazard is {point.fitted:.1%} "
            f"({point.ci_low:.1%} to {point.ci_high:.1%} from {point.n:,} cycles). "
            f"The curve is constrained to be non-decreasing, because a tool does "
            f"not become safer by being used more."
        )

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "unit": self.unit,
            "bands": [
                {
                    "from": p.lower_edge, "to": p.upper_edge, "n": p.n,
                    "failures": p.failures,
                    "hazard": round(p.fitted, 5),
                    "ci": [round(p.ci_low, 5), round(p.ci_high, 5)],
                    "reportable": p.reportable,
                }
                for p in self.points
            ],
        }


def fit_hazard(
    values: Sequence[float],
    failed: Sequence[bool],
    *,
    metric: str = "tool_wear_min",
    unit: str = "min",
    band: float = 10.0,
    lower: float | None = None,
    upper: float | None = None,
    confidence: float = 0.95,
) -> HazardCurve:
    """Estimate a monotone per-cycle hazard over an accumulating quantity.

    Bands are fixed-width rather than equal-count on purpose: an engineer thinks
    in "around 220 minutes", not "the seventh quantile", and a curve whose
    x-axis moves with the data is one nobody can compare across machines.
    """
    pairs = [(float(v), bool(f)) for v, f in zip(values, failed)]
    if not pairs:
        return HazardCurve(metric=metric, unit=unit)

    lo = math.floor((lower if lower is not None else min(v for v, _ in pairs)) / band) * band
    hi = math.ceil((upper if upper is not None else max(v for v, _ in pairs)) / band) * band
    # A single reading, or a set that all lands on one band edge, gives lo == hi
    # and an empty loop below — no curve at all for data that plainly has one
    # band's worth. Guarantee at least one band.
    if hi <= lo:
        hi = lo + band

    edges: list[tuple[float, float]] = []
    counts: list[int] = []
    fails: list[int] = []
    edge = lo
    while edge < hi:
        inside = [f for v, f in pairs if edge <= v < edge + band]
        if inside:
            edges.append((edge, edge + band))
            counts.append(len(inside))
            fails.append(sum(inside))
        edge += band

    if not counts:
        return HazardCurve(metric=metric, unit=unit)

    observed = [f / n for f, n in zip(fails, counts)]
    # Weight by exposure: a band of 500 cycles constrains the fit far more than
    # a band of 2, and an unweighted fit would let the sparse tail dominate.
    fitted = isotonic(observed, [float(c) for c in counts])

    points = []
    for (low, high), n, f, obs, fit in zip(edges, counts, fails, observed, fitted):
        ci = wilson_interval(f, n, confidence)
        points.append(HazardPoint(
            lower_edge=low, upper_edge=high, n=n, failures=f,
            observed=obs, fitted=fit, ci_low=ci.lo, ci_high=ci.hi,
        ))
    return HazardCurve(metric=metric, unit=unit, points=tuple(points))
