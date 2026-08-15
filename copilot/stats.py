"""Statistical primitives with explicit uncertainty.

Two rules are enforced here rather than left to the caller, because forgetting
them is how a copilot states a confident wrong number:

  * Every proportion carries a **Wilson** interval. The normal approximation is
    badly behaved at low p, and the base failure rate in this dataset is 3.39%.
  * A point estimate is **refused** when the interval is too wide relative to the
    estimate. Filters reach single-digit n quickly (there are only 1,003 H-variant
    rows), and "4.2%" from 12 observations is a lie of precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from copilot.evidence import Interval

__all__ = [
    "MIN_REPORTABLE_N",
    "MAX_CI_RATIO",
    "wilson_interval",
    "mean_interval",
    "cohens_d",
    "cohens_d_interval",
    "rate_ratio",
    "is_reportable",
    "Summary",
    "summarise",
    "pearson",
]

# Below this, a proportion is reported as a count with an interval, never a rate.
MIN_REPORTABLE_N = 30
# Refuse a point estimate when CI half-width exceeds this fraction of |estimate|.
MAX_CI_RATIO = 0.5

# Two-sided normal quantiles for the confidence levels we support.
_Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.98: 2.3263, 0.99: 2.5758}


def _z(confidence: float) -> float:
    if confidence in _Z:
        return _Z[confidence]
    # Acklam-style inverse normal, adequate for reporting intervals.
    p = 1.0 - (1.0 - confidence) / 2.0
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    return t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
        1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
    )


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Correct at the extremes, where the normal approximation produces intervals
    that extend below zero.
    """
    if n <= 0:
        return Interval(lo=0.0, hi=1.0)
    z = _z(confidence)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(lo=max(0.0, centre - spread), hi=min(1.0, centre + spread))


def mean_interval(mean: float, sd: float, n: int, confidence: float = 0.95) -> Interval:
    """Normal-approximation interval for a mean."""
    if n <= 1 or sd <= 0:
        return Interval(lo=mean, hi=mean)
    half = _z(confidence) * sd / math.sqrt(n)
    return Interval(lo=mean - half, hi=mean + half)


def cohens_d(mean_a: float, sd_a: float, n_a: int, mean_b: float, sd_b: float, n_b: int) -> float:
    """Standardised mean difference, pooled sd."""
    if n_a < 2 or n_b < 2:
        return 0.0
    pooled_var = ((n_a - 1) * sd_a**2 + (n_b - 1) * sd_b**2) / (n_a + n_b - 2)
    if pooled_var <= 0:
        return 0.0
    return (mean_a - mean_b) / math.sqrt(pooled_var)


def cohens_d_interval(d: float, n_a: int, n_b: int, confidence: float = 0.95) -> Interval:
    """Large-sample interval for Cohen's d."""
    if n_a < 2 or n_b < 2:
        return Interval(lo=d, hi=d)
    se = math.sqrt((n_a + n_b) / (n_a * n_b) + d * d / (2 * (n_a + n_b)))
    half = _z(confidence) * se
    return Interval(lo=d - half, hi=d + half)


def rate_ratio(succ_a: int, n_a: int, succ_b: int, n_b: int) -> float | None:
    """Ratio of two proportions. None when the denominator rate is zero."""
    if n_a <= 0 or n_b <= 0:
        return None
    p_b = succ_b / n_b
    if p_b == 0:
        return None
    return (succ_a / n_a) / p_b


def is_reportable(estimate: float, ci: Interval, n: int) -> tuple[bool, str]:
    """Should this be stated as a point estimate?

    Returns (ok, reason). The reason feeds the warning and the slot quality.
    """
    if n < MIN_REPORTABLE_N:
        return False, "low_sample"
    if estimate != 0:
        if (ci.width / 2.0) / abs(estimate) > MAX_CI_RATIO:
            return False, "wide_interval"
    elif ci.width > 0:
        return False, "wide_interval"
    return True, ""


@dataclass(frozen=True, slots=True)
class Summary:
    """Distribution summary. `n` is always carried — non-negotiable for trust."""

    n: int
    mean: float
    sd: float
    minimum: float
    p25: float
    median: float
    p75: float
    maximum: float

    def ci(self, confidence: float = 0.95) -> Interval:
        return mean_interval(self.mean, self.sd, self.n, confidence)


def summarise(values: list[float]) -> Summary:
    """Order statistics from a list. Used by the reference implementation and
    by ops that already hold rows in memory; SQL paths compute these in DuckDB."""
    if not values:
        return Summary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ordered = sorted(values)
    n = len(ordered)
    mean = sum(ordered) / n
    var = sum((v - mean) ** 2 for v in ordered) / (n - 1) if n > 1 else 0.0
    return Summary(
        n=n,
        mean=mean,
        sd=math.sqrt(var),
        minimum=ordered[0],
        p25=_quantile(ordered, 0.25),
        median=_quantile(ordered, 0.50),
        p75=_quantile(ordered, 0.75),
        maximum=ordered[-1],
    )


def _quantile(ordered: list[float], q: float) -> float:
    """Linear interpolation between order statistics — matches DuckDB quantile_cont."""
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation. Powers the collinearity warnings — r(rpm, torque)
    is -0.875 in this dataset, so every rpm analysis is confounded by torque."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0
