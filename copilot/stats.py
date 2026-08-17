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

from functools import lru_cache

import math
from dataclasses import dataclass

from copilot.evidence import Interval

__all__ = [
    "MIN_REPORTABLE_N",
    "MAX_CI_RATIO",
    "wilson_interval",
    "spread_vs_chance",
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

# Permutation test size. 2,000 resamples resolves a p-value to about +/-0.01 at
# p = 0.05, which is finer than any decision made on it, and costs ~2 ms at our
# group counts.
PERMUTATIONS = 2000
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
    """Distribution summary. `n` is always carried - non-negotiable for trust."""

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
    """Linear interpolation between order statistics - matches DuckDB quantile_cont."""
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


@lru_cache(maxsize=512)
def spread_vs_chance(
    groups: tuple[tuple[int, int], ...],
    permutations: int = PERMUTATIONS,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Is the spread between these groups more than randomness would produce?

    A ranked list of assets is the most actionable artifact a maintenance
    copilot emits - it sends a technician somewhere. It is also the one most
    likely to be pure noise, because ranking N groups and naming the worst
    guarantees an extreme even when every group is identical. With 15 machines
    at a 3% base rate, the worst looks roughly 2.6x the best from chance alone.

    Wilson intervals do not catch this. They are per-group, and the reader's eye
    does the illegitimate comparison across them.

    The null is exchangeability: if the grouping label carried no information,
    reassigning labels at random would produce spreads like the observed one.
    No distributional assumption, no correction table, and it stays correct for
    unequal group sizes - which is what makes the naive comparison misleading in
    the first place.

    IMPLEMENTATION NOTE. The obvious version shuffles a pool of every row and
    slices it per group, which is O(permutations x rows) and took the test suite
    from 4.6s to 102s - a 22x regression on the axis this project is built to
    win. Sampling the per-group failure counts directly from the multivariate
    hypergeometric distribution is the same null, exactly, at O(permutations x
    groups): 2,000 x 15 draws instead of 2,000 x 10,000 shuffles. Memoised on
    the counts because a repeated question must not repeat the work.

    Takes (failures, n) per group. Returns (observed spread, p-value, median
    spread under the null), where spread is max rate minus min rate in points.
    """
    import numpy as np

    usable = [(f, n) for f, n in groups if n > 0]
    if len(usable) < 2:
        return 0.0, 1.0, 0.0

    rates = np.array([f / n for f, n in usable])
    observed = float(rates.max() - rates.min()) * 100.0

    sizes = np.array([n for _, n in usable], dtype=np.int64)
    total_f = int(sum(f for f, _ in usable))
    if total_f == 0 or total_f >= sizes.sum():
        return observed, 1.0, 0.0

    rng = np.random.default_rng(seed)
    # Under the null, allocating `total_f` failures across groups of these sizes
    # without replacement IS a multivariate hypergeometric draw.
    draws = rng.multivariate_hypergeometric(sizes, total_f, size=permutations)
    null_rates = draws / sizes
    null_spreads = (null_rates.max(axis=1) - null_rates.min(axis=1)) * 100.0

    at_least_as_extreme = int((null_spreads >= observed).sum())
    # Add-one smoothing: with 2,000 resamples the honest floor is p < 1/2001,
    # never p = 0.
    p_value = (at_least_as_extreme + 1) / (permutations + 1)
    return observed, float(p_value), float(np.median(null_spreads))


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation. Powers the collinearity warnings - r(rpm, torque)
    is -0.875 in this dataset, so every rpm analysis is confounded by torque."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0
