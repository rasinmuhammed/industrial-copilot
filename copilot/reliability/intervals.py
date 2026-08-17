"""Interval-valued margins, and the three-state decision.

Outliers are the single most frequent documented sensor error. A binary
alerting system therefore converts the most common data fault directly into
false alarms, and alarm fatigue is the documented reason operators stop
trusting these systems.

So alerting is not binary. It is:

    ALERT    the entire margin interval is negative
    SAFE     the entire margin interval is positive
    ABSTAIN  the interval straddles zero - say so, and flag the instrument

Cost: 2x the flops. Gain: a bad reading can never produce a false alert. It
produces silence and a maintenance ticket.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from copilot.evidence import Interval
from copilot.physics import (
    HDF_SPEED_LIMIT,
    HDF_TEMP_LIMIT,
    OSF_THRESHOLD,
    PWF_HIGH,
    PWF_LOW,
    RAD_PER_RPM,
)

__all__ = ["Verdict", "Uncertainty", "IntervalMargins", "evaluate_interval", "DEFAULT_UNCERTAINTY"]


class Verdict(StrEnum):
    ALERT = "alert"
    SAFE = "safe"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class Uncertainty:
    """Half-width of each sensor's plausible range, in native units.

    Sources, in priority order:
      1. declared instrument accuracy from tag metadata
      2. observed short-window dispersion (Hampel/MAD)
      3. imputation bounds when a value is missing
      4. a wide default when provenance is unknown - which correctly yields
         ABSTAIN rather than false confidence
    """

    air_temp_k: float = 0.0
    process_temp_k: float = 0.0
    rotational_speed_rpm: float = 0.0
    torque_nm: float = 0.0
    tool_wear_min: float = 0.0

    @property
    def any(self) -> bool:
        return any(
            getattr(self, f) > 0
            for f in (
                "air_temp_k",
                "process_temp_k",
                "rotational_speed_rpm",
                "torque_nm",
                "tool_wear_min",
            )
        )


DEFAULT_UNCERTAINTY = Uncertainty()

# Used when a sensor is quarantined and no better bound exists. Deliberately
# wide: the correct behaviour for an untrusted input is to abstain.
UNKNOWN_PROVENANCE = Uncertainty(
    air_temp_k=5.0,
    process_temp_k=5.0,
    rotational_speed_rpm=200.0,
    torque_nm=15.0,
    tool_wear_min=10.0,
)


@dataclass(frozen=True, slots=True)
class IntervalMargins:
    """Every margin as a bound, plus the resulting three-state verdict."""

    temp_delta: Interval
    speed: Interval
    power_low: Interval
    power_high: Interval
    overstrain: Interval

    def rule_verdicts(self) -> dict[str, Verdict]:
        return {
            "HDF": _conjunctive(self.temp_delta, self.speed),
            "PWF": _disjunctive(self.power_low, self.power_high),
            "OSF": _single(self.overstrain),
        }

    def verdict(self) -> Verdict:
        """Worst-case across rules. ALERT beats ABSTAIN beats SAFE."""
        verdicts = set(self.rule_verdicts().values())
        if Verdict.ALERT in verdicts:
            return Verdict.ALERT
        if Verdict.ABSTAIN in verdicts:
            return Verdict.ABSTAIN
        return Verdict.SAFE

    def abstaining_rules(self) -> list[str]:
        return [k for k, v in self.rule_verdicts().items() if v is Verdict.ABSTAIN]

    def firing_rules(self) -> list[str]:
        return [k for k, v in self.rule_verdicts().items() if v is Verdict.ALERT]


def _single(margin: Interval) -> Verdict:
    return {
        "negative": Verdict.ALERT,
        "positive": Verdict.SAFE,
        "straddles": Verdict.ABSTAIN,
    }[margin.verdict()]


def _conjunctive(a: Interval, b: Interval) -> Verdict:
    """HDF: fires only if BOTH conditions are violated.

    Certainly firing requires both intervals wholly negative. Certainly safe
    requires at least one wholly positive - because that one condition alone
    prevents the rule regardless of the other.
    """
    va, vb = a.verdict(), b.verdict()
    if va == "negative" and vb == "negative":
        return Verdict.ALERT
    if va == "positive" or vb == "positive":
        return Verdict.SAFE
    return Verdict.ABSTAIN


def _disjunctive(a: Interval, b: Interval) -> Verdict:
    """PWF: fires if EITHER side is violated.

    Certainly firing needs one side wholly negative; certainly safe needs both
    wholly positive.
    """
    va, vb = a.verdict(), b.verdict()
    if va == "negative" or vb == "negative":
        return Verdict.ALERT
    if va == "positive" and vb == "positive":
        return Verdict.SAFE
    return Verdict.ABSTAIN


def evaluate_interval(
    *,
    air_temp_k: float,
    process_temp_k: float,
    rotational_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
    product_type: str = "L",
    uncertainty: Uncertainty = DEFAULT_UNCERTAINTY,
) -> IntervalMargins:
    """Propagate input uncertainty through to margin bounds.

    Each margin is monotone in its inputs, so the extremes of the output
    interval come from the extremes of the inputs - no sampling required, which
    is why this stays a fixed 2x cost rather than a Monte Carlo.
    """
    u = uncertainty
    threshold = OSF_THRESHOLD[product_type]

    # temp_delta = process - air : rises with process, falls with air.
    dt_lo = (process_temp_k - u.process_temp_k) - (air_temp_k + u.air_temp_k)
    dt_hi = (process_temp_k + u.process_temp_k) - (air_temp_k - u.air_temp_k)

    rpm_lo = max(0.0, rotational_speed_rpm - u.rotational_speed_rpm)
    rpm_hi = rotational_speed_rpm + u.rotational_speed_rpm
    tq_lo = max(0.0, torque_nm - u.torque_nm)
    tq_hi = torque_nm + u.torque_nm
    wear_lo = max(0.0, tool_wear_min - u.tool_wear_min)
    wear_hi = tool_wear_min + u.tool_wear_min

    # power = torque x omega : monotone increasing in both.
    power_lo = tq_lo * rpm_lo * RAD_PER_RPM
    power_hi = tq_hi * rpm_hi * RAD_PER_RPM

    # strain = wear x torque : monotone increasing in both.
    strain_lo = wear_lo * tq_lo
    strain_hi = wear_hi * tq_hi

    return IntervalMargins(
        temp_delta=Interval(lo=dt_lo - HDF_TEMP_LIMIT, hi=dt_hi - HDF_TEMP_LIMIT),
        speed=Interval(lo=rpm_lo - HDF_SPEED_LIMIT, hi=rpm_hi - HDF_SPEED_LIMIT),
        power_low=Interval(lo=power_lo - PWF_LOW, hi=power_hi - PWF_LOW),
        power_high=Interval(lo=PWF_HIGH - power_hi, hi=PWF_HIGH - power_lo),
        overstrain=Interval(lo=threshold - strain_hi, hi=threshold - strain_lo),
    )
