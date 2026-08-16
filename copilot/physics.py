"""The process model, in one place.

Every operator that reasons about hypothetical operating points — counterfactual,
envelope, forecast — needs the same derivations and the same rule predicates.
Duplicating them across ops is how a system ends up disagreeing with itself, so
they live here and the SQL in ingest.py mirrors this exactly.

The one subtlety that matters: **the base variables are coupled through the
derived ones.** Changing torque changes power *and* overstrain, so it can move a
cycle across the PWF boundary and the OSF boundary at once. Any counterfactual
that edits a single stored column rather than recomputing from base variables
will silently produce the wrong answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Final, Literal

from copilot.process_model import load_process_model

__all__ = [
    "RAD_PER_RPM",
    "OSF_THRESHOLD",
    "WEAR_RATE_PER_CYCLE",
    "HDF_TEMP_LIMIT",
    "HDF_SPEED_LIMIT",
    "PWF_LOW",
    "PWF_HIGH",
    "TWF_WINDOW",
    "BASE_VARIABLES",
    "OperatingPoint",
    "Margins",
    "evaluate",
]

RAD_PER_RPM: Final = 2 * math.pi / 60  # a unit conversion, not a process fact

# ── Every constant below is READ from the knowledge base, not declared here.
#
# These used to be literals in this file *and* in failure_modes.yaml: two
# sources of truth for the same number, with nothing to catch them diverging.
# For a system whose entire thesis is "one verified source of truth", having
# that duplication in the most load-bearing module was the sharpest possible
# instance of the failure it exists to prevent.
#
# It also reduced the 1,000-factory story to a claim. With compiled constants,
# onboarding factory #2 means editing Python and redeploying. Reading them makes
# a new process a new FILE, which is what turns the scale argument into
# something demonstrable — see tests/test_process_model.py, which runs the whole
# stack against a second, different process definition with no code change.
#
# The names are kept as an AI4I-shaped *view* over the general structure, so
# existing call sites and their tests continue to mean exactly what they meant.
_MODEL: Final = load_process_model()

HDF_TEMP_LIMIT: Final = _MODEL.limit("HDF", "temp_delta_k")
HDF_SPEED_LIMIT: Final = _MODEL.limit("HDF", "rotational_speed_rpm")
PWF_LOW: Final = _MODEL.limit("PWF", "power_w", "<")
PWF_HIGH: Final = _MODEL.limit("PWF", "power_w", ">")
TWF_WINDOW: Final = _MODEL.limit("TWF", "tool_wear_min")

OSF_THRESHOLD: Final[dict[str, float]] = _MODEL.limits_by_type("OSF", "overstrain_min_nm")

WEAR_RATE_PER_CYCLE: Final[dict[str, float]] = _MODEL.wear_rate_per_cycle

BASE_VARIABLES: Final = _MODEL.base_variables


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """A cycle's base state. Derived quantities are computed, never stored."""

    air_temp_k: float
    process_temp_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float
    product_type: Literal["L", "M", "H"] = "L"

    # --- derived -----------------------------------------------------------

    @property
    def temp_delta_k(self) -> float:
        return self.process_temp_k - self.air_temp_k

    @property
    def omega_rad_s(self) -> float:
        return self.rotational_speed_rpm * RAD_PER_RPM

    @property
    def power_w(self) -> float:
        return self.torque_nm * self.omega_rad_s

    @property
    def overstrain_min_nm(self) -> float:
        return self.tool_wear_min * self.torque_nm

    @property
    def osf_threshold(self) -> float:
        return OSF_THRESHOLD[self.product_type]

    @property
    def wear_rate(self) -> float:
        return WEAR_RATE_PER_CYCLE[self.product_type]

    def perturb(self, **deltas: float) -> OperatingPoint:
        """Return a new point with base variables shifted.

        Rejects derived quantities, because moving them is under-specified.
        """
        unknown = set(deltas) - BASE_VARIABLES
        if unknown:
            raise ValueError(
                f"cannot perturb derived quantities {sorted(unknown)}; "
                f"vary a base variable instead: {sorted(BASE_VARIABLES)}"
            )
        updates = {name: getattr(self, name) + delta for name, delta in deltas.items()}
        # Physical floors. Torque and wear cannot go negative; speed cannot stall
        # to zero without leaving the modelled regime entirely.
        if "torque_nm" in updates:
            updates["torque_nm"] = max(0.0, updates["torque_nm"])
        if "tool_wear_min" in updates:
            updates["tool_wear_min"] = max(0.0, updates["tool_wear_min"])
        if "rotational_speed_rpm" in updates:
            updates["rotational_speed_rpm"] = max(1.0, updates["rotational_speed_rpm"])
        return replace(self, **updates)


@dataclass(frozen=True, slots=True)
class Margins:
    """Signed distance to every boundary. Negative means violated."""

    temp_delta_k: float
    speed_rpm: float
    power_low_w: float
    power_high_w: float
    overstrain_min_nm: float
    wear_to_window_min: float

    # Rule-level distances, normalised by their own thresholds. HDF is
    # conjunctive so its binding constraint is the LARGER margin; PWF is
    # disjunctive so its binding constraint is the smaller.
    @property
    def hdf_distance(self) -> float:
        return max(self.temp_delta_k / HDF_TEMP_LIMIT, self.speed_rpm / HDF_SPEED_LIMIT)

    @property
    def pwf_distance(self) -> float:
        return min(self.power_low_w / PWF_LOW, self.power_high_w / PWF_HIGH)

    def osf_distance(self, threshold: float) -> float:
        return self.overstrain_min_nm / threshold

    @property
    def hdf_fired(self) -> bool:
        return self.temp_delta_k < 0 and self.speed_rpm < 0

    @property
    def pwf_fired(self) -> bool:
        return self.power_low_w < 0 or self.power_high_w < 0

    @property
    def osf_fired(self) -> bool:
        return self.overstrain_min_nm < 0

    def fired_modes(self) -> list[str]:
        out = []
        if self.hdf_fired:
            out.append("HDF")
        if self.pwf_fired:
            out.append("PWF")
        if self.osf_fired:
            out.append("OSF")
        return out


def evaluate(point: OperatingPoint) -> Margins:
    """Compute every margin for an operating point. Pure arithmetic, ~microseconds."""
    return Margins(
        temp_delta_k=point.temp_delta_k - HDF_TEMP_LIMIT,
        speed_rpm=point.rotational_speed_rpm - HDF_SPEED_LIMIT,
        power_low_w=point.power_w - PWF_LOW,
        power_high_w=PWF_HIGH - point.power_w,
        overstrain_min_nm=point.osf_threshold - point.overstrain_min_nm,
        wear_to_window_min=TWF_WINDOW[0] - point.tool_wear_min,
    )
