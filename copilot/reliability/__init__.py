"""Reliability gates: instrument honesty, knowledge staleness, input uncertainty."""

from copilot.reliability.intervals import (
    DEFAULT_UNCERTAINTY,
    UNKNOWN_PROVENANCE,
    IntervalMargins,
    Uncertainty,
    Verdict,
    evaluate_interval,
)
from copilot.reliability.invariants import (
    DriftReport,
    DriftVerdict,
    check_invariants,
    diagnose_drift,
)
from copilot.reliability.kb_monitor import (
    CalibrationReport,
    Direction,
    audit_calibration,
    estimate_threshold,
)

__all__ = [
    "Uncertainty",
    "UNKNOWN_PROVENANCE",
    "DEFAULT_UNCERTAINTY",
    "IntervalMargins",
    "Verdict",
    "evaluate_interval",
    "check_invariants",
    "diagnose_drift",
    "DriftReport",
    "DriftVerdict",
    "audit_calibration",
    "estimate_threshold",
    "CalibrationReport",
    "Direction",
]
