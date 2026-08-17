"""KB confidence update driven by CMMS work-order outcomes.

This is the learning loop that closes the diagnosis cycle:

    alert raised  →  technician investigates  →  outcome recorded  →  KB updated

Three outcome types and what they signal:

    confirmed    The alert was correct and the mode was right.
                 Strong evidence the rule is calibrated.
                 → nudge the KB entry's confidence up by ALPHA.

    false_alarm  The machine was fine; the alert was spurious.
                 Possible causes: threshold too tight, sensor noise, or
                 a rule form that doesn't generalise.
                 → record as false positive; if accumulated FP rate
                   exceeds ALERT_FP_RATE, emit a calibration warning.

    wrong_mode   The machine failed, but the attributed mode is different.
                 → confirm the attributed mode's rule instead.
                   Flag the alerted mode for review (its threshold fired
                   when it shouldn't have, OR the technician mis-diagnosed).

Updates are logged to ``cmms_kb_log`` for audit. They do NOT mutate the
knowledge base YAML on disk - they accumulate a signed weight that the
query layer uses to adjust confidence bounds on reported numbers. The
documented threshold is never changed without a human review step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from copilot.cmms import AlertOutcome, CMMSStore, WorkOrder

# Learning rate - each confirmation shifts confidence by this much.
ALPHA = 0.002

# If the rolling false-alarm rate for a mode exceeds this, emit a warning.
ALERT_FP_RATE = 0.10

# How many recent outcomes to use for the rolling rate.
ROLLING_WINDOW = 50


# --------------------------------------------------------------------------
# Weight store (in-process, persisted via cmms_kb_log)
# --------------------------------------------------------------------------


@dataclass
class ModeWeight:
    """Accumulated confidence adjustment for one (mode, variant) pair."""
    mode: str
    variant: str
    confirmations: int = 0
    false_alarms: int = 0
    wrong_modes: int = 0
    weight_delta: float = 0.0   # signed; positive = more confident

    @property
    def total_closed(self) -> int:
        return self.confirmations + self.false_alarms + self.wrong_modes

    @property
    def precision(self) -> float | None:
        denom = self.confirmations + self.false_alarms
        return self.confirmations / denom if denom else None

    @property
    def fp_rate(self) -> float:
        denom = self.confirmations + self.false_alarms
        return self.false_alarms / denom if denom else 0.0

    @property
    def calibration_flag(self) -> str | None:
        if self.total_closed < 5:
            return None
        if self.fp_rate > ALERT_FP_RATE:
            return f"fp_rate={self.fp_rate:.1%} > threshold={ALERT_FP_RATE:.0%}"
        if self.false_alarms == 0 and self.wrong_modes == 0 and self.confirmations >= 20:
            return None   # clean slate
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "variant": self.variant,
            "confirmations": self.confirmations,
            "false_alarms": self.false_alarms,
            "wrong_modes": self.wrong_modes,
            "weight_delta": round(self.weight_delta, 4),
            "precision": round(self.precision, 3) if self.precision is not None else None,
            "fp_rate": round(self.fp_rate, 3),
            "calibration_flag": self.calibration_flag,
        }


class FeedbackLearner:
    """Updates KB mode weights from closed work orders.

    Intended usage:

        learner = FeedbackLearner(store)
        learner.apply(work_order)   # called by POST /cmms/work_orders/{id}/close
        report = learner.report()   # called by GET /cmms/feedback
    """

    def __init__(self, store: CMMSStore) -> None:
        self._store = store
        self._weights: dict[tuple[str, str], ModeWeight] = {}
        self._load_from_log()

    # ---- public API --------------------------------------------------------

    def apply(self, wo: WorkOrder, variant: str = "L") -> dict[str, Any]:
        """Process a newly-closed work order and return the update summary."""
        if wo.outcome is None or wo.is_open():
            return {"status": "skipped", "reason": "work order is still open"}

        key   = (wo.alert_mode, variant)
        mw    = self._weights.setdefault(key, ModeWeight(wo.alert_mode, variant))
        delta = 0.0
        reason = ""

        if wo.outcome == AlertOutcome.CONFIRMED:
            mw.confirmations += 1
            delta   = +ALPHA
            reason  = "confirmed: alert correct and mode matched"

        elif wo.outcome == AlertOutcome.FALSE_ALARM:
            mw.false_alarms += 1
            delta   = -ALPHA * 0.5   # penalise but gently - one FA is not a rule change
            reason  = "false_alarm: no failure found; possible threshold too tight"

        elif wo.outcome == AlertOutcome.WRONG_MODE:
            mw.wrong_modes += 1
            delta   = -ALPHA * 0.25
            reason  = f"wrong_mode: technician attributed {wo.confirmed_mode!r}"
            # Also credit the correctly-attributed mode
            if wo.confirmed_mode:
                corr_key = (wo.confirmed_mode, variant)
                corr     = self._weights.setdefault(corr_key, ModeWeight(wo.confirmed_mode, variant))
                corr.confirmations += 1
                corr.weight_delta  += ALPHA
                self._store.log_kb_update(
                    wo.id, wo.confirmed_mode, variant, +ALPHA,
                    f"credited from wrong_mode WO {wo.id}",
                )

        elif wo.outcome == AlertOutcome.INCONCLUSIVE:
            reason = "inconclusive: no weight change"

        mw.weight_delta += delta
        if delta != 0.0:
            self._store.log_kb_update(wo.id, wo.alert_mode, variant, delta, reason)

        return {
            "status": "applied",
            "mode": wo.alert_mode,
            "variant": variant,
            "delta": delta,
            "reason": reason,
            "fp_rate": mw.fp_rate,
            "calibration_flag": mw.calibration_flag,
        }

    def report(self) -> list[dict[str, Any]]:
        """All mode weights, sorted by FP rate descending."""
        return sorted(
            [mw.as_dict() for mw in self._weights.values()],
            key=lambda d: d["fp_rate"],
            reverse=True,
        )

    def weight_for(self, mode: str, variant: str = "L") -> float:
        """Signed weight delta for a (mode, variant) pair. Zero if unseen."""
        mw = self._weights.get((mode, variant))
        return mw.weight_delta if mw else 0.0

    # ---- internal ----------------------------------------------------------

    def _load_from_log(self) -> None:
        """Replay the KB log to reconstruct in-memory weights on startup."""
        for row in self._store.kb_log(limit=10_000):
            key = (row["mode"], row["variant"])
            mw  = self._weights.setdefault(key, ModeWeight(row["mode"], row["variant"]))
            mw.weight_delta += row["delta_weight"]
            # Reconstruct counts from sign of delta
            if row["delta_weight"] > 0:
                mw.confirmations += 1
            elif row["delta_weight"] < 0:
                mw.false_alarms  += 1
