"""Did the alert turn out to be right?

THE GAP THIS CLOSES
-------------------
Nothing in this system ever found out. It raised an alert, the alert scrolled
off the screen, and no part of the code ever learned whether a failure followed.
Every other quality here is measured - coverage, soundness, false-alarm rate -
except the one an operator actually judges the product on: **when it said
something was about to break, was it?**

A predictive system without an outcome loop cannot improve from being wrong, and
worse, cannot notice that it has started being wrong. Threshold drift, a changed
product mix, a recalibrated sensor: each degrades precision silently.

WHY THIS AND THE CMMS ARE THE SAME PIECE OF WORK
------------------------------------------------
An outcome loop needs ground truth, and in a plant the ground truth is the
maintenance record. "Was the alert right" is answered by whether somebody
subsequently opened a work order against that asset, and what they found when
they got there.

So the ledger takes outcomes from any source:

  * **labels** - a replay over historical data, where the failure flag is known.
    This is how the loop is validated here.
  * **work orders** - a CMMS feed in production. The same interface; a
    different `Outcome.source`.
  * **operator confirmation** - someone pressing "yes, that was real".

WHAT IT MEASURES, AND WHY THESE THREE
-------------------------------------
  precision   of the alerts raised, how many preceded a real failure. This is
              the number that decides whether anyone keeps looking at the
              screen; alarm fatigue is a precision problem.
  recall      of the failures that occurred, how many were alerted first. The
              number that decides whether the product is worth having.
  lead time   how long the warning arrived before the event. An alert two
              cycles ahead is technically a hit and operationally useless, so a
              precision figure alone flatters a system that fires late.

An alert is scored against a HORIZON. Fire too early and the event falls outside
the window and counts against you; fire after it and it is not a prediction. The
window is the honest way to say what "right" means, and it is declared rather
than tuned.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable

__all__ = [
    "OutcomeSource",
    "Outcome",
    "OpenAlert",
    "ScoredAlert",
    "Scorecard",
    "AlertLedger",
    "WorkOrder",
    "work_orders_from_csv",
]

#: An alert claims a failure is coming. This is how long it has to be right for.
#: Declared, not tuned: it states what the product means by a useful warning.
DEFAULT_HORIZON = 60          # cycles
#: A warning that arrives with less notice than this is a hit on paper and of no
#: operational use - there is no time to act on it.
ACTIONABLE_LEAD = 5           # cycles


class OutcomeSource(StrEnum):
    LABEL = "label"                    # historical replay; ground truth known
    WORK_ORDER = "work_order"          # a CMMS record; production ground truth
    OPERATOR = "operator"              # somebody confirmed it by hand


@dataclass(frozen=True, slots=True)
class Outcome:
    """Something that actually happened to a machine."""

    machine_id: str
    at: float                          # cycle index or epoch, same clock as alerts
    failed: bool
    source: OutcomeSource = OutcomeSource.LABEL
    mode: str | None = None            # which failure mode, when known
    note: str = ""


@dataclass(frozen=True, slots=True)
class WorkOrder:
    """A maintenance record, the production form of ground truth.

    Deliberately thin. A real CMMS row carries dozens of fields; the loop needs
    only who, when, and whether the visit found a genuine fault. Modelling more
    would be inventing a schema for a system we cannot see.
    """

    machine_id: str
    raised_at: float
    kind: str                          # corrective | preventive | inspection
    found_fault: bool
    mode: str | None = None
    reference: str = ""

    def as_outcome(self) -> Outcome:
        return Outcome(
            machine_id=self.machine_id, at=self.raised_at,
            failed=self.found_fault and self.kind == "corrective",
            source=OutcomeSource.WORK_ORDER, mode=self.mode,
            note=f"{self.kind} work order {self.reference}".strip(),
        )


@dataclass(frozen=True, slots=True)
class OpenAlert:
    machine_id: str
    at: float
    mode: str
    predicted_lead: float | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ScoredAlert:
    alert: OpenAlert
    correct: bool
    lead: float | None = None          # actual notice given, when correct
    matched: Outcome | None = None

    @property
    def actionable(self) -> bool:
        return self.correct and (self.lead or 0) >= ACTIONABLE_LEAD


@dataclass(frozen=True, slots=True)
class Scorecard:
    alerts: int
    correct: int
    actionable: int
    failures: int
    caught: int
    median_lead: float | None
    horizon: float

    @property
    def precision(self) -> float:
        return self.correct / self.alerts if self.alerts else 0.0

    @property
    def actionable_precision(self) -> float:
        """Precision counting only warnings that arrived in time to act on.

        Reported separately because a system that fires one cycle before the
        event scores perfectly on ordinary precision and helps nobody.
        """
        return self.actionable / self.alerts if self.alerts else 0.0

    @property
    def recall(self) -> float:
        return self.caught / self.failures if self.failures else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "alerts": self.alerts,
            "precision": round(self.precision, 4),
            "actionable_precision": round(self.actionable_precision, 4),
            "failures": self.failures,
            "recall": round(self.recall, 4),
            "median_lead_cycles": self.median_lead,
            "horizon_cycles": self.horizon,
        }

    def summary(self) -> str:
        if not self.alerts:
            return "No alerts were raised, so there is nothing to score."
        lead = (f"{self.median_lead:.0f} cycles" if self.median_lead is not None
                else "not measurable")
        return (
            f"Of {self.alerts:,} alerts, {self.precision:.0%} preceded a real "
            f"failure within {self.horizon:.0f} cycles, and {self.actionable_precision:.0%} "
            f"arrived with at least {ACTIONABLE_LEAD} cycles of notice. "
            f"Of {self.failures:,} failures, {self.recall:.0%} were alerted first. "
            f"Median warning: {lead}."
        )


@dataclass(slots=True)
class AlertLedger:
    """Records alerts, matches them to outcomes, and scores the result.

    Bounded on purpose. An unresolved alert older than the horizon can never be
    matched, so it is retired rather than accumulated - a ledger that grows
    without limit is the same defect as the session store that once retained
    5,001 states.
    """

    horizon: float = DEFAULT_HORIZON
    open_alerts: deque[OpenAlert] = field(default_factory=deque)
    scored: list[ScoredAlert] = field(default_factory=list)
    outcomes: list[Outcome] = field(default_factory=list)
    _alerted_failures: set[tuple[str, float]] = field(default_factory=set)

    # -- recording ---------------------------------------------------------
    def record_alert(
        self, machine_id: str, at: float, mode: str,
        predicted_lead: float | None = None, detail: str = "",
    ) -> None:
        self.open_alerts.append(
            OpenAlert(machine_id, float(at), mode, predicted_lead, detail)
        )

    def record_outcome(self, outcome: Outcome) -> None:
        """Resolve every open alert this outcome speaks to."""
        self.outcomes.append(outcome)
        if not outcome.failed:
            return
        still_open: deque[OpenAlert] = deque()
        for alert in self.open_alerts:
            if alert.machine_id != outcome.machine_id:
                still_open.append(alert)
                continue
            lead = outcome.at - alert.at
            if 0 <= lead <= self.horizon:
                self.scored.append(
                    ScoredAlert(alert, correct=True, lead=lead, matched=outcome)
                )
                self._alerted_failures.add((outcome.machine_id, outcome.at))
            else:
                still_open.append(alert)
        self.open_alerts = still_open

    def close(self, now: float) -> None:
        """Retire alerts whose horizon has passed without a failure.

        This is where a false alarm becomes a false alarm. Until the window
        closes an unmatched alert is merely unresolved, and scoring it early
        would punish a warning that is still in time to come true.
        """
        still_open: deque[OpenAlert] = deque()
        for alert in self.open_alerts:
            if now - alert.at > self.horizon:
                self.scored.append(ScoredAlert(alert, correct=False))
            else:
                still_open.append(alert)
        self.open_alerts = still_open

    # -- scoring -----------------------------------------------------------
    def scorecard(self) -> Scorecard:
        correct = [s for s in self.scored if s.correct]
        leads = [s.lead for s in correct if s.lead is not None]
        failures = [o for o in self.outcomes if o.failed]
        return Scorecard(
            alerts=len(self.scored),
            correct=len(correct),
            actionable=sum(1 for s in correct if s.actionable),
            failures=len(failures),
            caught=len({(o.machine_id, o.at) for o in failures}
                       & self._alerted_failures),
            median_lead=statistics.median(leads) if leads else None,
            horizon=self.horizon,
        )

    def by_mode(self) -> dict[str, Scorecard]:
        """Precision per failure mode.

        The aggregate hides the useful signal. A mode whose precision collapses
        while the others hold is a threshold that has drifted, and naming which
        one turns "the system got worse" into a work instruction.
        """
        modes = {s.alert.mode for s in self.scored}
        out: dict[str, Scorecard] = {}
        for mode in sorted(modes):
            subset = [s for s in self.scored if s.alert.mode == mode]
            correct = [s for s in subset if s.correct]
            leads = [s.lead for s in correct if s.lead is not None]
            out[mode] = Scorecard(
                alerts=len(subset), correct=len(correct),
                actionable=sum(1 for s in correct if s.actionable),
                failures=0, caught=0,
                median_lead=statistics.median(leads) if leads else None,
                horizon=self.horizon,
            )
        return out


def work_orders_from_csv(path: str | Any) -> list[WorkOrder]:
    """Read a CMMS export.

    Column names follow the common Maximo/SAP-PM shape, lowercased. A plant with
    different headers supplies a mapping rather than editing this - the same
    discipline the tag map uses, and for the same reason: guessing which column
    means "corrective" is how a system silently scores itself against the wrong
    ground truth.
    """
    import csv
    from pathlib import Path

    rows: list[WorkOrder] = []
    with Path(path).open(newline="") as fh:
        for raw in csv.DictReader(fh):
            row = {k.strip().lower(): (v or "").strip() for k, v in raw.items()}
            try:
                raised = float(row.get("raised_at") or row.get("cycle") or 0)
            except ValueError:
                continue
            rows.append(WorkOrder(
                machine_id=row.get("machine_id") or row.get("asset") or "unknown",
                raised_at=raised,
                kind=(row.get("kind") or row.get("work_type") or "corrective").lower(),
                found_fault=(row.get("found_fault") or "").lower()
                in ("1", "true", "yes", "y"),
                mode=row.get("mode") or None,
                reference=row.get("reference") or row.get("wo") or "",
            ))
    return rows


def score_replay(
    alerts: Iterable[tuple[str, float, str]],
    failures: Iterable[tuple[str, float]],
    horizon: float = DEFAULT_HORIZON,
) -> Scorecard:
    """Score a historical replay, where the labels are the ground truth.

    The production path uses the same ledger with work orders instead; this
    exists so the loop can be validated against data where the answer is known.
    """
    ledger = AlertLedger(horizon=horizon)
    events: list[tuple[float, int, Any]] = []
    for machine, at, mode in alerts:
        events.append((float(at), 0, (machine, mode)))
    for machine, at in failures:
        events.append((float(at), 1, machine))
    events.sort(key=lambda e: (e[0], e[1]))

    last = 0.0
    for at, kind, payload in events:
        ledger.close(at)
        if kind == 0:
            machine, mode = payload
            ledger.record_alert(machine, at, mode)
        else:
            ledger.record_outcome(Outcome(payload, at, failed=True))
        last = at
    ledger.close(last + horizon + 1)
    return ledger.scorecard()
