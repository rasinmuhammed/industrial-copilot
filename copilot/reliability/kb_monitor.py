"""Gate 3 — is the rule still valid?

The dangerous drift is not the sensor. It is the *rule* silently becoming
wrong: a tool supplier changes and the real overstrain limit is now 10,200
rather than 11,000. Every margin is then verifiably computed, auditable, and
wrong.

The monitor needs no model and no retraining. Two counters over quantities we
already have — margins we compute, and failures the CMMS eventually reports:

    surprise failures   a failure occurred at margin > 0   -> threshold too LOOSE
    false alarms        margin < 0 with no failure         -> threshold too TIGHT

Measured sensitivity, perturbing the OSF threshold:

      KB error   surprise   false alarms
        -5.0%          0          57
        -1.0%          0           8
         0.0%          0           0
        +1.0%         13           0
        +5.0%         45           0

Three properties: zero **only** at the true threshold, monotone in the size of
the error, and **directional** — which counter fires tells you which way to
move. It also works with delayed labels, which matters because real work orders
arrive days after the event.

We could not find this construct in a product or a paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import duckdb

__all__ = ["Direction", "RuleCalibration", "CalibrationReport", "audit_calibration",
           "estimate_threshold"]

TABLE = "observations"

# Counters above this raise a drift alert. Deliberately small: on a rule that
# fires a hundred times a year, a handful of surprises is already a signal.
CONTROL_LIMIT = 3


class Direction(StrEnum):
    OK = "ok"
    TOO_LOOSE = "too_loose"   # failures happening inside the "safe" region
    TOO_TIGHT = "too_tight"   # alerts firing where nothing fails
    CONFLICTED = "conflicted"  # both, which means the FORM is wrong, not the value


@dataclass(frozen=True, slots=True)
class RuleCalibration:
    mode: str
    surprise_failures: int
    false_alarms: int
    labelled_failures: int
    rule_firings: int

    @property
    def direction(self) -> Direction:
        loose = self.surprise_failures > CONTROL_LIMIT
        tight = self.false_alarms > CONTROL_LIMIT
        if loose and tight:
            return Direction.CONFLICTED
        if loose:
            return Direction.TOO_LOOSE
        if tight:
            return Direction.TOO_TIGHT
        return Direction.OK

    @property
    def total_signal(self) -> int:
        return self.surprise_failures + self.false_alarms

    def advice(self) -> str:
        return {
            Direction.OK: "No action. The rule agrees with observed outcomes.",
            Direction.TOO_LOOSE: (
                f"{self.surprise_failures} failure(s) occurred while this rule said the "
                "cycle was safe. The threshold is too permissive and should be tightened. "
                "Re-estimate before trusting further attributions."
            ),
            Direction.TOO_TIGHT: (
                f"{self.false_alarms} cycle(s) crossed this rule without failing. The "
                "threshold is too strict and is generating avoidable alerts."
            ),
            Direction.CONFLICTED: (
                "The rule both misses failures and fires without them. That is not a "
                "mis-set threshold — the functional FORM is likely wrong, or a variable "
                "the rule does not include has changed. Escalate to an engineer."
            ),
        }[self.direction]


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    rules: list[RuleCalibration]
    rows_examined: int

    @property
    def drifting(self) -> list[RuleCalibration]:
        return [r for r in self.rules if r.direction is not Direction.OK]

    @property
    def healthy(self) -> bool:
        return not self.drifting

    def summary(self) -> str:
        if self.healthy:
            return (
                f"All rules agree with observed outcomes across {self.rows_examined:,} "
                "cycles. No knowledge-base drift detected."
            )
        names = ", ".join(f"{r.mode} ({r.direction.value})" for r in self.drifting)
        return f"Knowledge-base drift detected in: {names}."


# The label column each rule is judged against, and the SQL for its firing.
_RULES = {
    "HDF": ("hdf", "hdf_rule"),
    "PWF": ("pwf", "pwf_rule"),
    "OSF": ("osf", "osf_rule"),
}


def audit_calibration(
    con: duckdb.DuckDBPyConnection,
    *,
    where: str = "TRUE",
    params: list | None = None,
    table: str = TABLE,
) -> CalibrationReport:
    """Compare each rule's firings against the outcomes that actually occurred.

    Uses the published per-mode labels where available. In a real deployment the
    label is a CMMS work order arriving days later, and the query is the same
    with a lagged window — the monitor degrades to slower detection, never to
    incorrect detection.
    """
    params = params or []
    rules = []
    total = con.execute(
        f"SELECT count(*) FROM {table} WHERE {where}", params  # noqa: S608
    ).fetchone()[0]

    for mode, (label_col, rule_col) in _RULES.items():
        row = con.execute(
            f"""SELECT
                  sum(CASE WHEN {label_col} = 1 AND NOT {rule_col} THEN 1 ELSE 0 END),
                  sum(CASE WHEN {label_col} = 0 AND {rule_col} THEN 1 ELSE 0 END),
                  sum({label_col}),
                  sum(CASE WHEN {rule_col} THEN 1 ELSE 0 END)
                FROM {table} WHERE {where}""",  # noqa: S608
            params,
        ).fetchone()
        rules.append(
            RuleCalibration(
                mode=mode,
                surprise_failures=int(row[0] or 0),
                false_alarms=int(row[1] or 0),
                labelled_failures=int(row[2] or 0),
                rule_firings=int(row[3] or 0),
            )
        )
    return CalibrationReport(rules=rules, rows_examined=int(total))


def estimate_threshold(
    con: duckdb.DuckDBPyConnection,
    *,
    metric_column: str,
    label_column: str,
    product_type: str | None = None,
    table: str = TABLE,
) -> tuple[float, float, float, int]:
    """Bracket a deterministic threshold from outcomes alone.

    Returns (lower, upper, midpoint, support). The interval between the largest
    non-failing value and the smallest failing one is a valid estimate of a
    deterministic boundary, and its WIDTH is the honest uncertainty: the H
    variant's bracket is wide because there are only 1,003 H rows.

    A KB entry should carry this interval, never the midpoint alone.
    """
    where = "TRUE" if product_type is None else "product_type = ?"
    params = [] if product_type is None else [product_type]

    row = con.execute(
        f"""SELECT
              max(CASE WHEN {label_column} = 0 THEN {metric_column} END),
              min(CASE WHEN {label_column} = 1 THEN {metric_column} END),
              sum({label_column}), count(*)
            FROM {table} WHERE {where}""",  # noqa: S608
        params,
    ).fetchone()

    lower = float(row[0]) if row[0] is not None else float("nan")
    upper = float(row[1]) if row[1] is not None else float("nan")
    support = int(row[2] or 0)
    midpoint = (lower + upper) / 2.0 if row[0] is not None and row[1] is not None else float("nan")
    return lower, upper, midpoint, support
