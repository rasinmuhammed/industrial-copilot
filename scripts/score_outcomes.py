#!/usr/bin/env python3
"""Score the streaming alerts against what actually happened.

    python scripts/score_outcomes.py

The one quality nothing here used to measure. Coverage, soundness and
false-alarm rate were all gated; whether a warning was followed by the failure
it warned about was never checked at all.

Ground truth is the failure label on a historical replay. In production the same
ledger takes work orders instead - see copilot/outcomes.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb  # noqa: E402

from copilot.config import settings  # noqa: E402
from copilot.outcomes import AlertLedger, Outcome  # noqa: E402
from copilot.stream import replay  # noqa: E402

HORIZON = 60
LIMIT = 4000
PREDICTIVE = {"predicted", "approaching", "crossed"}


def main() -> int:
    con = duckdb.connect(str(settings().db_path), read_only=True)
    failures = {
        (r[0], float(r[1]))
        for r in con.execute(
            "SELECT machine_id, udi FROM observations WHERE machine_failure = 1"
        ).fetchall()
    }

    ledger = AlertLedger(horizon=HORIZON)
    last = 0.0
    for tick in replay(limit=LIMIT):
        last = float(tick.udi)
        ledger.close(last)
        for alert in tick.alerts:
            if alert.kind.value in PREDICTIVE:
                ledger.record_alert(tick.machine_id, last, alert.kind.value)
        if (tick.machine_id, last) in failures:
            ledger.record_outcome(Outcome(tick.machine_id, last, failed=True))
    ledger.close(last + HORIZON + 1)

    card = ledger.scorecard()
    print(f"\n  OUTCOME LOOP - {LIMIT:,} cycles, {HORIZON}-cycle horizon")
    print("  " + "=" * 66)
    for line in card.summary().split(". "):
        if line.strip():
            print(f"  {line.strip().rstrip('.')}.")

    print("\n  by alert kind:")
    for kind, sub in ledger.by_mode().items():
        lead = f"{sub.median_lead:.0f}" if sub.median_lead is not None else "-"
        print(f"    {kind:<14} {sub.alerts:>4} alerts   "
              f"precision {sub.precision:>4.0%}   median lead {lead:>3} cycles")

    print(
        "\n  Not every mode is forecastable. HDF and PWF are threshold conditions\n"
        "  on the CURRENT operating point, so the point IS the failure and no\n"
        "  lead exists to give. Only OSF and TWF accumulate, and only they can\n"
        "  be warned about in advance.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
