#!/usr/bin/env python3
"""Measure the exemplar-retrieval thresholds.

The constants in copilot/planner/exemplars.py are set from this, not from
taste. Re-run it after any change to the embedder.

    python scripts/calibrate_exemplars.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.planner.exemplars import REUSE_THRESHOLD, HashingEmbedder  # noqa: E402

PARAPHRASES = [
    ("what is the failure rate by variant", "failure rate broken down by product type"),
    ("why did cycle 9016 fail", "what caused cycle 4045 to fail"),
    ("compare failed and healthy machines", "contrast failed against healthy machines"),
    ("how does failure rate vary with tool wear", "how does failure rate change with tool wear"),
    ("show me the cycles closest to failing", "list the cycles nearest to failure"),
    ("what drives failures", "which factors drive failures"),
]
UNRELATED = [
    ("why did cycle 9016 fail", "what is the average torque"),
    ("failure rate by variant", "when will the tool cross the overstrain limit"),
    ("compare failed and healthy", "can i trust this data"),
    ("what drives failures", "show me the safe torque range"),
    ("how does failure rate vary with tool wear", "why did cycle 9016 fail"),
]


def main() -> int:
    e = HashingEmbedder()
    sim = lambda a, b: float(e.embed(a) @ e.embed(b))  # noqa: E731

    same = [sim(a, b) for a, b in PARAPHRASES]
    diff = [sim(a, b) for a, b in UNRELATED]

    print("paraphrase pairs — must sit ABOVE the reuse threshold")
    for (a, b), v in zip(PARAPHRASES, same):
        flag = "ok " if v >= REUSE_THRESHOLD else "MISS"
        print(f"  [{flag}] {v:.3f}  {a[:36]:<36} | {b[:36]}")

    print("\nunrelated pairs — must sit BELOW it")
    for (a, b), v in zip(UNRELATED, diff):
        flag = "ok " if v < REUSE_THRESHOLD else "LEAK"
        print(f"  [{flag}] {v:.3f}  {a[:36]:<36} | {b[:36]}")

    lo, hi = max(diff), min(same)
    print(f"\n  paraphrase min {hi:.3f}   unrelated max {lo:.3f}")
    print(f"  separation gap [{lo:.3f}, {hi:.3f}]")
    print(f"  threshold in use {REUSE_THRESHOLD}  "
          f"({REUSE_THRESHOLD / lo:.1f}x the highest unrelated score)")

    leaks = sum(v >= REUSE_THRESHOLD for v in diff)
    if leaks:
        print(f"\n  FAILED: {leaks} unrelated pair(s) above the threshold")
        return 1
    print("\n  PASSED: no unrelated pair reaches the reuse threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
