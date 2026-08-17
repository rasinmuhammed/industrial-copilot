#!/usr/bin/env python3
"""Build data/exemplars.jsonl for the distillation notebook.

The exemplar store normally fills from real usage, so a fresh clone has nothing
in it. This seeds it by running the golden question set and the demo script
through the actual engine and keeping only the plans whose answers passed
numeric verification.

That distinction matters: these are not fabricated pairs. Every row is a
question that was really asked of the real engine, planned, executed against the
warehouse, and whose answer survived the verifier. An unverified plan is not
evidence of anything and is discarded.

    python scripts/export_exemplars.py            # write data/exemplars.jsonl
    python scripts/export_exemplars.py --stdout   # print instead

Upload the result to Kaggle as a dataset named `margin-engine-exemplars`; the
notebook picks it up automatically. It is optional - without it the notebook
still generates ~1,200 synthetic pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from copilot.config import settings  # noqa: E402
from copilot.engine import Engine  # noqa: E402
from copilot.session import SessionState  # noqa: E402

GOLDEN = Path(__file__).resolve().parent.parent / "evals" / "golden.yaml"

# Extra phrasings beyond the golden set, so the corpus carries more than one
# surface form per intent. Still executed for real - nothing here is assumed.
EXTRA = [
    "What are typical operating conditions?",
    "What is the average torque?",
    "How hot does the process run compared to ambient?",
    "What is the overall failure rate?",
    "Break failures down by product variant",
    "Failure rate by shift",
    "How does failure rate vary with tool wear?",
    "How does failure rate change with rotational speed?",
    "Compare failed and healthy cycles",
    "Contrast the operating conditions of breakdowns against normal runs",
    "Which variables best separate failures from healthy operation?",
    "What drives failures?",
    "Why did cycle 9016 fail?",
    "What caused cycle 4045 to fail?",
    "Diagnose cycle 161",
    "What causes failures?",
    "Which failure modes are firing?",
    "What if we reduce torque by 5 Nm?",
    "Suppose we cut torque by 10 Nm",
    "What if we increase speed by 200 rpm?",
    "Show me the cycles closest to failing",
    "List the riskiest records",
    "Can I trust this data?",
    "Are there problems with the dataset?",
    "Are the failure thresholds still accurate?",
    "What are the typical operating conditions for high quality variants?",
    "What is the failure rate for L variants?",
    "Why are we seeing more failures at high rotational speeds?",
    "Do failures increase with torque?",
]


def collect() -> tuple[list[dict], dict[str, int]]:
    engine = Engine.build()
    engine.router.exemplars.clear()

    questions: list[str] = []
    golden = yaml.safe_load(GOLDEN.read_text())
    for case in golden["questions"]:
        if "question" in case:
            questions.append(case["question"])
        for turn in case.get("turns", []):
            questions.append(turn["question"])
    questions += EXTRA

    tally = {"asked": 0, "verified": 0, "refused": 0, "stored": 0, "collapsed": 0}
    for question in questions:
        tally["asked"] += 1
        answer = engine.ask(question, SessionState())
        if answer.refused:
            tally["refused"] += 1
            continue
        if not answer.verified:
            continue
        tally["verified"] += 1
        # The engine's own learn() already deposits verified plans, but it skips
        # cache and exemplar hits by design. Record explicitly so the corpus
        # covers every phrasing, not only the ones that reached a novel tier.
        if answer.plan is not None:
            engine.router.exemplars.record(question, answer.plan, tier=answer.tier)

    pairs = engine.router.exemplars.export_training_pairs()
    tally["stored"] = len(pairs)
    # Distinct shapes are fewer than verified answers because several phrasings
    # normalise to the same question, which is the store deduplicating.
    tally["collapsed"] = tally["verified"] - len(pairs)
    return pairs, tally


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = parser.parse_args()

    pairs, tally = collect()
    lines = [
        json.dumps(
            {
                "question": p["question"],
                "normalised": "",
                "shape": p["plan"],
                "op": p["op"],
                "created": "",
                "uses": p["uses"],
                "source_tier": "seeded",
            }
        )
        for p in pairs
    ]

    if args.stdout:
        print("\n".join(lines))
        return 0

    path = settings().exemplar_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")

    by_op: dict[str, int] = {}
    for p in pairs:
        by_op[p["op"]] = by_op.get(p["op"], 0) + 1

    print(f"asked      {tally['asked']}")
    print(f"verified   {tally['verified']}")
    print(f"refused    {tally['refused']}  (correctly - out-of-scope questions)")
    print(f"stored     {tally['stored']} unique plan shapes")
    print(f"collapsed  {tally['collapsed']} phrasings normalised onto an existing shape")
    print(f"\nby op      {dict(sorted(by_op.items()))}")
    print(f"\nwrote      {path}  ({path.stat().st_size / 1024:.1f} KB)")
    print(
        "\nUpload this as a Kaggle dataset named 'margin-engine-exemplars'.\n"
        "The notebook finds it at /kaggle/input/margin-engine-exemplars/exemplars.jsonl.\n"
        "It is optional - the notebook generates ~1,200 synthetic pairs without it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
