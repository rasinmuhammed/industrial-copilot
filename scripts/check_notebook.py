#!/usr/bin/env python3
"""Dry-run the Kaggle notebook without a GPU.

"Runs in one go" is a requirement, not a hope. This executes every code cell on
the CPU path — environment, DSL, corpus generation, validation — and verifies
that every generated target survives a DSL -> plan -> DSL round trip. Training,
evaluation and export are gated behind HAS_GPU and are skipped here.

    python scripts/check_notebook.py
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "planner_distillation.ipynb"


def main() -> int:
    nb = json.loads(NOTEBOOK.read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    scope: dict = {"__name__": "__main__"}

    print(f"Executing {len(cells)} code cells from {NOTEBOOK.name} (CPU path)\n")
    for i, cell in enumerate(cells, start=1):
        source = "".join(cell["source"])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, f"<cell {i}>", "exec"), scope)  # noqa: S102
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] cell {i}: {type(exc).__name__}: {exc}")
            return 1
        if i == 1:
            scope["HAS_GPU"] = False  # force the CPU path for the rest
        print(f"  [ok  ] cell {i}")

    rows = scope["uniq"]
    # The notebook now trains on JSON plans rather than a positional notation,
    # because the positional one made the model count pipe separators and it
    # scored 14.2%. See the notebook's section 3 for the evidence.
    to_target, to_plan = scope["plan_to_target"], scope["target_to_plan"]

    stable = sum(1 for r in rows if to_target(to_plan(r["target"])) == r["target"])
    ops = Counter(r["plan"]["op"] for r in rows)

    print(f"\n  corpus            {len(rows)} unique pairs")
    print(f"  round-trip stable {stable}/{len(rows)}")
    print(f"  ops covered       {len(ops)}  {dict(sorted(ops.items()))}")

    problems = []
    if stable != len(rows):
        problems.append(f"{len(rows) - stable} targets do not round-trip")
    if len(ops) < 10:
        problems.append(f"only {len(ops)} ops represented")
    if not any(r["plan"]["op"] == "refuse" for r in rows):
        problems.append(
            "no refusal examples: a constrained decoder that cannot decline "
            "will always answer, with the nearest valid plan"
        )
    # Word boundaries matter here: a naive substring match flags "describe the
    # THE-rmal gradient", which is correct English. The guard was wrong, not the
    # corpus — and a false positive in a build gate is worse than no gate,
    # because the next author learns to ignore it.
    doubled = re.compile(r"\b(the|a|an|of|to)\s+\1\b", re.IGNORECASE)
    if any(doubled.search(r["question"]) for r in rows):
        problems.append("duplicated articles in generated questions")

    if problems:
        print("\n  FAILED:")
        for p in problems:
            print(f"    · {p}")
        return 1
    print("\n  PASSED — the notebook runs top to bottom on the CPU path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
