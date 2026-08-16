#!/usr/bin/env python3
"""The README's example must be what the system actually says.

    python scripts/verify_readme.py [--update]

WHY
---
The README opened with a hand-typed transcript of the brief's own example
question. Over time it drifted from the product in two ways, and the second is
the serious one:

  * the rates were stale — "12.17%" against a live 12.2%
  * it asserted a mechanism the system never emits: "every high-rpm failure is a
    power STALL ... mean 10.6 N·m at 2638 rpm". The real figures are 15.3 N·m at
    2385 rpm, and no code path produces that sentence.

The first is drift. The second is a documentation claim that the product is
better than it is — written in the first twenty lines, where an evaluator reads
it and then discovers the gap themselves.

Everything else here is gated. Every numeral in an answer traces to a computed
slot; every documented figure is re-derived by `make verify`; every threshold is
audited against the data. The README was the one surface with hard-coded numbers
and no verifier, which is exactly where an error survives longest.

So the transcript is generated and this script gates it. `--update` regenerates
the block; without it, a mismatch fails the build.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.engine import Engine  # noqa: E402
from copilot.session import SessionState  # noqa: E402

README = Path(__file__).resolve().parent.parent / "README.md"
QUESTION = "Why are we seeing more failures at high rotational speeds?"

BEGIN = "<!-- BEGIN GENERATED EXAMPLE -->"
END = "<!-- END GENERATED EXAMPLE -->"

# The provenance footer carries a wall-clock timing that legitimately varies run
# to run. Everything else in it — plan hash, kb version, row count — is content
# and must match, so only the timing is stripped.
_TIMING = re.compile(r"\s*·\s*[\d.]+\s*ms")


def render() -> str:
    answer = Engine.build().ask(QUESTION, SessionState())
    body = _TIMING.sub("", answer.text).rstrip()
    return f'$ make ask Q="{QUESTION}"\n\n{body}'


def extract(text: str) -> str | None:
    match = re.search(
        rf"{re.escape(BEGIN)}\s*```\n(.*?)\n```\s*{re.escape(END)}", text, re.S
    )
    return match.group(1).strip() if match else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="rewrite the block instead of failing on a mismatch")
    args = ap.parse_args()

    text = README.read_text()
    current = extract(text)
    if current is None:
        print("  FAILED — no generated-example block found in README.md")
        return 1

    expected = render()
    if current == expected:
        print("  PASSED — the README example is what the system says.")
        return 0

    if args.update:
        block = f"{BEGIN}\n```\n{expected}\n```\n{END}"
        README.write_text(
            re.sub(
                rf"{re.escape(BEGIN)}.*?{re.escape(END)}", block, text, flags=re.S
            )
        )
        print("  UPDATED — README example regenerated from live output.")
        return 0

    print("  FAILED — the README example no longer matches the system.\n")
    for line in _diff(current, expected):
        print(f"    {line}")
    print("\n  Run `python scripts/verify_readme.py --update` to regenerate it.")
    return 1


def _diff(current: str, expected: str) -> list[str]:
    import difflib

    return list(difflib.unified_diff(
        current.splitlines(), expected.splitlines(),
        fromfile="README", tofile="live", lineterm="", n=1,
    ))[:24]


if __name__ == "__main__":
    raise SystemExit(main())
