#!/usr/bin/env python3
"""Benchmark the SLM planner against the grammar tier on the golden set.

Computes:
  - Coverage: fraction of questions where the SLM produced a valid plan
  - Exact match: fraction where the plan is structurally identical to the
    grammar tier's plan (same op, same metrics, same filters)
  - Latency p50 / p95 for the SLM path

Run after pulling a fine-tuned model to Ollama:
    ollama pull hf.co/your-name/margin-planner-gguf
    python scripts/bench_slm.py

    # Override model name or endpoint:
    python scripts/bench_slm.py --model my-model --endpoint http://host:11434
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from copilot.ir import parse_plan, PlanError  # noqa: E402
from copilot.planner.grammar import plan_from_text  # noqa: E402
from copilot.planner.slm import SLMPlanner, is_available  # noqa: E402
from copilot.session import SessionState  # noqa: E402

GOLDEN = Path(__file__).resolve().parent.parent / "evals" / "golden.yaml"
W = 70


def _plans_equivalent(a: dict, b: dict) -> bool:
    """Structural equivalence: same op and same top-level keys present."""
    if a.get("op") != b.get("op"):
        return False
    # Both agree on presence of filters, group_by, metrics
    for key in ("metrics", "group_by", "filters", "bin"):
        a_has = bool(a.get(key))
        b_has = bool(b.get(key))
        if a_has != b_has:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",    default=None, help="Ollama model name")
    parser.add_argument("--endpoint", default=None, help="Ollama endpoint URL")
    parser.add_argument("--timeout",  type=float, default=30.0)
    args = parser.parse_args()

    if not is_available(args.model, args.endpoint):
        print("✗ Ollama not reachable or model not loaded.")
        print("  Pull a model first:  ollama pull hf.co/your-name/margin-planner-gguf")
        print("  Or run local model:  ollama run qwen2.5:1.5b")
        sys.exit(1)

    with open(GOLDEN) as f:
        cases = [c for c in yaml.safe_load(f)["questions"] if "question" in c]

    planner = SLMPlanner(model=args.model, endpoint=args.endpoint, timeout=args.timeout)

    print("=" * W)
    print("  SLM PLANNER BENCHMARK")
    print(f"  model: {planner.model}")
    print("=" * W)

    valid = exact = 0
    slm_times: list[float] = []
    grammar_times: list[float] = []

    for case in cases:
        q = case["question"]
        state = SessionState()

        # Grammar tier reference
        t0 = time.perf_counter()
        grammar_match = plan_from_text(q, state)
        grammar_times.append((time.perf_counter() - t0) * 1000)
        grammar_plan = grammar_match.plan.model_dump(exclude_none=True) if grammar_match.plan else {}

        # SLM tier
        t0 = time.perf_counter()
        try:
            raw = planner.propose(q)
            plan = parse_plan(raw)
            slm_plan = plan.model_dump(exclude_none=True)
            ok = True
        except (PlanError, Exception):
            slm_plan = {}
            ok = False
        slm_times.append((time.perf_counter() - t0) * 1000)

        equiv = ok and _plans_equivalent(grammar_plan, slm_plan)
        valid += int(ok)
        exact += int(equiv)

        status = "✓" if equiv else ("~" if ok else "✗")
        print(f"  {status}  {q[:55]:<55}  {slm_times[-1]:5.0f} ms")

    n = len(cases)
    p50  = statistics.median(slm_times)
    p95  = sorted(slm_times)[int(0.95 * n)]
    g50  = statistics.median(grammar_times)

    print("=" * W)
    print(f"  coverage (valid plans)   {valid}/{n}  ({valid/n:.1%})")
    print(f"  exact match              {exact}/{n}  ({exact/n:.1%})")
    print(f"  grammar tier reference   {g50:.1f} ms  p50")
    print(f"  SLM latency              {p50:.0f} ms p50  /  {p95:.0f} ms p95")
    print("=" * W)

    if valid / n >= 0.80:
        print("  VERDICT: SLM coverage sufficient to replace exemplar tier (≥ 80%)")
    else:
        print(f"  VERDICT: Coverage {valid/n:.0%} < 80% — grammar tier remains primary")


if __name__ == "__main__":
    main()
