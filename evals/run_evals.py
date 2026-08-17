#!/usr/bin/env python3
"""Eval harness.

    python evals/run_evals.py            full suite
    python evals/run_evals.py --json     machine-readable report
    python evals/run_evals.py --id X     one case

Hard gates fail the build. They are the properties the architecture claims, and
a claim that is not enforced is a wish:

    unsourced_numeral_rate  must be 0.000
    numeric_exactness       must be 1.000
    plan_validity_rate      must be >= 0.98
    refusal_correctness     must be 1.000
    premise_refutation      must be 1.000
"""

from __future__ import annotations

import argparse
import json

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.engine import Answer, Engine  # noqa: E402
from copilot.session import SessionState  # noqa: E402
from evals import reference  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.yaml"
REPORTS = Path(__file__).resolve().parent / "reports"

HARD_GATES = {
    "unsourced_numeral_rate": ("<=", 0.0),
    "numeric_exactness": (">=", 1.0),
    "plan_validity_rate": (">=", 0.98),
    "refusal_correctness": (">=", 1.0),
    "premise_refutation": (">=", 1.0),
}


@dataclass(slots=True)
class Check:
    kind: str
    passed: bool
    detail: str = ""


@dataclass(slots=True)
class CaseResult:
    id: str
    category: str
    criterion: int
    question: str
    checks: list[Check] = field(default_factory=list)
    latency_ms: float = 0.0
    plan_ms: float = 0.0
    tier: str = ""
    verified: bool = True
    refused: bool = False

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


# --------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------


def _expected(spec: dict[str, Any]) -> Any:
    if "ref" in spec:
        return reference.resolve(spec["ref"])
    return spec.get("value")


def check_assertion(spec: dict[str, Any], answer: Answer) -> Check:
    kind = spec["kind"]
    bundle = answer.bundle

    if kind == "refuses":
        return Check(kind, answer.refused, "" if answer.refused else "did not refuse")

    if answer.refused or bundle is None:
        return Check(kind, False, f"answer was refused: {answer.text[:80]}")

    if kind == "op_equals":
        actual = answer.plan.op.value if answer.plan else None
        return Check(kind, actual == spec["value"], f"op={actual} expected={spec['value']}")

    if kind == "numeric":
        slot = bundle.slots.get(spec["slot"])
        if slot is None:
            return Check(kind, False, f"slot {spec['slot']!r} missing")
        if slot.value is None:
            return Check(kind, False, f"slot {spec['slot']!r} abstained")
        expected = _expected(spec)
        tol = float(spec.get("tol", 0.0))
        ok = abs(float(slot.value) - float(expected)) <= tol
        return Check(kind, ok, f"{spec['slot']}={slot.value} expected={expected} tol={tol}")

    if kind == "abstains":
        slot = bundle.slots.get(spec["slot"])
        ok = slot is not None and slot.value is None
        return Check(kind, ok, f"{spec['slot']} did not abstain")

    if kind == "slot_absent":
        ok = spec["slot"] not in bundle.slots
        return Check(kind, ok, f"{spec['slot']} was present")

    if kind == "refutes_premise":
        ok = any(w.code == "premise_refuted" for w in bundle.warnings)
        return Check(kind, ok, "no premise_refuted warning")

    if kind == "raises_warning":
        ok = any(w.code == spec["code"] for w in bundle.warnings)
        codes = sorted({w.code for w in bundle.warnings})
        return Check(kind, ok, f"want {spec['code']}, got {codes}")

    if kind == "mentions":
        haystack = (answer.narration + " " + answer.text).lower()
        ok = any(phrase.lower() in haystack for phrase in spec["any_of"])
        return Check(kind, ok, f"none of {spec['any_of']} present")

    if kind == "row_count":
        actual = bundle.provenance.row_count
        return Check(kind, actual == spec["value"], f"rows={actual} expected={spec['value']}")

    if kind == "no_unsourced_numerals":
        leftovers = _unsourced(answer)
        return Check(kind, not leftovers, f"leaked {leftovers}")

    return Check(kind, False, f"unknown assertion kind {kind!r}")


def _unsourced(answer: Answer) -> list[str]:
    """Numerals in the narration that do not trace to a slot.

    Scans only the narrated prose. The scope line, provenance footer and warning
    strings are produced by engine code, never sampled from a model.
    """
    if answer.bundle is None:
        return []
    residual = answer.narration
    permitted = {s.render() for s in answer.bundle.slots.values() if s.value is not None}
    for value in sorted(permitted, key=len, reverse=True):
        residual = residual.replace(value, " ")
    return [t for t in residual.split() if any(c.isdigit() for c in t)]


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def run_case(engine: Engine, case: dict[str, Any]) -> list[CaseResult]:
    turns = case.get("turns") or [{"question": case["question"], "assertions": case["assertions"]}]
    state = SessionState()
    results = []

    for i, turn in enumerate(turns):
        question = turn["question"]
        started = time.perf_counter()
        answer = engine.ask(question, state)
        elapsed = (time.perf_counter() - started) * 1000.0

        result = CaseResult(
            id=case["id"] if len(turns) == 1 else f"{case['id']}#{i + 1}",
            category=case.get("category", "uncategorised"),
            criterion=int(case.get("criterion", 0)),
            question=question,
            latency_ms=elapsed,
            plan_ms=answer.plan_ms,
            tier=answer.tier,
            verified=answer.verified,
            refused=answer.refused,
        )
        for spec in turn.get("assertions", []):
            result.checks.append(check_assertion(spec, answer))
        results.append(result)
    return results


def aggregate(results: list[CaseResult], engine: Engine) -> dict[str, Any]:
    total = len(results)
    answered = [r for r in results if not r.refused]
    # A refusal the golden set asked for is a success, not a planning failure.
    expected_refusals = sum(
        1 for r in results if r.refused and any(c.kind == "refuses" and c.passed for c in r.checks)
    )

    numeric = [c for r in results for c in r.checks if c.kind == "numeric"]
    unsourced = [c for r in results for c in r.checks if c.kind == "no_unsourced_numerals"]
    refusals = [c for r in results for c in r.checks if c.kind == "refuses"]
    premise = [c for r in results for c in r.checks if c.kind == "refutes_premise"]

    latencies = sorted(r.latency_ms for r in results)
    plan_latencies = sorted(r.plan_ms for r in results)

    def pct(checks: list[Check]) -> float:
        return sum(c.passed for c in checks) / len(checks) if checks else 1.0

    def p(values: list[float], q: float) -> float:
        return values[min(int(len(values) * q), len(values) - 1)] if values else 0.0

    tiers: dict[str, int] = {}
    for r in results:
        tiers[r.tier] = tiers.get(r.tier, 0) + 1

    return {
        "cases": total,
        "cases_passed": sum(r.passed for r in results),
        "case_pass_rate": sum(r.passed for r in results) / total if total else 0.0,
        # Hard gates
        "unsourced_numeral_rate": 1.0 - pct(unsourced),
        "numeric_exactness": pct(numeric),
        "plan_validity_rate": (len(answered) + expected_refusals) / total if total else 0.0,
        "refusal_correctness": pct(refusals),
        "premise_refutation": pct(premise),
        # Quality
        "verification_rate": sum(r.verified for r in answered) / len(answered) if answered else 1.0,
        "numeric_assertions": len(numeric),
        # Performance
        "latency_p50_ms": round(p(latencies, 0.50), 2),
        "latency_p95_ms": round(p(latencies, 0.95), 2),
        "plan_p50_ms": round(p(plan_latencies, 0.50), 3),
        "plan_p95_ms": round(p(plan_latencies, 0.95), 3),
        "tier_counts": tiers,
        "tier_below_llm": sum(v for k, v in tiers.items() if k in {"cache", "grammar"}) / total
        if total
        else 0.0,
        "cache_hit_rate": round(engine.router.cache.hit_rate, 3),
        "provider": engine.provider_name,
    }


def gate_failures(metrics: dict[str, Any]) -> list[str]:
    failures = []
    for name, (op, bound) in HARD_GATES.items():
        value = metrics.get(name)
        if value is None:
            continue
        ok = value <= bound if op == "<=" else value >= bound
        if not ok:
            failures.append(f"{name} = {value:.3f}, requires {op} {bound}")
    return failures


def render(results: list[CaseResult], metrics: dict[str, Any]) -> str:
    lines = ["", "=" * 78, "  EVAL REPORT", "=" * 78, ""]

    by_category: dict[str, list[CaseResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    for category, group in sorted(by_category.items()):
        passed = sum(r.passed for r in group)
        lines.append(f"  {category:<14} {passed}/{len(group)}")
        for r in group:
            if r.passed:
                continue
            lines.append(f"      FAIL {r.id}: {r.question}")
            for check in r.failures:
                lines.append(f"           [{check.kind}] {check.detail}")
    lines.append("")

    lines.append("  " + "-" * 74)
    lines.append("  HARD GATES")
    for name, (op, bound) in HARD_GATES.items():
        value = metrics[name]
        ok = value <= bound if op == "<=" else value >= bound
        lines.append(f"    [{'PASS' if ok else 'FAIL'}] {name:<26} {value:.3f}  ({op} {bound})")

    lines.append("")
    lines.append("  QUALITY")
    lines.append(f"    cases passed                {metrics['cases_passed']}/{metrics['cases']}")
    lines.append(f"    numeric assertions checked  {metrics['numeric_assertions']}")
    lines.append(f"    verification rate           {metrics['verification_rate']:.3f}")

    lines.append("")
    lines.append("  PERFORMANCE")
    lines.append(f"    latency p50 / p95           {metrics['latency_p50_ms']} / {metrics['latency_p95_ms']} ms")
    lines.append(f"    planning p50 / p95          {metrics['plan_p50_ms']} / {metrics['plan_p95_ms']} ms")
    lines.append(f"    resolved below the LLM tier {metrics['tier_below_llm']:.0%}")
    lines.append(f"    cache hit rate              {metrics['cache_hit_rate']:.0%}")
    lines.append(f"    provider                    {metrics['provider']}")
    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--id", help="run a single case by id")
    parser.add_argument("--no-save", action="store_true", help="skip writing a report file")
    args = parser.parse_args(argv)

    with GOLDEN.open() as fh:
        golden = yaml.safe_load(fh)

    cases = golden["questions"]
    if args.id:
        cases = [c for c in cases if c["id"] == args.id]
        if not cases:
            print(f"no case with id {args.id!r}", file=sys.stderr)
            return 2

    engine = Engine.build()
    results: list[CaseResult] = []
    for case in cases:
        results.extend(run_case(engine, case))

    metrics = aggregate(results, engine)
    failures = gate_failures(metrics)

    if args.json:
        print(json.dumps({"metrics": metrics, "gate_failures": failures}, indent=2))
    else:
        print(render(results, metrics))
        if failures:
            print("\n  BUILD FAILED - hard gate violations:")
            for failure in failures:
                print(f"    · {failure}")
            print()

    if not args.no_save:
        REPORTS.mkdir(exist_ok=True)
        payload = {
            "metrics": metrics,
            "gate_failures": failures,
            "cases": [
                {
                    "id": r.id,
                    "category": r.category,
                    "criterion": r.criterion,
                    "question": r.question,
                    "passed": r.passed,
                    "tier": r.tier,
                    "latency_ms": round(r.latency_ms, 2),
                    "failures": [{"kind": c.kind, "detail": c.detail} for c in r.failures],
                }
                for r in results
            ],
        }
        (REPORTS / "latest.json").write_text(json.dumps(payload, indent=2))

    return 1 if failures or any(not r.passed for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
