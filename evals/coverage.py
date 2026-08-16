#!/usr/bin/env python3
"""Risk-coverage: what fraction of real questions do we actually answer?

    python evals/coverage.py

WHY THIS EXISTS
---------------
Every metric this project had measured CORRECTNESS: 27/27 golden cases, 46/46
dataset claims, zero unsourced numerals. All of them are satisfied perfectly by
a system that answers almost nothing.

That is not hypothetical. The streaming path drifted to a 48% abstain rate and
nobody noticed, because no gate looked at coverage. A system that is never wrong
because it rarely speaks is not trusted — it is ignored, which is a worse
outcome than being occasionally wrong and known to be.

The honest object is the pair, which is what selective prediction has always
used: how much do you answer, and how good is what you answer. Optimising either
alone is trivial. Coverage alone is a system that guesses; soundness alone is a
system that shrugs.

WHY IT IS ALSO THE DIFFERENTIATOR
---------------------------------
Nobody publishes the pair. AssetOpsBench reports ~65% task completion without
checking whether the completions were right. AgentAbstain measures refusal in
isolation. A plant manager needs both numbers from the same run, and the
architecture here is built to produce them.

WHAT IS MEASURED
----------------
  coverage           of questions a competent analyst could answer from this
                     data, the fraction we answered
  soundness          of answers given, the fraction that are structurally
                     sound: a valid plan, every numeral traced to a slot
  refusal precision  of questions we declined, the fraction that genuinely
                     should be declined
  silent failure     answered when we should have refused — the only truly
                     bad cell, because it is a confident wrong answer

An honest system maximises coverage subject to silent failures being zero.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.engine import Answer, Engine  # noqa: E402
from copilot.session import SessionState  # noqa: E402

# ── The question bank.
#
# `answerable` means a competent analyst with this dataset in front of them
# could produce a defensible answer. It does NOT mean our system does — that is
# the thing being measured, and labelling it any other way would make the
# benchmark self-congratulatory.
#
# Phrasings are deliberately uneven: terse, verbose, misspelled, casual. An
# engineer at a terminal does not write like a golden test case.
ANSWERABLE: list[str] = [
    # -- 1. Understand machine behaviour -----------------------------------
    "What are the typical operating conditions?",
    "what does normal look like here",
    "Describe the torque distribution.",
    "how hot does the process usually run",
    "typical rpm?",
    "What is the average tool wear?",
    "give me a summary of the air temperature",
    "What are the operating conditions for H variants?",
    "describe conditions on low quality product",
    "What's the spread on rotational speed?",
    "how much power do these machines draw",
    "what is the normal temperature differential",
    "Show me the safe operating window.",
    "what torque should I run at 150 minutes of wear",
    "Show me the cycles closest to failing.",
    "which cycles are nearest the boundary",
    # -- 2. Analyse historical data ----------------------------------------
    "What's the overall failure rate?",
    "how often do these machines fail",
    "what proportion of cycles failed",
    "How many failures are in this data?",
    "failure rate by product variant",
    "break the failures down by type",
    "how does failure rate differ across variants",
    "failure rate by shift",
    "Show failures grouped by machine.",
    "How does failure rate vary with tool wear?",
    "does failure rate go up with torque",
    "trend of failures against rotational speed",
    "relationship between failures and power",
    "failure rate across speed bands",
    "how many H variants are there",
    # -- 3. Investigate failures -------------------------------------------
    "Compare the operating conditions of machines that failed versus those that did not.",
    "contrast failed and healthy cycles",
    "how do broken machines differ from working ones",
    "what distinguishes the failures",
    "Why did cycle 9016 fail?",
    "what caused cycle 4045 to fail",
    "root cause for cycle 2750",
    "diagnose cycle 7",
    "What causes failures?",
    "what are the main failure modes",
    "which failure modes are firing",
    "attribute the failures to causes",
    "Which variables best separate failures from healthy operation?",
    "what drives failures",
    "biggest factors behind breakdowns",
    "which parameters predict failure",
    "is tool wear the main cause of failure",
    "how many failures had more than one mode",
    # -- Premise-bearing (must be answered AND the premise tested) ---------
    "Why are we seeing more failures at high rotational speeds?",
    "Why do high quality variants fail more often?",
    "why do low quality variants fail more",
    "why is torque causing so many failures",
    # -- Prescriptive / forecast -------------------------------------------
    "What if we reduce torque by 5 Nm?",
    "what happens if we increase speed by 100 rpm",
    "impact of dropping torque 10 Nm",
    "when will the tool cross the overstrain limit at 200 min wear",
    "how long until failure at 60 Nm",
    # RNF is a parameter-independent background rate. Answering IS correct here,
    # provided the answer states that the cause is not determinable — which it
    # does. An earlier version of this bank labelled it unanswerable and scored
    # a correct answer as a silent failure.
    "Why did the random failures happen?",
    # -- Data quality -------------------------------------------------------
    "Can I trust this data?",
    "are there problems with this dataset",
    "is the labelling reliable",
    "data quality report",
]

# Questions that MUST be declined. Answering any of these is a silent failure —
# a confident answer to something the data cannot support.
UNANSWERABLE: list[tuple[str, str]] = [
    ("What is the vibration signature on this machine?", "no vibration sensor exists"),
    ("show me the bearing temperature", "no bearing sensor exists"),
    ("what is the oil pressure", "no pressure sensor exists"),
    ("how loud is the machine", "no acoustic sensor exists"),
    ("what is the spindle runout", "not measured"),
    ("purple monkey dishwasher", "not a question"),
    ("asdfgh", "not a question"),
    ("what is the meaning of life", "out of domain"),
    ("who is the operator on shift 3", "no personnel data"),
    ("how much does a replacement tool cost", "no cost data"),
    ("what is the weather tomorrow", "out of domain"),
]

_NUMERAL = re.compile(r"\d")


@dataclass
class Outcome:
    question: str
    expected_answerable: bool
    answered: bool
    sound: bool
    refused: bool
    tier: str
    ms: float
    degraded: bool = False
    detail: str = ""

    @property
    def silent_failure(self) -> bool:
        """Answered something we had no business answering. The only fatal cell."""
        return self.answered and not self.expected_answerable

    @property
    def missed(self) -> bool:
        """Declined something we should have handled. Costly, not dangerous."""
        return not self.answered and self.expected_answerable


@dataclass
class Report:
    outcomes: list[Outcome] = field(default_factory=list)

    def _sel(self, **kw) -> list[Outcome]:
        return [
            o for o in self.outcomes
            if all(getattr(o, k) == v for k, v in kw.items())
        ]

    @property
    def coverage(self) -> float:
        answerable = self._sel(expected_answerable=True)
        if not answerable:
            return 0.0
        return sum(o.answered for o in answerable) / len(answerable)

    @property
    def soundness(self) -> float:
        answered = [o for o in self.outcomes if o.answered]
        if not answered:
            return 0.0
        return sum(o.sound for o in answered) / len(answered)

    @property
    def refusal_precision(self) -> float:
        refused = [o for o in self.outcomes if not o.answered]
        if not refused:
            return 1.0
        return sum(not o.expected_answerable for o in refused) / len(refused)

    @property
    def silent_failures(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.silent_failure]

    @property
    def missed(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.missed]

    def as_dict(self) -> dict:
        answered = [o for o in self.outcomes if o.answered]
        latencies = sorted(o.ms for o in answered) or [0.0]
        return {
            "questions": len(self.outcomes),
            "coverage": round(self.coverage, 4),
            "soundness": round(self.soundness, 4),
            "refusal_precision": round(self.refusal_precision, 4),
            "silent_failures": len(self.silent_failures),
            "missed": len(self.missed),
            "latency_ms_p50": round(latencies[len(latencies) // 2], 2),
            "latency_ms_p95": round(latencies[int(len(latencies) * 0.95)], 2),
            "degraded": sum(o.degraded for o in self.outcomes),
            "tiers": _histogram(o.tier for o in self.outcomes),
        }


def _histogram(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# NOTE: an earlier version of this harness re-implemented the unsourced-numeral
# check here and reported 53.8% soundness — against a shipping verifier that
# enforces zero and passes every golden case. The harness was wrong: a slot
# holding 39.98690 renders at its declared 4 significant figures as "39.99",
# and the naive comparison called that untraceable.
#
# The lesson generalises past the bug. A benchmark that re-implements the thing
# it measures is measuring its own reimplementation. `Answer.verified` is the
# verdict the fail-closed renderer actually reached, so soundness now reads the
# shipping decision rather than a lookalike of it.


def evaluate(engine: Engine, question: str, answerable: bool) -> Outcome:
    session = SessionState()
    try:
        answer = engine.ask(question, session)
    except Exception as exc:                      # a crash is not a refusal
        return Outcome(question, answerable, False, False, False, "error", 0.0,
                       detail=f"{type(exc).__name__}: {exc}")

    refused = bool(getattr(answer, "refused", False))
    bundle = getattr(answer, "bundle", None)
    # An "answer" with no computed slots is a refusal wearing a sentence, and a
    # refusal carries no bundle at all.
    has_content = bool(bundle is not None and bundle.slots)
    answered = not refused and has_content

    # The renderer is fail-closed and runs on every answer, so this reads its
    # verdict rather than second-guessing it. `degraded` means it fell back to
    # the template after the model narration failed verification — still sound,
    # but worth counting separately.
    sound = bool(getattr(answer, "verified", False))
    degraded = bool(getattr(answer, "degraded", False))
    detail = "" if sound else "failed numeric verification"
    if sound and degraded:
        detail = "degraded to template narration"

    return Outcome(
        question=question,
        expected_answerable=answerable,
        answered=answered,
        sound=sound,
        refused=refused,
        tier=getattr(answer, "tier", "?"),
        ms=bundle.provenance.elapsed_ms if bundle is not None else 0.0,
        degraded=degraded,
        detail=detail,
    )


def run(engine: Engine) -> Report:
    report = Report()
    for question in ANSWERABLE:
        report.outcomes.append(evaluate(engine, question, True))
    for question, _why in UNANSWERABLE:
        report.outcomes.append(evaluate(engine, question, False))
    return report


def render(report: Report) -> str:
    m = report.as_dict()
    lines = [
        "",
        "  RISK-COVERAGE",
        "  " + "=" * 66,
        f"  questions            {m['questions']}",
        f"  coverage             {m['coverage']:.1%}   (answerable questions we answered)",
        f"  soundness            {m['soundness']:.1%}   (answers with every numeral sourced)",
        f"  refusal precision    {m['refusal_precision']:.1%}   (declines that were correct)",
        f"  silent failures      {m['silent_failures']}   (answered what we should not have)",
        f"  missed               {m['missed']}   (declined what we should have answered)",
        f"  latency p50 / p95    {m['latency_ms_p50']:.1f} / {m['latency_ms_p95']:.1f} ms",
        f"  tiers                {m['tiers']}",
    ]
    if report.silent_failures:
        lines += ["", "  SILENT FAILURES — a confident answer to an unanswerable question:"]
        for o in report.silent_failures:
            lines.append(f"    {o.question[:62]:64s} [{o.tier}]")
    if report.missed:
        lines += ["", f"  MISSED — declined but answerable ({len(report.missed)}):"]
        for o in report.missed[:20]:
            lines.append(f"    {o.question[:62]:64s} [{o.tier}] {o.detail[:28]}")
    unsound = [o for o in report.outcomes if o.answered and not o.sound]
    if unsound:
        lines += ["", f"  UNSOUND ({len(unsound)}):"]
        for o in unsound[:10]:
            lines.append(f"    {o.question[:52]:54s} {o.detail[:40]}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("artifacts/coverage.json"))
    # Gates. Silent failures are the only hard zero: everything else is a
    # tradeoff, but answering an unanswerable question is never one.
    ap.add_argument("--min-coverage", type=float, default=0.0)
    ap.add_argument("--min-soundness", type=float, default=1.0)
    args = ap.parse_args()

    report = run(Engine.build())
    metrics = report.as_dict()

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(render(report))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2))

    failures = []
    if metrics["silent_failures"] > 0:
        failures.append(f"{metrics['silent_failures']} silent failure(s)")
    if metrics["soundness"] < args.min_soundness:
        failures.append(f"soundness {metrics['soundness']:.3f} < {args.min_soundness}")
    if metrics["coverage"] < args.min_coverage:
        failures.append(f"coverage {metrics['coverage']:.3f} < {args.min_coverage}")
    if failures:
        print("  FAILED: " + "; ".join(failures) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
