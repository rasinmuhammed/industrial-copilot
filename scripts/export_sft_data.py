#!/usr/bin/env python3
"""Export verified (question, plan) pairs for SLM fine-tuning.

Output format: OpenAI JSONL — one JSON object per line, each a full
chat-format conversation.  Compatible with Unsloth, Axolotl, LLaMA-Factory,
and Hugging Face TRL SFTTrainer.

Every row in the output:
  - Was asked of the real engine
  - Went through parse_plan validation
  - Produced an answer that survived the PCN verifier
  - Has its plan serialised with Pydantic model_dump_json

Unverified plans are DISCARDED.  The fine-tuning signal is only from
questions whose full pipeline succeeded — planning, execution, and
verification.  A plan that produced an unverified answer teaches the SLM
nothing useful and may teach it something wrong.

Usage
-----
    python scripts/export_sft_data.py                   # write data/sft_train.jsonl
    python scripts/export_sft_data.py --stdout          # print to stdout
    python scripts/export_sft_data.py --augment 5       # augment each example N times

Fine-tuning recipe (Kaggle / Colab T4)
---------------------------------------
    # Install
    pip install unsloth trl datasets

    # Load data
    from datasets import load_dataset
    ds = load_dataset("json", data_files="sft_train.jsonl", split="train")

    # Fine-tune Qwen3-1.7B-Instruct (fits on 1x T4)
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Qwen3-1.7B-Instruct-bnb-4bit", max_seq_length=512
    )
    # ... see notebooks/planner_distillation.ipynb for the full recipe

After training
--------------
    # Export to GGUF and push to HF Hub
    # Pull with Ollama:
    ollama pull hf.co/your-name/margin-planner-gguf

    # Enable in the copilot:
    COPILOT_PROVIDER=slm COPILOT_SLM_MODEL=margin-planner make serve
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.engine import Engine                             # noqa: E402
from copilot.planner.constrained import system_prompt        # noqa: E402
from copilot.session import SessionState                     # noqa: E402

GOLDEN  = Path(__file__).resolve().parent.parent / "evals" / "golden.yaml"
OUT_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "sft_train.jsonl"

# ---------------------------------------------------------------------------
# Extra phrasings — surface-form diversity beyond the golden set.
# These are still executed for real; nothing is assumed or fabricated.
# ---------------------------------------------------------------------------
EXTRA_QUESTIONS = [
    # Rate / breakdown
    "What percentage of cycles result in failure?",
    "Show me the failure rate for each product type.",
    "How many L-variant cycles failed?",
    "What's the overall failure rate?",
    "Break down failures by quality variant.",
    # Describe
    "What does a normal operating cycle look like?",
    "Describe the typical torque and speed.",
    "What's the usual temperature differential?",
    "Summarise tool wear across the dataset.",
    # Root cause
    "What are the main failure modes?",
    "Which failure mode fires most often?",
    "Why do failures cluster at low rotational speeds?",
    "What causes heat dissipation failures?",
    "Attribute cycle 51 to a failure mode.",
    "Why did cycle 9016 fail?",
    # Compare
    "How do failed and healthy cycles differ in torque?",
    "Compare operating conditions between failed and non-failed cycles.",
    "What separates the H-variant cycles that failed from those that didn't?",
    # Trend
    "How does failure rate change with tool wear?",
    "Show failure rate as tool wear increases.",
    "How does failure rate vary across rotational speed?",
    # Drivers
    "Which variable best predicts failure?",
    "What drives failures in this dataset?",
    "What are the biggest predictors of machine failure?",
    # Counterfactual
    "What if torque increased by 10 Nm?",
    "What happens if we reduce rotational speed by 200 rpm?",
    "Simulate reducing tool wear by replacing the tool at 150 min.",
    # Data quality
    "Can I trust the labels?",
    "Are there any anomalies in the dataset?",
    "Is the data reliable?",
    # Records
    "Show me cycles closest to failing.",
    "List the 10 most dangerous cycles.",
    "Give me examples of overstrain events.",
    # Premise verification
    "Why are H variants failing more than L?",
    "Are high-torque cycles more likely to fail?",
    "Is tool wear the biggest driver of failure?",
]

AUGMENTATIONS = [
    "Can you tell me {q}",
    "I need to understand {q}",
    "Please help with: {q}",
    "{q} — can you look into this?",
    "Quick question: {q}",
]


def _augment(question: str, n: int, rng: random.Random) -> list[str]:
    templates = rng.sample(AUGMENTATIONS, min(n, len(AUGMENTATIONS)))
    q_lower = question[0].lower() + question[1:]
    return [t.format(q=q_lower) for t in templates]


def _to_sft_row(question: str, plan_json: str, sys_prompt: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": sys_prompt},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": plan_json},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stdout",  action="store_true", help="print to stdout instead of file")
    parser.add_argument("--augment", type=int, default=2, metavar="N",
                        help="augment each example with N paraphrases (default: 2)")
    parser.add_argument("--output",  default=str(OUT_DEFAULT), metavar="PATH")
    args = parser.parse_args()

    import yaml
    with open(GOLDEN) as f:
        golden_cases = yaml.safe_load(f)["questions"]

    golden_questions = [c["question"] for c in golden_cases if "question" in c]
    all_questions = golden_questions + EXTRA_QUESTIONS

    engine   = Engine.build()
    sys_p    = system_prompt()
    rng      = random.Random(42)
    rows     = []
    skipped  = 0

    for q in all_questions:
        state  = SessionState()
        answer = engine.ask(q, state)
        if not answer.verified or answer.plan is None or answer.refused:
            skipped += 1
            continue

        plan_json = answer.plan.model_dump_json(exclude_none=True)
        rows.append(_to_sft_row(q, plan_json, sys_p))

        # Surface-form augmentation
        for variant in _augment(q, args.augment, rng):
            a2 = engine.ask(variant, SessionState())
            if a2.verified and a2.plan is not None and not a2.refused:
                rows.append(_to_sft_row(variant, a2.plan.model_dump_json(exclude_none=True), sys_p))

    print(f"Generated {len(rows)} SFT rows  ({skipped} skipped — unverified or refused)",
          file=sys.stderr)

    out_lines = [json.dumps(r, ensure_ascii=False) for r in rows]

    if args.stdout:
        print("\n".join(out_lines))
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"Written {len(rows)} rows → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
