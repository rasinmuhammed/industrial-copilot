# 13 — Build Plan

> Ordered so a **complete, defensible, evaluable system exists after Phase 4.**
> Everything after that is additive, never load-bearing.
>
> A complete good system beats a half-built exceptional one. This ordering is the
> main defense against over-engineering.

---

## Phase 1 — Physics warehouse and knowledge base

**Deliverable:** the data foundation and the SOP.

| File | Purpose |
|---|---|
| `copilot/config.py` | all tunables ✅ drafted |
| `copilot/knowledge/failure_modes.yaml` | documented rules + provenance ✅ drafted |
| `copilot/knowledge/semantic_layer.yaml` | the only permitted vocabulary ✅ drafted |
| `copilot/ingest.py` | CSV → DuckDB + physics + margins ✅ drafted |
| `scripts/verify_dataset.py` | reproduces every figure in doc 01 |

**Acceptance gate**
- `make verify` reproduces all of [01-DATASET.md](01-DATASET.md)
- Rule audit: HDF/PWF/OSF **0 FP, 0 FN**
- Invariants I1–I4 pass
- Build **fails** if the KB and the data disagree

---

## Phase 2 — Analysis IR and operator registry

**Deliverable:** the correctness foundation. Everything above this is presentation.

| File | Purpose |
|---|---|
| `copilot/ir.py` | plan schema + six-stage validation |
| `copilot/ops/*.py` | twelve operators |
| `copilot/evidence.py` | evidence bundle, slot IDs, provenance |
| `evals/reference.py` | **independent** numpy reference |
| `tests/` | per-op unit tests against the reference |

**Order within the phase:** `describe` → `rate` → `compare` → `root_cause` →
`trend` → `drivers` → `records` → `data_quality` → `counterfactual` → `envelope`
→ `forecast` → `sql_explore`.

**Acceptance gate**
- Every op matches `reference.py` exactly
- Invalid plans rejected at the correct validation stage
- Dimensional validation rejects cross-unit comparison
- Wilson CIs on every rate; small-*n* refusal works
- Collinearity warning fires on rpm/torque (r = −0.875)

---

## Phase 3 — Router, session, narrator, verifier

**Deliverable:** the latency and anti-hallucination machinery.

| File | Purpose |
|---|---|
| `copilot/planner/grammar.py` | deterministic NL → plan |
| `copilot/planner/cache.py` | normalised-key plan cache |
| `copilot/planner/llm.py` | Claude / Cerebras / Ollama adapters |
| `copilot/planner/router.py` | 3-tier dispatch + speculative execution |
| `copilot/session.py` | typed session state |
| `copilot/narrate.py` | slot-only narration |
| `copilot/verify.py` | PCN verifier, fail-closed |
| `copilot/engine.py` | orchestrator |

**Acceptance gate**
- `unsourced_numeral_rate` = **0.000**
- `misattribution_rate` = **0.000**
- Follow-up filter mutation resolves without re-planning
- Token count does **not** grow with turn index
- Works end-to-end with **no API key**

---

## Phase 4 — Evals and CLI ← **system is complete and defensible here**

| File | Purpose |
|---|---|
| `evals/golden.yaml` | ~45 questions, 8 categories |
| `evals/run_evals.py` | harness + reports |
| `copilot/cli.py` | terminal chat |
| `README.md` | the deliverable |

**Acceptance gate**
- All hard gates in [10-EVALS.md](10-EVALS.md) §2.1 pass
- Premise refutation on the high-rpm question: **1.000**
- Refusal correctness: **1.000**
- `make eval-fast` runs green with zero credentials

> **Stop here and ship if time runs short.** Phases 1–4 satisfy all four
> acceptance criteria plus latency, context engineering, and hallucination
> reduction.

---

## Phase 5 — Reliability

| File | Purpose |
|---|---|
| `copilot/reliability/invariants.py` | Gate 2 |
| `copilot/reliability/kb_monitor.py` | Gate 3 |
| `copilot/reliability/intervals.py` | interval margins → ABSTAIN |
| `evals/adversarial.yaml` | injected drift, corrupt sensors |

**Acceptance gate**
- Injected air drift → **SENSOR**; injected slowdown → **PROCESS**
- ±1 % KB perturbation raises a drift alert
- Corrupt torque → **ABSTAIN**, never an alert

---

## Phase 6 — Discovery

| File | Purpose |
|---|---|
| `discovery/dimensional.py` | unit-coherent feature construction |
| `discovery/threshold.py` | boundary estimation + CIs |
| `discovery/audit.py` | KB vs data reconciliation |
| `scripts/discover_rules.py` | the demo that answers the fair criticism |

**Acceptance gate**
- Recovers PWF to < 0.2 %, HDF speed to < 0.1 %, OSF (L) to < 0.1 %
- Reports honest uncertainty on H (n = 1,003)

---

## Phase 7 — Streaming

| File | Purpose |
|---|---|
| `copilot/stream.py` | replay, online scorer, alerts |
| `copilot/api.py` | FastAPI + SSE |
| `scripts/bench.py` | throughput benchmark |

**Acceptance gate**
- ≥ 1 M events/sec/core
- Alerts carry lead time with a calibrated interval
- Forecast coverage in 0.88–0.92

---

## Phase 8 — Interfaces

- **Evidence-card chat** — every answer expands to show the plan, the evidence
  table, and the margin arithmetic. *"Show your work"* as a UI primitive: the
  anti-hallucination thesis made visible rather than claimed.
- **Envelope Explorer** — true failure boundary in rpm × torque, draggable
  setpoint, live margins. A computed region, not a decision surface.
- **Fleet view** — streaming margin tiles, lead-time alerts, click-to-focus chat.

---

## Phase 9 — Planner fine-tuning *(optional bonus)*

Fine-tune **question → Analysis Plan**, never the answer. Narrow, mechanical,
machine-verifiable, with synthetically generable training data.

| Step | Detail |
|---|---|
| Data | Generate 5–10k question/plan pairs by templating the semantic layer + paraphrase |
| Train | Unsloth LoRA on a 7–8B base |
| Serve | Together / local |
| Measure | plan exact-match and end-to-end numeric accuracy, **against the same golden set**, with and without |

Target: planner tier from ~400 ms to < 100 ms, API removed from the hot path.

> Teaching a model *facts about machines* is the failure mode this project exists
> to eliminate. Teaching it *intent → plan* is the correct target.

---

## Sequencing rationale

```
Phase 1 ─► 2 ─► 3 ─► 4  ══ SHIPPABLE ══► 5 ─► 6 ─► 7 ─► 8 ─► 9
   data    truth  UX   proof              trust  scale  live  polish  bonus
```

Correctness before capability, capability before presentation, evidence before
polish. Phases 5–9 are independent of each other and can be reordered or dropped
by remaining time without weakening the submission.

---

## Definition of done

- [ ] `git clone && make demo` works with **zero credentials**
- [ ] README carries the architecture diagram, setup, config, and run instructions
- [ ] All hard eval gates green
- [ ] Every assumption stated in [12-ASSUMPTIONS.md](12-ASSUMPTIONS.md)
- [ ] Scale argument written and honest about what is unbuilt
- [ ] Every claim in the docs reproducible by a script in `scripts/`
