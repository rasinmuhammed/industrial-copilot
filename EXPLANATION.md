# Argus - Deep Technical Explanation

> Use this to understand every component well enough to explain it in real time,
> answer follow-up questions, and defend every design decision.

---

## 1. The Core Thesis - Why This Is Different

### The standard approach (and why it fails)

Most predictive maintenance systems train a classifier (gradient boost, LSTM, transformer) on sensor data and output a probability: *"this machine has a 73% chance of failing in the next 24 hours."*

That approach has three deep problems:

**Problem 1 - A probability cannot prescribe.** If the model says 73%, what does an engineer do? Reduce torque? By how much? Change speed? Nothing in the output tells you. The diagnosis is: "something is wrong." That's not actionable.

**Problem 2 - A probability cannot scale.** You cannot aggregate probabilities across machines. The mean of 0.02, 0.71, 0.01 is 0.25 - and the 0.71 spike that represents a real near-miss is gone, averaged away. You would have to re-query every machine raw to find it.

**Problem 3 - Probabilities cannot tell sensor failure from process failure.** A drifting thermocouple and a genuinely hotter process produce the same symptom: the alert count moves. A probability model reports both identically. They demand opposite responses.

### The alternative: compute distance to the boundary

The AI4I failure modes are not probabilistic patterns - they are published physical rules:

| Mode | Rule | Type |
|------|------|------|
| **HDF** (Heat Dissipation) | ΔT < 8.6 K **AND** rpm < 1380 | Conjunctive |
| **PWF** (Power) | torque × ω < 3500 W **OR** > 9000 W | Disjunctive |
| **OSF** (Overstrain) | wear × torque > 11,000/12,000/13,000 min·N·m | Threshold |

These rules score perfectly against 10,000 rows: **0 false positives, 0 false negatives** for 287 of 339 failures.

Instead of asking "will it fail?" we ask: **"how far is this machine from the boundary at which it would fail?"**

That signed distance is called a **margin**. For example: `−1,433 min·N·m past the overstrain limit`.

Three properties of margins that probabilities don't have:

| Property | Margin | Probability |
|----------|--------|-------------|
| **Invertible** | Yes - algebra tells you how far to move to return positive | No |
| **Composable** | Yes - `min()` is associative, tiles merge losslessly | No - mean() discards events |
| **Separable** | Yes - physics separates sensor drift from process change | No |

---

## 2. Architecture - Layer by Layer

```
┌─────────────────────────────────────────────────────────────────────┐
│ L1  PLANT          AI4I CSV (simulated PLC replay at 1 Hz)          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ raw samples (air_temp, rpm, torque, …)
┌──────────────────────────────▼──────────────────────────────────────┐
│ L2  INGEST + NORMALISE                                               │
│     copilot/ingest.py                                                │
│     · dimensional typing - every column has a physical unit          │
│     · derived physics: power = torque × (rpm × 2π/60)               │
│                         overstrain = wear × torque                   │
│                         temp_delta = process_temp − air_temp         │
│     · margin computation - 5 signed scalars per row, 0.22 µs        │
│     · invariant evaluation - physics laws that must always hold      │
│     · loads into DuckDB (data/warehouse.duckdb)                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ typed columns + margins
┌──────────────────────────────▼──────────────────────────────────────┐
│ L3  REASONING CORE                                                   │
│     copilot/engine.py  ← the orchestrator                            │
│                                                                      │
│  Question ──► ROUTER (4 tiers) ──► Analysis Plan (JSON IR)          │
│                                          │                           │
│                                          ▼                           │
│                              EXECUTOR (copilot/ops/)                 │
│                              typed operator over DuckDB              │
│                                          │                           │
│                                          ▼                           │
│                              Evidence Bundle (slots + units + n)     │
│                                          │                           │
│                                          ▼                           │
│                              NARRATOR (prose with {{slot}} refs)     │
│                                          │                           │
│                                          ▼                           │
│                              PCN VERIFIER (fail-closed)              │
│                                          │                           │
│                                          ▼                           │
│                              Verified Answer                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ verified prose + evidence
┌──────────────────────────────▼──────────────────────────────────────┐
│ L4  ACTION                                                           │
│     Web UI (ask.html, fleet.html, explorer.html, reliability.html)  │
│     REST API (FastAPI on port 8000)                                  │
│     CLI (copilot.cli)                                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Four-Tier Planning Router

This is the most important architectural decision. When a question arrives, the system routes it through four tiers cheapest-first:

### Tier 1 - Plan Cache (~0 ms)
A question's intent is hashed. If the same question shape has been answered before, the validated plan is returned immediately - no computation, no model call. Plans cache *across sessions* because they are parameterised by filters, not by the specific values.

> "What's the failure rate for L variants?" and "What's the failure rate for M variants?" are the same plan with different filter values. One cache entry serves both.

### Tier 2 - Grammar Planner (~1 ms)
`copilot/planner/grammar.py` uses ~50 regex patterns to classify intent and directly construct a validated `AnalysisPlan` object. No model, no network call.

Covers: failure rate, root cause, comparison, trend, describe, counterfactual, data quality, records, drivers, premise verification.

**92% of all questions in the eval suite hit this tier.** This is why latency is 7 ms median.

### Tier 3 - Verified Exemplars (~1 ms)
A library of hand-curated (question, plan) pairs. The incoming question is embedded, the nearest exemplar is retrieved, and its plan is validated and adapted. This is the *distillation* path - a weak model that has been fine-tuned to produce structured plans can feed verified exemplars here, making the system progressively smarter without changing a line of orchestration code.

### Tier 4 - LLM Planner (~400 ms)
Falls back to a language model (Anthropic Claude, Cerebras, Groq, Ollama - all configurable). The model is given:
- A system prompt describing every available metric and dimension
- A JSON schema for constrained decoding (the model cannot produce structurally invalid output)
- The question + session state

**The model is never asked for a number.** It is asked for a plan. Numbers come from DuckDB.

---

## 4. The Analysis IR (Intermediate Representation)

Every plan is a typed Python object (`copilot/ir.py :: AnalysisPlan`):

```python
class AnalysisPlan(BaseModel):
    op: OpName              # describe | rate | compare | trend | root_cause | …
    metrics: list[str]      # ["torque_nm", "rotational_speed_rpm"]
    filters: list[Filter]   # [Filter(field="product_type", op="=", value="L")]
    group_by: list[str]     # ["product_type"]
    bin: BinSpec | None     # for trend: {field, method, bins}
    cohorts: list[Cohort]   # for compare: [{name, filters}, {name, filters}]
    verify_premise: bool    # True → check the question's claim before answering
    …
```

This IR is language-model-agnostic. When the backend changes from DuckDB to ClickHouse to Flink SQL, the prompts, evals, and orchestration don't change - only the compiler does.

> **This is why text-to-SQL copilots die in production.** SQL is dialect-specific. Every time you switch database, you re-tune every prompt for the new SQL syntax.

---

## 5. The Four Gates

### Gate 1 - Is the premise true?
Many questions contain embedded claims: *"Why are there MORE failures at high speed?"*

Before answering *why*, the system tests *whether*. It runs the rate op over speed quintiles. The result:

```
1168–1405 rpm:  12.2% failure rate
1644–2886 rpm:   2.24% failure rate
```

The premise is false. The answer leads with that, not with an explanation of the (non-existent) phenomenon. This is **premise refutation** - the most distinctive feature and the one that directly addresses hallucination.

A correlation-based system would produce a confident causal story supporting the false premise. This system says: *"You aren't. The relationship is U-shaped. Failures are 5.4× more common at low speed."*

### Gate 2 - Is the instrument telling the truth?
A 0.4 K thermocouple drift causes heat-dissipation alerts to drop 54% - a conventional copilot would report "failures down 54% - good month" while the plant goes blind.

The discriminator is physics invariants that must hold at any operating point:
- `I1`: Temperature differential is always positive (process > ambient)
- `I2a`: Mean temperature ratio ≈ 10.0 K (process offset is structural, not volatile)
- `I2b`: Std ratio ≈ 1.0 (noise is symmetric)
- `I3`: rpm-torque correlation ≈ −0.875 (coupled at the design point)
- `I4`: Power centred at ~6,280 W (design operating point)

A sensor drift shifts one invariant while leaving the others in place. A real process change shifts them in a physically consistent pattern. The system can separate them with z-scores.

### Gate 3 - Is the rule still valid?
The dangerous drift is not the sensor - it's a rule that silently becomes wrong. A supplier changes tool material and the real overstrain limit moves; every margin is verifiably computed, and *wrong*.

Two counters catch it with no model and no retraining:
- **Surprise failures**: failure at positive margin → threshold too loose
- **False alarms**: alert with no failure → threshold too tight

At 0% perturbation: 0 surprise failures, 0 false alarms. The monitoring signal is zero only at the true threshold, rising in both directions - visible in the Reliability page chart.

### Gate 4 - Is every number sourced?
The narrator (language model or deterministic template) is instructed to write only `{{slot_id}}` references - never digits. For example:

```
narrator output:  "Torque averages {{failed.torque_nm.mean}} in failed cycles…"
                                       ↑ slot reference, not a number
after rendering:  "Torque averages 50.17 N·m in failed cycles…"
                                       ↑ value pulled from evidence bundle
```

The **PCN verifier** (`copilot/engine.py :: verify`) scans the rendered prose for bare numerals using a regex. Any bare numeral that cannot be traced to a slot causes the answer to be rejected and re-narrated - or refused entirely.

**This is fail-closed.** A weak model produces worse prose, not fabricated numbers.

---

## 6. Context Engineering (Session State)

`copilot/session.py :: SessionState` tracks:
- **Focus** - a single cycle (`cycle 9016`) or range in scope
- **Filters** - active dimension filters (`L variant`, `failed cycles`)
- **Metrics seen** - what the conversation has been about
- **Last plan** - previous operation, for follow-up resolution

When a follow-up arrives ("What about M?"), the resolver carries forward the `rate` op and previous filters, swaps the variant filter, and produces a new valid plan - no model call needed.

**The context block is bounded.** It fits in ~100 tokens regardless of conversation length. This keeps the LLM tier cost flat across long conversations.

---

## 7. The Knowledge Base

`data/knowledge.yaml` - a structured, versioned store of:
- **Failure rules** with physical derivation and evidence
- **Thresholds** with confidence intervals and support counts
- **Invariants** with expected ranges and tolerance
- **Collinearity warnings** (torque/rpm/power are strongly coupled)

Every KB entry has: `author`, `version`, `provenance`, `confirmed_at`. This is how the system scales from one factory to 1,000 - a threshold confirmed at one site becomes a prior elsewhere, adapted to local duty cycle.

---

## 8. Scaling to 1,000 Factories

The key insight: **margins compose losslessly, probabilities do not.**

`min()` is associative. The worst margin over a time window is the `min` of the worst margins of its sub-windows. This means tiles merge without loss:

```
raw → 1s tile → 1min tile → 1h tile → 1day tile → query answered from tile
```

A probability spike (0.71) averaged into a window becomes 0.25 - the event is gone forever. A margin minimum (−58 W past the stall floor) propagates intact.

For 1,000 factories × 2,000 machines × 1 Hz = **2 million events/second**:

| Component | What scales |
|-----------|-------------|
| Margin evaluation | Linear in samples, **negligible** - measured at 444 M/sec/core |
| LLM inference | **Sublinear** - same plan serves all tenants; cost driven by question diversity, not fleet size |
| KB maintenance | Linear in **asset classes**, not assets - a rule confirmed once is inherited |

The entire fleet of 2 million machines needs **0.8 CPU cores** for margin evaluation.

---

## 9. What's Unbuilt (and Why That's Honest)

Claude Code's remaining items are real gaps for a **production system** - not for a **prototype assessment**:

### CMMS ingest and outcome feedback
A CMMS (Computerised Maintenance Management System) closes the loop: when a technician actually fixes a machine, that work order feeds back to confirm or deny the copilot's diagnosis. Without a real CMMS source, faking one would teach the system nothing - the feedback would be circular. **This is the right call.** Faking data to pretend the loop is closed would be worse than leaving it open.

### The retrained SLM
The training notebook exists. It emits JSON training data (question → constrained plan), and the decoding is wired for when weights arrive. The current grammar tier covers 92% of questions - the SLM is the path to covering the remaining 8% without an expensive frontier model call. It needs a GPU run to produce weights. **An assessment doesn't need to run GPU training.**

### Trajectory learning
Remaining useful life (RUL) prediction requires a dataset with run-to-failure segments - C-MAPSS (NASA) or FEMTO bearings. AI4I has no segment longer than 16 cycles; you cannot learn degradation trajectories from it. **This is intellectual honesty.** Claiming RUL capability on a dataset that doesn't support it would be a red flag.

The honest framing: the architecture is designed so these three pieces plug in - the IR, the KB, and the verifier remain unchanged. They are **understood work, not research**.

---

## 10. How to Explain It in an Interview

### Opening (30 seconds)
> "Most predictive maintenance systems output a probability of failure. Probabilities can't tell you what to do about it, they can't be aggregated across machines without losing information, and they can't separate a sensor fault from a real process change. We took a different approach: compute the signed distance to the physical failure boundary. That number is actionable, composable, and separable."

### On hallucinations
> "Every number in every answer comes from a SQL query over DuckDB - never from the language model. The model writes slot references in double braces. A verifier checks that no bare numeral appears in the output. If one does, the answer is rejected. The system is fail-closed: a weaker model produces worse prose, not fabricated numbers."

### On latency
> "92% of questions hit the grammar tier - a deterministic regex matcher - and answer in under 10 milliseconds. The LLM tier is only reached for open-ended questions the grammar doesn't cover. Plans are cached across sessions, so the same question shape is never sent to a model twice."

### On premise refutation (the flagship)
> "The brief's own example question - 'Why are we seeing more failures at high rotational speeds?' - is factually wrong. Failures are 5.4× more common at low speed. A correlation-based system would confabulate a confident causal story. We test the premise before answering it. The answer is: 'You aren't.' That's the most important thing a copilot can do."

### On scale
> "The architecture scales because margins compose. min() is associative, so a tile tree can pre-aggregate across time without losing information. 2 million events per second - 1,000 factories at 1 Hz - needs 0.8 CPU cores for margin evaluation. LLM cost is driven by the diversity of question *shapes*, not by fleet size. The KB scales by asset class, not by individual asset."

### On what's missing
> "There are three things not built: CMMS feedback, a retrained SLM, and trajectory RUL on a proper dataset. All three are understood work. Faking them would compromise the intellectual honesty of the system. The architecture is designed so they plug in without changing the reasoning core."

---

## 11. Deployment

### The architecture as deployed

The whole system is a **single FastAPI process** that serves:
- The REST API (`/ask`, `/fleet`, `/health`, etc.)
- All static HTML/CSS/JS from `copilot/static/`
- A DuckDB file (`data/warehouse.duckdb`) stored on disk

There is **no separate frontend service**. Vercel is not applicable - the HTML files are not a Next.js/React app.

### Railway ($5/month) - Host everything here

Railway Starter ($5/month) gives you:
- 512 MB RAM, shared vCPU, 1 GB disk
- Sufficient for this prototype (DuckDB file is ~3 MB, process uses ~150 MB RAM)

**Deployment steps:**

1. Create `railway.toml` in project root:
```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "make build && uvicorn copilot.api:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 60
restartPolicyType = "on_failure"
```

2. Create `Procfile` as fallback:
```
web: make build && uvicorn copilot.api:app --host 0.0.0.0 --port $PORT
```

3. Make sure `data/ai4i2020.csv` is committed (it is - it's the source data).

4. The `make build` step runs `python -m copilot.ingest` which produces `data/warehouse.duckdb`.

**Environment variables to set in Railway:**
```
COPILOT_PROVIDER=deterministic   # no LLM needed for the eval suite
# Optional, for LLM tier:
COPILOT_PROVIDER=cerebras
CEREBRAS_API_KEY=your_key
```

**Limitations on Railway Starter:**
- No persistent disk across redeploys by default - add a Railway Volume for `data/` if you want the DuckDB to persist across deploys (otherwise it rebuilds from CSV each boot - takes ~3 seconds)
- 512 MB RAM is fine; DuckDB is embedded, not a server

### Vercel - Not needed for this project

Vercel hosts Node.js/static sites. Your frontend is HTML served by FastAPI. Skip it.

If you later want to build a proper React frontend (separate from the FastAPI server), then Vercel would be the right home for that. For this assessment, deploy everything to Railway.

### What the evaluators will see at your Railway URL

```
https://your-app.railway.app/          → Ask console (copilot)
https://your-app.railway.app/fleet/view → Fleet control room
https://your-app.railway.app/explorer  → Envelope explorer
https://your-app.railway.app/reliability → Gates 2 & 3
https://your-app.railway.app/health    → System health JSON
https://your-app.railway.app/docs      → FastAPI auto-generated API docs
```

---

## 12. Files Worth Knowing

| File | What it does |
|------|-------------|
| `copilot/engine.py` | Orchestrates everything: Router → Execute → Narrate → Verify |
| `copilot/planner/grammar.py` | 50 regex patterns → deterministic plan construction |
| `copilot/planner/router.py` | Tier selection: cache → grammar → exemplars → LLM |
| `copilot/planner/llm.py` | LLM providers + prompt construction (never digits) |
| `copilot/ir.py` | `AnalysisPlan` - the typed IR that all tiers produce |
| `copilot/ops/` | One file per operator: `rate.py`, `compare.py`, `root_cause.py`, … |
| `copilot/evidence.py` | `EvidenceBundle` - slots, units, provenance, row count |
| `copilot/session.py` | Bounded context: focus, filters, metrics_seen, last_plan |
| `copilot/physics.py` | Margin computation - the 5 signed scalars |
| `copilot/knowledge.py` | KB loader - failure rules, thresholds, invariants |
| `copilot/reliability.py` | Gates 2 & 3: drift diagnosis, KB calibration |
| `evals/golden.yaml` | 25 golden questions with numeric assertion contracts |
| `evals/reference.py` | Independent reference implementation - shares no code with engine |
| `docs/00-THESIS.md` | The argument: why margins beat probabilities |
| `docs/11-SCALE.md` | 1,000-factory architecture with cost breakdown |
