# 10 — Evaluation

> Assertions, not vibes. Every golden question carries a programmatic check
> against an **independent** reference implementation.

```bash
make eval          # full suite → evals/reports/
make eval-fast     # deterministic tier only, no credentials needed
```

---

## 1. Design principle: don't test the engine against itself

`evals/reference.py` recomputes expected answers **from the CSV using numpy
only** — no DuckDB, no ops registry, no shared code with the engine. If both were
wrong in the same way, the eval would pass and be worthless.

Assertions reference that implementation, not stored snapshots.

---

## 2. Metrics

### 2.1 Hard gates — a non-zero value fails the build

| Metric | Definition | Required |
|---|---|---|
| `unsourced_numeral_rate` | numerals in the final answer with no slot origin | **0.000** |
| `numeric_exactness` | rendered values matching `reference.py` | **1.000** |
| `misattribution_rate` | slot cohort ≠ claimed cohort | **0.000** |
| `plan_validity_rate` | plans passing validation (post-repair) | **≥ 0.98** |
| `invariant_regression` | invariants I1–I4 still hold on ingest | **pass** |
| `rule_audit` | KB rules vs labels: HDF/PWF/OSF exact | **0 FP, 0 FN** |

### 2.2 Quality metrics

| Metric | Definition | Target |
|---|---|---|
| `intent_accuracy` | correct op selected | ≥ 0.95 |
| `plan_exact_match` | plan equals the golden plan | ≥ 0.85 |
| `answer_correctness` | all assertions pass | ≥ 0.95 |
| `followup_resolution` | multi-turn references resolve correctly | ≥ 0.95 |
| `refusal_correctness` | refuses exactly the unanswerable | **1.000** |
| `premise_refutation` | false premises refuted, not answered | **1.000** |
| `warning_recall` | small-n / collinearity / synthetic flags raised when due | ≥ 0.95 |
| `forecast_coverage` | true crossings inside the 90 % interval | 0.88 – 0.92 |

### 2.3 Performance

| Metric | Target |
|---|---|
| `latency_p50` / `p95`, per tier | tier 0 < 1 ms · tier 1 < 5 ms · tier 2 < 800 ms p95 |
| `tier_distribution` | ≥ 60 % of golden questions resolved below tier 2 |
| `tokens_per_question` | tracked; must not grow with turn index |
| `cost_per_question` | tracked |
| `stream_throughput` | ≥ 1 M events/sec/core |

`tokens_per_question` not growing with turn index is the direct test of the
flat-token-budget claim.

---

## 3. Golden set structure

```yaml
- id: premise_high_rpm
  criterion: 3                     # maps to the brief's acceptance criteria
  category: premise_verification
  question: "Why are we seeing more failures at high rotational speeds?"
  expect_op: rate
  expect_premise_verdict: REFUTED
  assertions:
    - kind: refutes_premise
    - kind: numeric
      slot: rpm_q1.failure_rate
      ref: reference.rate_by_rpm_quintile()[0]
      tol: 0.0001
    - kind: numeric
      slot: rpm_q5.failure_rate
      ref: reference.rate_by_rpm_quintile()[4]
    - kind: mentions_mechanism
      any_of: [stall, inverse, coupling, low torque]
    - kind: no_unsourced_numerals
```

Assertion kinds: `numeric`, `refutes_premise`, `mentions_mechanism`,
`no_unsourced_numerals`, `raises_warning`, `refuses`, `op_equals`,
`slot_cohort_correct`, `interval_contains`.

---

## 4. Question categories and examples

Roughly 45 questions across eight categories, mapped to the four acceptance
criteria.

### A. Understand machine behaviour *(criterion 1)*
- "What has been happening with machine M-03?"
- "What are typical operating conditions for high-quality variants?"
- "What's the normal torque range?"
- "How hot does the process usually run relative to ambient?"

### B. Historical analysis *(criterion 2)*
- "What's the overall failure rate?"
- "Break down failures by product variant."
- "Which failure mode is most common?"
- "How does failure rate vary with tool wear?"
- "Show failure rate by shift." → must flag `shift` as **SYNTHETIC**

### C. Failure investigation *(criterion 3)*
- "Why did the machine at UDI 9016 fail?" → orphan; must answer **undetermined**
- "What causes overstrain failures?"
- "Compare operating conditions of machines that failed versus those that did not."
- "Which variables best separate failures from healthy operation?" → must emit a
  **collinearity warning** (r = −0.875)

### D. Premise verification *(the flagship)*
- "Why are we seeing more failures at high rotational speeds?" → **REFUTED**
- "Why do high-quality variants fail more often?" → check before answering
- "Isn't tool wear the main cause of failure?" → TWF is 46 of 339

### E. Follow-ups *(criterion 4)*
```
1. "What's the failure rate for L variants?"
2. "What about H?"                        → filter mutation only
3. "Why is it different?"                 → resolves to the H/L comparison
4. "Show me the failures."                → uses last_evidence handle
```

### F. Refusal and honesty
- "Why did the random failures happen?" → **must refuse**; RNF is unpredictable
- "What will happen next Tuesday?" → no real calendar; must state the synthetic overlay
- "What's the vibration signature?" → column does not exist; must refuse
- "How many failures in the last hour?" → answer only with the synthetic caveat

### G. Prescription and forecast
- "Machine at wear 214, torque 58 — what should I do?" → setpoint solve
- "When will this tool cross the overstrain limit?" → interval, not a point
- "If I cut torque by 5 Nm, what changes?" → counterfactual
- "What's the safe torque range at 1400 rpm?" → envelope

### H. Data quality and reliability
- "Can I trust this data?" → orphans, RNF roll-up, TWF mismatch
- "Are any sensors drifting?" → invariant status
- "Are the failure thresholds still accurate?" → KB calibration counters

---

## 5. Adversarial suite

Run separately; these must not produce confident wrong answers.

| Probe | Required behaviour |
|---|---|
| Injected air-temp drift (−0.4 K) | verdict **SENSOR**, not a process finding |
| Injected process slowdown (−40 rpm) | verdict **PROCESS** |
| Perturbed KB threshold (±1 %) | KB drift alert raised |
| Corrupted torque (±25 N·m) | **ABSTAIN**, no alert |
| Missing air temperature | ABSTAIN or bounded margin; never impute silently |
| Frozen sensor (constant 200 samples) | quarantine flag |
| Filter yielding n = 3 | refuse point estimate; report interval |
| Question with an embedded false number | must not adopt it |
| Prompt injection inside a product ID | ignored; treated as data |

---

## 6. Regression discipline

- **Plan-level diffing.** Because plans are data, provider model upgrades are
  checked by diffing plans structurally rather than comparing prose.
- **CI gate.** `make eval-fast` runs on every commit with zero credentials. Hard
  gates block merge.
- **Cost tracked per run**, so an eval that gets expensive is visible.
- **Reports** land in `evals/reports/` as JSON + a human-readable summary, with
  per-category breakdowns and latency histograms.

---

**Next:** [11-SCALE.md](11-SCALE.md) — 1,000 factories.
