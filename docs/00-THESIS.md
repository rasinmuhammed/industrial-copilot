# 00 — Thesis

> **The copilot computes distance to the failure boundary. It never predicts the
> probability of failure, and it never authors a number.**

---

## 1. The problem as stated

Build an industrial copilot over the AI4I 2020 Predictive Maintenance Dataset.
An engineer asks questions in natural language; the system returns analysis that
is *accurate, explainable, and useful*.

Four acceptance criteria:

| # | Criterion | Where it is satisfied |
|---|---|---|
| 1 | Understand machine behaviour | `describe`, `records`, `envelope` ops → [05](06-RELIABILITY.md), [04](04-ANALYSIS-IR.md) |
| 2 | Analyse historical data conversationally | `rate`, `compare`, `trend`, `drivers` ops |
| 3 | Investigate failures | `root_cause`, `counterfactual`, `data_quality` ops |
| 4 | Follow-up questions | Typed session state → [06](07-CONTEXT.md) |

Three things the brief explicitly rewards — **latency, context engineering,
hallucination reduction** — and three it explicitly does not: using the biggest
LLM, building a pretty frontend, or stacking AI frameworks.

## 2. The observation that determines the whole design

The dataset documentation describes five failure modes. Four are stated as
explicit conditions over derived physical quantities. We transcribed each from
the prose and scored it against the published labels on all 10,000 rows:

| Mode | Rule | Fires | Labelled | FP | FN |
|------|------|------:|---------:|---:|---:|
| HDF | ΔT < 8.6 K **and** rpm < 1380 | 115 | 115 | **0** | **0** |
| PWF | torque·ω < 3500 W **or** > 9000 W | 95 | 95 | **0** | **0** |
| OSF | wear·torque > 11k/12k/13k (L/M/H) | 98 | 98 | **0** | **0** |
| TWF | wear ∈ [200,240] → random draw | 790 | 46 | — | 3 |
| RNF | 0.1 % background | — | 19 | — | — |

**287 of 339 failures are exactly explainable by arithmetic.**

This is not a statistical learning problem dressed as one. The ground truth is a
formula. Any system that answers *"why did this fail?"* with a gradient-boosted
probability converts a **known quantity into an estimate** — and then cannot act
on it. That is the default submission for this assignment and it is the thing to
beat.

Full analysis and every verified figure: **[01-DATASET.md](01-DATASET.md)**.

## 3. Why "margin" and not "probability"

A margin is a signed scalar in native engineering units — `−1,433 min·Nm past
the overstrain limit`, `+412 W below the overload ceiling`. Three properties
follow, and they are the entire architecture:

### 3.1 It inverts → the system can prescribe

The constraint is analytic, so you solve it for the control variable. Not
*"73 % risk"* but *"reduce torque by 4.2 Nm and every margin returns positive."*
A classifier can only score a state you hand it; it cannot produce one.

### 3.2 It is continuous → the system can forecast

A binary label carries no information until it flips. A margin trending toward
zero yields a crossing time. Measured on this dataset:

- **811 healthy rows (8.1 %) sit within 2 % of a boundary**
- **2,508 (25.1 %) within 5 %**
- There are only **339 failures** in the entire dataset

The failure label sees none of those near-misses. The margin sees all of them.
That is a **7.4× larger early-warning surface than the total failure count**,
available with no model at all.

### 3.3 It aggregates → the system scales

`min()` over a window is the worst approach to the boundary, and `min` is
associative. Margins therefore roll up losslessly through a tile tree
(1 s → 1 min → 1 h → 1 day). *"How close did this machine come to overload last
quarter?"* is answered from a daily tile without touching a raw row.

Mean probability over a window is not a probability. It loses the event that
mattered by the second level of aggregation, which forces you to keep raw data
hot forever. This is the difference between an architecture that reaches 1,000
factories and one that does not. See **[11-SCALE.md](11-SCALE.md)**.

## 4. The language-model incapability we exist to solve

Language models are structurally bad at time-series data: tokenization destroys
numeric magnitude, context windows cannot hold a sensor stream, and arithmetic is
approximated rather than computed.

**Our answer is structural, not mitigative: the model never sees the time series.**

The model's only job is `intent → validated plan`. Every numeric operation is
performed by deterministic operators. This bounds our time-series competence by
numpy and DuckDB, not by a model's numeric reasoning.

This argument is central enough to have its own document:
**[02-TIME-SERIES.md](02-TIME-SERIES.md)**.

## 5. Four structural guarantees

| # | Guarantee | Mechanism |
|---|-----------|-----------|
| 1 | The model cannot name a column that does not exist | Plans validated against the semantic layer before execution |
| 2 | The model cannot author a number | Narrator emits slot IDs only; any bare digit is rejected (Proof-Carrying Numbers) |
| 3 | A bad sensor reading cannot raise a false alert | Interval-valued margins; ambiguity yields **ABSTAIN** |
| 4 | A stale rule cannot stay silently wrong | KB calibration monitor, directional and model-free |

Guarantees 1–2: **[05-GROUNDING.md](05-GROUNDING.md)**.
Guarantees 3–4: **[06-RELIABILITY.md](06-RELIABILITY.md)**.

## 6. What the industry actually gets wrong

**80 % of industrial AI projects never leave pilot**, at roughly $2.3 M wasted per
failed scaling attempt. The documented cause is explicit:

> "The scaling barrier emerges **not from algorithm performance** but from
> inability to consistently replicate training data quality, feature
> availability, and inference latency across diverse manufacturing
> environments."

Organisations in pilot purgatory *"typically misattribute scaling failures to
algorithm accuracy."*

Every hour spent on model accuracy is spent on the 20 % that is not the problem.
This architecture targets the documented 80 %:

| Documented failure cause | Our mechanism |
|---|---|
| Training-data quality not replicable | No training. Rules + margins are computed, not fitted. |
| Feature availability varies by site | Features are derived from units, which every site has. |
| Inference latency across environments | 0.22 µs/sample. No model runs at inference time. |
| Integration architecture gaps | Analysis IR recompiles per backend; no prompt changes. |
| Operators stop trusting alerts | ABSTAIN on bad data; every number traceable. |

## 7. Intellectual honesty

Most of this is **not novel, and that is the point.** Each component is
established practice from a field that already solved it:

| Component | Precedent |
|---|---|
| Rules as diagnostic core | Consistency-based / model-based diagnosis (Reiter 1987; de Kleer & Williams 1987) |
| Margin as health indicator | Standard PHM construct |
| Time-to-crossing | RUL as first-passage time to a failure threshold |
| Semantic layer over raw schema | Snowflake Cortex Analyst: 57 % → 78 % on BIRD |
| Slot-bound numeric verification | Proof-Carrying Numbers, arXiv:2509.06902 |
| Knowledge-mediated industrial agent | Cognite Atlas AI; Siemens Senseye Copilot |

**The contribution is the composition**, plus three pieces for which we could not
find prior art: the KB calibration monitor, invariant-based sensor/process
discrimination, and the four-gate epistemic perimeter.

### Where this breaks

- Our exactness is partly an artifact of the dataset documenting its own rules.
  Mitigated — not eliminated — by the discovery pipeline in
  **[08-DISCOVERY.md](08-DISCOVERY.md)**, which re-derives those rules from data
  alone to within 0.01–3.4 %.
- Real failure modes with spectral signatures have no closed form. Those stay
  ML-shaped; the margin abstraction then applies to a *learned* health indicator.
- Drift defenses are validated against *injected* drift on synthetic data. Field
  validation is the missing step.

Full list: **[12-ASSUMPTIONS.md](12-ASSUMPTIONS.md)**.

---

**Next:** [01-DATASET.md](01-DATASET.md) — every verified figure and its derivation.
