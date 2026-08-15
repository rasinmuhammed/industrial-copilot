# 02 — Why Language Models Fail on Time Series, and What We Do Instead

> **The model never sees the time series.**
>
> This is the single most important sentence in the architecture. Everything
> below is the justification and the mechanism.

---

## 1. The failure is structural, not a matter of model quality

A larger model does not fix any of the following. They are properties of how
language models represent and consume input.

### 1.1 Tokenization destroys numeric magnitude

A transformer has no continuous representation of a number. `1380` reaches the
model as one or more subword tokens; `1379` may decompose differently. Magnitude,
ordering, and distance must be *reconstructed* from symbol sequences the model
was never given a metric over.

Our HDF boundary sits at **1380 rpm**. Distinguishing 1379.4 from 1380.6 is the
entire decision. Asking a token predictor to do that reliably, across millions of
samples, is asking it to emulate a comparator it does not have.

### 1.2 Context windows cannot hold industrial data

| Scope | Values | Rough tokens |
|---|---:|---:|
| One machine, 5 sensors, 1 hour @ 1 Hz | 18,000 | ~40,000 |
| One machine, 5 sensors, 1 day @ 1 Hz | 432,000 | ~1,000,000 |
| One 2,000-machine site, 1 day | 864,000,000 | ~2 billion |
| **AI4I in full (10,000 × 8 numeric)** | **80,000** | **~180,000** |

Even the *toy* dataset for this assignment does not comfortably fit a prompt —
and one real machine-day exhausts a million-token window. Any design whose
answer to "analyse this sensor data" is "put it in the context window" does not
survive contact with a plant.

### 1.3 Attention encodes position, not time

Positional encodings give ordering. They do not give **duration**, **sampling
rate**, or **irregular intervals**. A gap of 1 s and a gap of 6 hours occupy the
same positional distance. Real industrial data is multi-rate, irregularly
sampled, and gap-ridden; serialising it into a token sequence discards exactly
the structure that makes it a time series.

### 1.4 Arithmetic is approximated, not computed

Aggregation, differencing, integration, and threshold comparison are exact
operations. A language model approximates them, and error compounds silently
across a chain. There is no reason to accept an approximation of `mean()`.

### 1.5 No dimensional awareness

Nothing in a token stream distinguishes 300 K from 300 °C from 300 rpm. Unit
mismatch across sites is a documented, deployment-killing class of bug, and the
model has no type system to catch it.

### 1.6 The empirical corroboration

Best-in-world text-to-SQL reaches **~72–76 %** execution accuracy on BIRD against
real databases, versus **92.96 %** for humans. Roughly one query in four is
wrong. For an engineer acting on a failure diagnosis, that is unusable — and
text-to-SQL is the *easier* task, because the database does the arithmetic.

---

## 2. What most systems do about it, and why it still fails

| Approach | Mechanism | Why it fails here |
|---|---|---|
| Dump rows into context | Serialise CSV into the prompt | Breaks at ~1 machine-hour; model still does arithmetic badly |
| RAG over time series | Embed windows, retrieve nearest | Embeddings of numeric windows are not semantically meaningful; retrieval ≠ computation |
| Text-to-SQL | Model writes SQL, DB computes | Fixes arithmetic, not correctness: ~25 % of queries wrong; unverifiable |
| Time-series foundation models | Pretrained forecaster (Chronos, TimesFM, Moirai) | Genuinely useful for forecasting, but they *forecast* — they do not answer "why", cannot prescribe, and are another opaque model to calibrate per site |
| Fine-tune on plant data | Bake behaviour into weights | Facts in weights cannot be updated, audited, or unit-checked. Worst option. |

Every one of these keeps the model **inside the numeric path**. That is the error.

---

## 3. Our answer: remove the model from the numeric path entirely

```
                    ┌──────────────────────────────────────────┐
   QUESTION ──────► │  LANGUAGE MODEL                          │
                    │  job: intent → validated Analysis Plan   │
                    │  sees: question, vocabulary, session     │
                    │  NEVER sees: a sensor value, a series,   │
                    │              a row, a raw aggregate      │
                    └───────────────────┬──────────────────────┘
                                        │  Analysis Plan (JSON)
                                        ▼
                    ┌──────────────────────────────────────────┐
   TIME SERIES ───► │  DETERMINISTIC OPERATORS                 │
   (never leaves    │  DuckDB / numpy. Exact. Typed. Unit-safe │
    the engine)     │  0.22 µs per sample                      │
                    └───────────────────┬──────────────────────┘
                                        │  Evidence bundle:
                                        │  ~10–50 scalars, each with
                                        │  unit, filter, row count
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │  LANGUAGE MODEL                          │
                    │  job: phrase the finding                 │
                    │  writes {{slot}} refs — NEVER digits     │
                    └──────────────────────────────────────────┘
```

The model is used **twice, for language only** — once to understand intent, once
to phrase a result. Between those two points sits arithmetic.

**Consequence:** our time-series competence is bounded by numpy and DuckDB, not by
a model's numeric reasoning. Swapping Haiku for a 7B local model changes phrasing
quality. It cannot change a number.

---

## 4. The margin is the interface between continuous reality and discrete language

This is the conceptual core.

A time series is continuous, unbounded, and high-volume. Language is discrete,
bounded, and low-volume. Something must bridge them, and **the bridge must be
lossless with respect to the decision**, not with respect to the data.

A margin does exactly that:

```
86,400 raw samples/day  ──►  min(margin) over the day  ──►  one scalar
```

That single scalar answers *"how close did we come to failing?"* — which is the
decision-relevant question — **exactly**, because `min` is associative and the
worst approach to a boundary is precisely what matters. Nothing decision-relevant
is lost.

Contrast: `mean(P(failure))` over the same day is not a probability of anything,
loses the spike that mattered, and cannot be inverted to an action.

> **Margins are a decision-preserving compression of a time series into a scalar
> a language model can safely handle.**

---

## 5. Every time-series capability, and where it is computed

None of these touch the language model.

| Capability | Question it answers | Operator | Method |
|---|---|---|---|
| Summary | "What are conditions on M-03?" | `describe` | DuckDB aggregate |
| Rate over time | "Failure rate by shift?" | `rate` | group-by + Wilson CI |
| Trend | "Is torque climbing?" | `trend` | windowed OLS slope + CI |
| Rate of change | "How fast is wear accruing?" | `trend` | first difference over grain |
| Seasonality | "Worse on nights?" | `rate` | group-by time grain |
| Changepoint | "When did behaviour shift?" | `trend` | CUSUM / binary segmentation |
| Anomaly | "Anything unusual?" | `root_cause` | invariant violation + margin sign |
| Cohort contrast | "Failed vs healthy?" | `compare` | Cohen's *d*, rate ratio, CI |
| Driver ranking | "What separates them?" | `drivers` | std. mean diff + collinearity check |
| Attribution | "Why did it fail?" | `root_cause` | rule evaluation + margin |
| **Forecast** | "When will it cross?" | `forecast` | **closed-form first-passage** |
| Counterfactual | "If torque drops 5 Nm?" | `counterfactual` | re-evaluate rules |
| Prescription | "What should I change?" | `envelope` | invert the constraint |
| Aggregation | "Worst point last quarter?" | rollup | `min` over tile tree |

### 5.1 Forecasting deserves special note

Tool wear accrues with known drift (2/3/5 min per cycle by variant); torque is
N(40, 10). Strain is therefore a Wiener process with drift, and the crossing time
is a **first-passage problem with a closed-form inverse-Gaussian solution**:

| wear | torque | margin | E[cycles to crossing] | 90 % interval |
|---:|---:|---:|---:|---|
| 150 | 45 | 4250 | 36.3 | 34.1 – 38.5 |
| 180 | 52 | 1640 | **12.1** | 11.0 – 13.2 |
| 200 | 48 | 2400 | 19.2 | 17.7 – 20.7 |
| 120 | 60 | 3800 | 24.4 | 23.0 – 25.7 |

**No training. No inference. No model artifact.** A full predictive distribution
from arithmetic — which is also the standard PHM formulation (Wiener degradation
→ IG remaining useful life). Conformal calibration on top supplies
distribution-free coverage guarantees on those intervals.

This is how we forecast failure *before it occurs* while remaining exact and
auditable.

---

## 6. What this buys, measured

| Property | Result |
|---|---|
| Per-sample margin evaluation | **0.22 µs** (4.6 M events/sec/core, scalar Python) |
| Vectorised | **419 M samples/sec/core** |
| Full stream scorer (robust track + alerting) | **~14 µs** → ~72 k/sec/core |
| Requirement, 2,000-machine site @ 1 Hz | 2,000 events/sec |
| **Headroom on one core — raw arithmetic** | **2,310×** |
| **Headroom on one core — full scorer** | **~36×** |
| Numeric error | **0** — arithmetic, not estimation |
| Context tokens for a 10-million-row question | **~400** (the evidence bundle) |

The speed is not the result of optimisation. **There is no model to run at
inference time.** The intelligence lives in the knowledge base, built offline;
online is arithmetic.

That inversion is the deepest idea in the project:

> Conventional systems place intelligence in **weights evaluated online**, so
> latency scales with intelligence and every site needs the artifact versioned
> and drift-monitored.
>
> We place intelligence in **rules discovered offline**, so inference cost is
> constant, edge deployment is trivial, and there is no artifact to drift.

It also answers the third documented cause of industrial-AI pilot failure —
*"inference latency across diverse manufacturing environments"* — by making
inference latency structurally irrelevant.

---

## 7. Honest limits

- **Not every mode has a closed form.** Bearing spectral signatures, cavitation,
  and lubrication breakdown will not reduce to `f(x) > θ`. Those require learned
  health indicators. The architecture survives — you apply the margin abstraction
  to a *learned* indicator instead of a derived one, which is standard PHM — but
  the exactness does not.
- **The rule language currently lacks temporal operators.** AI4I modes are
  per-sample predicates. Real modes need windowed aggregates, rate-of-change, and
  persistence ("true for N consecutive samples"). This is a well-understood
  extension (it is what CEP engines do) but it is an extension, not free. Spec'd
  in [11-SCALE.md](11-SCALE.md).
- **Time-series foundation models are complementary, not competing.** Where a
  degradation path has no analytic form, a pretrained forecaster is a reasonable
  source for the *health indicator*. It would sit in the offline discovery layer
  and feed the KB, never in the online answer path.
- **The forecast assumes stationary drift.** A step change in duty cycle
  invalidates the projection. The KB calibration monitor
  ([06-RELIABILITY.md](06-RELIABILITY.md)) is what catches that.

---

**Next:** [03-ARCHITECTURE.md](03-ARCHITECTURE.md) — components, data flow, and the
request lifecycle.
