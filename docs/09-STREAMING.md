# 09 — Real-Time Streaming Inference and Forecasting

> *"By the time you find the anomaly, the batch is already compromised."*
>
> Lead time is the product. This document is how we get it.

---

## 1. Throughput: why speed is free here

| Measurement | Result |
|---|---|
| Vectorised margin evaluation | **419 M samples/sec/core** |
| Scalar raw arithmetic, per event | **0.22 µs → 4.6 M events/sec/core** |
| `evaluate()` returning a margin object | **1.09 µs → 918 k/sec/core** |
| Full stream scorer (robust track + alerting) | **~14 µs** → ~72 k/sec/core |
| Requirement, 2,000-machine site @ 1 Hz | 2,000 events/sec |
| **Headroom on one core — raw arithmetic** | **2,310×** |
| **Headroom on one core — full scorer** | **~36×** |

This is not the result of optimisation. **There is no model to run at inference
time.** Intelligence lives in the knowledge base, built offline; online is
arithmetic.

> Conventional systems put intelligence in **weights evaluated online** — latency
> scales with intelligence, every site needs the artifact versioned and
> drift-monitored.
>
> We put intelligence in **rules discovered offline** — inference cost is
> constant, edge deployment is trivial, and there is no artifact to drift.

This directly answers the third documented cause of industrial-AI pilot failure
(*"inference latency across diverse manufacturing environments"*) by making
inference latency structurally irrelevant.

---

## 2. The pipeline

```
 replay / live source
     (configurable speed multiplier)
          │  cycle event
          ▼
 ┌──────────────────────────┐
 │ INGEST GUARD             │   range · staleness · rate-limit · clock
 │ → interval width         │   bad reading widens the interval
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ MARGIN EVALUATION        │   5 interval-valued margins
 │ stateless, 0.44 µs       │   DIAGNOSIS
 └──────┬───────────────┬───┘
        │               │
        ▼               ▼
 ┌─────────────┐  ┌──────────────────────┐
 │ RULE EVAL   │  │ TRAJECTORY FORECAST  │  ANALYSIS
 │ which mode  │  │ first-passage time   │
 └──────┬──────┘  └──────────┬───────────┘
        └───────┬────────────┘
                ▼
      ┌───────────────────────┐
      │ ALERT DECISION        │  three-state · debounce · persistence
      │ ALERT / SAFE / ABSTAIN│
      └──────────┬────────────┘
                 ▼
      ┌───────────────────────┐
      │ SETPOINT SOLVER       │  PRESCRIPTION
      └──────────┬────────────┘
                 ▼
             ┌───────┐
             │ GATE  │  confidence × delta-within-safe-band
             └───┬─┬─┘
        operator │ │ [PLC writeback — roadmap]
                 ▼ ▼
              SSE / UI
```

Everything left of the gate, plus the operator branch, ships in the prototype.
Writeback is architecture, not scope.

---

## 3. Forecasting: closed-form first passage

### 3.1 The degradation path is real

Verified in [01-DATASET.md](01-DATASET.md) §8: consecutive wear deltas are exactly
**2 / 3 / 5 min** (n = 5927 / 2963 / 990) matching documented L/M/H accrual, with
**119 tool resets** against the documentation's "tool replaced 120 times."

This is one tool wearing across grade-mixed production. There is a genuine
degradation trajectory, so forecasts can be validated against *observed* events.

### 3.2 The formulation

Strain = wear × torque. Wear accrues with known drift; torque is N(40, 10).
Strain is therefore a **Wiener process with drift**, and time-to-threshold is a
first-passage problem with a closed-form **inverse-Gaussian** solution — the
standard PHM degradation-to-RUL formulation.

```
margin      = θ − wear·torque
drift       = wear_rate × torque              [strain per cycle]
E[cycles]   = margin / drift
λ           = margin² / (wear_rate · σ_torque)²
sd[cycles]  = sqrt(E³ / λ)
```

| wear | torque | margin | **E[cycles]** | 90 % interval |
|---:|---:|---:|---:|---|
| 150 | 45 | 4250 | 36.3 | 34.1 – 38.5 |
| 180 | 52 | 1640 | **12.1** | 11.0 – 13.2 |
| 200 | 48 | 2400 | 19.2 | 17.7 – 20.7 |
| 120 | 60 | 3800 | 24.4 | 23.0 – 25.7 |

**No training. No inference. No model artifact.** A full predictive distribution
from arithmetic.

### 3.3 Calibration

Conformal prediction on the residuals supplies **distribution-free finite-sample
coverage** on those intervals: a stated 90 % interval contains the true crossing
90 % of the time, with a guarantee rather than an assumption. Measured as
`forecast_coverage` in the evals.

### 3.4 TWF is different, deliberately

TWF is stochastic (5.4 % per in-window observation). The forecaster returns a
**hazard**, never a crossing time:

> *"Tool enters its replacement window in 4 cycles. In-window failure probability
> is ≈5.4 % per cycle; ≈24 % over the remaining 5 cycles of typical window
> residency."*

Reporting a certainty here would be a lie. The system distinguishes deterministic
crossings from stochastic hazards in its wording, and the eval checks that it does.

---

## 4. Alert semantics

An alert carries **lead time**, not a severity label:

```
ALERT   M-03   OSF                          confidence: high
        margin  +1,640 min·N·m and falling
        crossing in 12.1 cycles  [11.0 – 13.2]   ≈ 24 min at current takt
        cause   wear 180 min × torque 52.0 N·m, L variant (limit 11,000)
        fix     reduce torque to ≤ 51.4 N·m  (Δ −0.6)  → margin +180
                or replace tool → margin resets to 11,000
        replay  plan#8f3c1a  kb@1.2.0  data@e91b
```

### 4.1 Firing conditions

| Trigger | Condition |
|---|---|
| Crossing | margin interval wholly negative, persisted N samples |
| Predicted crossing | E[crossing] < horizon **and** lower CI bound < horizon |
| Approach | margin < x % of threshold **with** negative slope |
| Instrument | invariant violation → SENSOR alert, not a machine alert |
| KB drift | calibration counters exceed control limits → KB alert |

### 4.2 Flood control

Debounce, persistence counts, per-asset rate limits, and ISA-18.2 prioritisation.
Suppressed alerts are still recorded — suppression is auditable.

### 4.3 Self-audit

Every alert is later reconciled against whether the crossing occurred and whether
anyone acted, producing closed-loop precision/recall **on our own alerting**. That
is the input to tuning the gate, and almost nobody instruments it.

---

## 5. Interfaces

| Surface | Purpose |
|---|---|
| `GET /stream/alerts` (SSE) | live alert feed |
| `GET /stream/margins?machine=` (SSE) | live margin telemetry |
| `POST /ask` | copilot Q&A |
| `GET /replay/{hash}` | re-execute a past answer |
| CLI `copilot stream` | terminal fleet monitor |
| Envelope Explorer | true failure boundary, draggable setpoint |
| Fleet view | margin tiles, lead-time alerts, click-to-focus chat |

The Envelope Explorer draws a **computed region**, not a classifier decision
surface. No approach based on a learned failure probability can draw it correctly
— which is why it is a proof of the thesis rather than decoration.

---

## 6. Honest limits

- Replay is not live OT. Real streams bring late arrival, out-of-order events,
  multi-rate sensors, and reconnect storms. Watermarking is specified but only
  exercised against synthetic disorder.
- The forecast assumes **stationary drift**. A step change in duty cycle
  invalidates it; Gate 3 is what catches that.
- Prescriptions are advisory. Closed-loop writeback needs process hazard analysis
  and interlock design well beyond this prototype.

---

**Next:** [10-EVALS.md](10-EVALS.md) — how all of this is measured.
