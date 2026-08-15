# 06 — Reliability: Messy Sensors, Drift, and Stale Knowledge

> **80 % of industrial AI projects never leave pilot**, and the documented cause
> is *not* algorithm performance. This document is where the project spends its
> effort accordingly.

---

## 1. What actually kills these systems

> "The scaling barrier emerges **not from algorithm performance** but from
> inability to consistently replicate training data quality, feature
> availability, and inference latency across diverse manufacturing
> environments."

Organisations in pilot purgatory *"typically misattribute scaling failures to
algorithm accuracy."*

Documented sensor error taxonomy, **in decreasing order of frequency**, mapped to
mechanism:

| Error type | Frequency rank | Defense | Output |
|---|---|---|---|
| **Outliers** | 1 | invariant bounds + interval widening | ABSTAIN |
| **Missing data** | 2 | interval bounded by physical rate limits | ABSTAIN / bounded |
| **Bias** | 3 | invariant monitor | SENSOR verdict |
| **Drift** | 4 | invariant monitor + KB calibration | SENSOR vs PROCESS |
| **Noise** | 5 | interval from sensor spec | widened margin |
| **Constant value** | 6 | staleness detector | quarantine |
| **Uncertainty** | 7 | native — it *is* the interval | propagated |
| **Stuck-at-zero** | 8 | range + staleness | quarantine |

The literature flags the nastiest property: *"drift is difficult because the
sensor may still produce values that look normal, but the measurement slowly
becomes less trustworthy."* That is precisely what §3 catches.

---

## 2. Three-state output: ALERT / SAFE / **ABSTAIN**

The most consequential design decision in this document.

Alerting must not be binary. Since **outliers are the most frequent error type**,
a binary alerting system converts the most common data fault directly into false
alarms — and alarm fatigue is the documented reason operators stop trusting these
systems.

### 2.1 Interval-valued margins

When an input carries uncertainty, propagate it as an interval and evaluate the
margin as an interval:

```
Scenario: wear = 210 min, L variant (limit 11,000 min·N·m)

torque reading      margin interval        decision
───────────────────────────────────────────────────────────────────
45 N·m (trusted)    [ 1550,  1550 ]        SAFE     — wholly positive
45 ± 8  (suspect)   [ −130,  3230 ]        ABSTAIN  — straddles zero
45 ± 25 (degraded)  [−3700,  6800 ]        ABSTAIN  — straddles zero
```

**Rule:** alert only when the *entire* interval is negative; declare safe only
when the *entire* interval is positive; otherwise **abstain and raise a data-quality
flag**.

Cost: 2× the flops (0.44 µs instead of 0.22 µs). Gain: **a bad reading can never
produce a false alert.** It produces silence plus a maintenance ticket for the
instrument.

Interval width sources, in priority order:
1. Declared sensor accuracy from instrument metadata
2. Observed short-window dispersion (Hampel/MAD estimator)
3. Imputation bounds when a value is missing (last known ± physical rate limit)
4. A wide default when provenance is unknown — which correctly yields ABSTAIN

### 2.2 Two-track evaluation

Robustness needs a window; speed needs statelessness. Do both:

| Track | State | Latency | Use |
|---|---|---|---|
| Instantaneous | none | 0.22 µs | display, drill-down, queries |
| Robust (Hampel over short window) | small ring buffer | ~1 µs | **alerting decisions** |

Alerts fire only on the robust track. A single spike never pages anyone; the
instantaneous track still shows it, tagged.

---

## 3. Gate 2 — is the instrument honest?

### 3.1 Physics invariants

Relationships that must hold *regardless of operating point*. All four verified
on the real data:

| # | Invariant | Measured | Basis |
|---|---|---|---|
| I1 | `process_temp > air_temp` | **0 violations / 10,000** | thermodynamic |
| I2 | ΔT ~ N(10, 1) | **μ = 10.001, σ = 1.001** | documented process |
| I3 | r(rpm, torque) ≈ −0.875 | **−0.8750** | design coupling at 2860 W |
| I4 | mean power stable | **6,280 W** | design point |

I2 matches the documented generative process to three decimals.

### 3.2 The discrimination that matters

An invariant break means **instrumentation**. Invariants holding while margins
shift means **operations**. Tested by injection:

| Scenario | HDF alerts | z(ΔT) | z(rpm) | Verdict |
|---|---:|---:|---:|---|
| baseline | 115 | 0.1 | 0.0 | ok |
| air sensor drifts −0.4 K | **53 (46 %)** | **40.0** | 0.0 | **SENSOR** |
| process genuinely slows | **188 (163 %)** | 0.1 | **−22.3** | **PROCESS** |

Read the second row carefully.

> A **0.4 K thermocouple drift makes heat-dissipation alerts drop by 54 %.**
> A conventional copilot reports *"HDF failures down 54 % — good month."* The
> sensor is broken and the plant is now blind to exactly the failure mode it
> believes it solved.
>
> **A safety incident dressed as a KPI improvement.**

Same symptom, opposite cause, cleanly separated. No classifier can make this
distinction — it has no notion of what *must* be true.

### 3.3 Additional instrument checks

| Check | Trigger |
|---|---|
| Range | value outside physically possible bounds |
| Staleness | identical value for N consecutive samples |
| Rate limit | change exceeds physically possible slew |
| Cross-sensor | two sensors that must co-move stop co-moving |
| Clock | out-of-order or future timestamps → watermark |

---

## 4. Gate 3 — is the rule still valid?

The dangerous drift is not the sensor. It is the **rule silently becoming wrong**
— a tool supplier changes and the real overstrain limit is now 10,200, not
11,000. The system is confidently, invisibly wrong.

### 4.1 The KB calibration monitor

Two counters, computed from quantities we already have:

- **Surprise failures** — a failure occurred at `margin > 0` → threshold too *loose*
- **False alarms** — `margin < 0` with no failure → threshold too *tight*

Measured sensitivity by perturbing the OSF threshold:

| KB error | surprise failures | false alarms | total signal |
|---:|---:|---:|---:|
| −5.0 % | 0 | 57 | 57 |
| −2.0 % | 0 | 24 | 24 |
| −1.0 % | 0 | 8 | 8 |
| −0.5 % | 0 | 2 | 2 |
| **0.0 %** | **0** | **0** | **0** |
| +1.0 % | 13 | 0 | 13 |
| +2.0 % | 24 | 0 | 24 |
| +5.0 % | 45 | 0 | 45 |

Three properties:

1. **Zero only at the true threshold.**
2. **Monotone** in the magnitude of the error.
3. **Directional** — which counter fires tells you *which way to move*.

It requires **no model and no retraining**: only margins we already compute and
failures the CMMS eventually reports. It therefore works with **delayed labels**,
which is essential — real work orders arrive days after the event.

We could not find this construct in any product or paper.

### 4.2 Operating procedure

```
rolling window (default 30 days, lagged for label delay)
  → compute surprise + false-alarm counts per rule
  → if either exceeds its control limit:
        raise KB DRIFT ALERT (not a machine alert)
        re-estimate the threshold with CI (discovery/threshold.py)
        propose the update with evidence
        → ENGINEER CONFIRMS
        → shadow mode: run old and new in parallel, diff the alerts
        → cut over; KB versions with provenance and author
```

**Shadow mode** is routine in software deployment and rare in industrial rule
systems. It is what makes a threshold change safe.

---

## 5. Agent-level failure modes

| Failure | Defense |
|---|---|
| Silent regression on provider model upgrade | Plans are **data** — diff plans across versions structurally, not prose. Eval suite is a CI gate. |
| Sycophancy / accepting a false premise | Gate 1 premise verification |
| Overconfidence on thin data | Wilson CI mandatory; refuse point estimates when CI half-width > 50 % |
| Stale cache serving wrong answers | Answer cache keyed on `data_version` + `kb_version`; plan cache is data-independent and safe |
| Context poisoning across turns | Session state typed and **displayed**; every answer restates its resolved scope |
| Confounded conclusions | Automatic collinearity detection (r = −0.875 here) |
| Unit mismatch across sites | Dimensional validation refuses cross-unit arithmetic |

---

## 6. Operational reliability

- **Alarm rationalisation** — debounce, persistence counts (N consecutive), and
  ISA-18.2 prioritisation. A margin crossing at fleet scale will bury an operator
  otherwise.
- **Alert self-audit** — for every alert, later record whether the crossing
  actually occurred and whether anyone acted. Yields closed-loop precision/recall
  on our *own alerting*, which is the input to tuning the gate. Almost nobody
  instruments this.
- **Explaining non-events** — *"why did nothing fire?"* is answerable:
  *"all five margins stayed positive; closest approach was power at +412 W at
  14:32."* Classifiers cannot explain an absence.
- **Cold start** — a new asset class with no failure history gets hierarchical
  priors from sibling assets, with wide intervals that correctly produce more
  ABSTAIN until evidence accumulates.

---

## 7. Honest limits

- These defenses are validated against **injected** drift on a synthetic dataset.
  Real drift is gradual, multi-sensor, and correlated with season and product
  mix. The mechanisms are sound and the sensitivity curves are real; field
  validation is the missing step and is **not** claimed.
- The KB monitor needs failures to accumulate. On a rule that fires twice a year
  it will take years to detect a 1 % error.
- Interval widths are only as good as the instrument metadata. Garbage
  uncertainty estimates produce either excessive abstention or false confidence.

---

**Next:** [07-CONTEXT.md](07-CONTEXT.md) — routing, session state, and latency.
