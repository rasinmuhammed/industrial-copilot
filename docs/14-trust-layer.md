# 14 · The Trust Layer — what we got wrong, and the thing that fixes it

> Reflection written after adversarial hardening, before the next build phase.
> Everything here was verified against the code or against published research.
> Nothing is aspirational.

---

## 1 · The sentence that indicts the whole system

`copilot/stream.py:191` — the entry point of the only code path that would ever
see production data:

```python
point = OperatingPoint(
    air_temp_k=float(row["air_temperature_k"]),
    process_temp_k=float(row["process_temperature_k"]),
    rotational_speed_rpm=float(row["rotational_speed_rpm"]),
    torque_nm=float(row["torque_nm"]),
    tool_wear_min=float(row["tool_wear_min"]),
    product_type=row["product_type"],
)
```

Six `float()` calls. No NaN check. No range check. No staleness check. No
timestamp. No test that the value differs from the last one. A missing key
raises `KeyError` and kills the tick.

Downstream of this line sits everything we are proud of: signed margins,
interval arithmetic, three-state ALERT/SAFE/ABSTAIN, first-passage forecasting,
proof-carrying narration, a verifier that refuses any numeral without a slot.

**All of it is rigorous arithmetic performed on unexamined numbers.**

We built a system that cannot state a wrong number. We did not build a system
that can doubt its inputs. In a factory those are not the same problem, and the
second one is the one that gets you fired.

---

## 2 · What we actually got wrong

Three errors, in descending order of severity.

### 2.1 We solved layers 2 and 3 and skipped layer 1

Every answer this system gives depends on three questions being true:

| Layer | Question | Failure looks like | Our coverage |
|---|---|---|---|
| **1 · Instrument** | Is the signal real? | Confident answer about a machine we are blind to | **none** |
| **2 · Computation** | Is the number right? | Hallucinated figure, unit error, wrong join | strong |
| **3 · Inference** | Is the claim supported? | False premise, confounded ranking, noise read as signal | strong |

Layer 2 is the Analysis IR, the dimensional type system, the fail-closed
renderer. Layer 3 is the four gates and premise verification. Both are good, and
both are the layers that *research papers* are written about.

Layer 1 is the layer that *deployments* die on. We have nothing.

This is not a small omission at the edge. It inverts the value of everything
else: the more rigorous layers 2 and 3 are, the more confidently the system will
assert a conclusion drawn from a dead sensor.

### 2.2 We inherited the dataset's cleanliness as an architectural assumption

AI4I 2020 is synthetic, complete, in-order, single-regime, correctly-calibrated,
and never has a stuck channel. We never wrote code to handle any of those going
wrong, because the data never made us. That is exactly the criticism we already
answered once for *thresholds* — we showed threshold discovery recovers the
documented values from data alone — but we never answered it for *data quality*.

The threshold work proved the physics is not memorised. It did not prove the
pipeline is not brittle. Those are different claims, and we only earned one.

### 2.3 ABSTAIN is defined by interval width alone

Today a margin abstains when its uncertainty interval straddles zero. That is a
statement about *sensor precision*. It is the right rule for a healthy noisy
sensor and the wrong rule for a broken one, because a stuck sensor has **zero**
noise — its interval is tight, so it will never abstain. It will say SAFE, with
maximum confidence, forever.

The research is explicit on this: stuck faults are hard to detect precisely
because their signature is *the absence of noise* after fault onset. Our
uncertainty model reads absence of noise as high confidence. We built the one
abstention rule that inverts on the most common sensor fault in industry.

---

## 3 · The edge cases we have not addressed

Grouped by layer. Every one is a real production condition, not a hypothetical.

### Layer 1 — instrument

| # | Condition | What our system does today |
|---|---|---|
| 1 | **Stuck-at / frozen channel** | Reports large healthy headroom forever, with tight intervals and high confidence |
| 2 | **Bias / miscalibration** after sensor swap | Silently shifts every margin by the bias; the KB monitor eventually blames the *threshold* |
| 3 | **Slow drift** | Undetectable at onset by design; the hardest fault class in the literature |
| 4 | **Spike / dropout** | The robust track damps torque spikes only — the other five channels are unguarded |
| 5 | **Precision degradation** | Uncertainty model is static; a degrading sensor keeps its original assumed σ |
| 6 | **Saturation / railing** | A channel pinned at full scale reads as a legitimate extreme value |
| 7 | **NaN / null / missing key** | `float(None)` → `TypeError`; missing key → `KeyError`; tick dies |
| 8 | **Unit change** (K→°C, N·m→lb-ft) | Dimensional system validates *plan* units, never *ingest* units |
| 9 | **Staleness** | No timestamp is read at all. A margin from four hours ago is presented as current |

### Layer 1.5 — transport

| # | Condition | What our system does today |
|---|---|---|
| 10 | **Out-of-order arrival** | No event-time model; last message wins |
| 11 | **Duplicate delivery** (at-least-once MQTT/Kafka) | Double-counts into the robust track's window |
| 12 | **Backpressure / dropped samples** | A gap is invisible; slope is computed across it as if continuous |
| 13 | **Clock skew, DST, NTP step** | Synthetic 2-min takt assumed; no real clock discipline |

### Layer 2.5 — regime

| # | Condition | What our system does today |
|---|---|---|
| 14 | **Recipe / product changeover** | One global threshold set; a new regime looks like mass anomaly |
| 15 | **Maintenance reset** (tool change zeroes wear) | Forecasts a crossing that will never happen — no CMMS work-order ingest |
| 16 | **Cold start** on a new asset | No baseline; robust track needs history it does not have |
| 17 | **Fleet heterogeneity** across 1,000 sites | Thresholds are global constants, not per-asset priors |

### Layer 3 — inference

| # | Condition | What our system does today |
|---|---|---|
| 18 | **Confounded grouping** | Ranks groups that are perfectly confounded with another dimension, with no warning that the ranking is the confounder in disguise |
| 19 | **Multiple comparisons** | Ranks N groups and names a worst; with enough groups the extreme is guaranteed by chance |
| 20 | **Simpson's paradox** on rollups | `min()` composes margins correctly; grouped *rates* have no such protection |
| 21 | **Unforecastable failures** | Literature puts 15–25% of failures as having no precursor signature. We have no "this class was not forecastable" category, so those land as silent misses |

Items 18 and 19 deserve emphasis because they are the ones that *look* like
insight. A ranked list of assets is the single most actionable artifact a
maintenance copilot produces — it dispatches technicians. It is also the artifact
most likely to be pure noise, and we currently print it without qualification.

---

## 4 · The thing that fixes it

### 4.1 The insight — and the first version of it that was wrong

The original form of this argument was: our semantic layer declares
`power_w = torque_nm * rotational_speed_rpm * 2 * pi / 60`, we use it as a type
rule, and it is *also* a parity equation.

**That is false, and the error is instructive.** Power is *derived* from torque
and speed. Its residual is identically zero by construction, so it carries no
information at all. A derived quantity provides no redundancy. Measuring it
confirmed the point twice over: the relation has 17% scatter against the design
figure, because it is not a constraint, it is a definition.

The correct statement is narrower and survives contact with data:

> A relation is useful only if it links **independently measured** quantities.

Two such relations exist here, both verified against all 10,000 rows:

| Relation | Measured | Status |
|---|---|---|
| `process_temp − air_temp = 10.001 ± 1.001 K` | two independent thermocouples | **real redundancy** |
| wear is monotone within a tool life | kinematic constraint | **real** |
| `power = torque × ω` | derived, not measured | **no information** |

In fault detection and isolation theory, an analytical redundancy relation is any
equation among measured signals that must hold when every instrument is honest.
Its residual is ~0 in the fault-free case and departs sharply when a signal lies
— detecting instrument faults *without adding hardware*, because the redundancy
is analytical rather than physical.

We already have the relations. We wrote them down for a different reason. We
have simply never evaluated them as residuals.

> **The semantic layer is not a vocabulary. It is a physical model, and a
> physical model generates its own self-test.**

### 4.2 Why this is the high-value move, not merely a clever one

Because the residual **separates instrument faults from process faults**, which
is the distinction that determines whether an operator ever trusts the system
again.

| Parity residual | Margins | Diagnosis | Action |
|---|---|---|---|
| ≈ 0 | positive | healthy | none |
| ≈ 0 | **negative** | **process fault** — the machine is genuinely in trouble | dispatch to the machine |
| **large** | either | **instrument fault** — a channel is lying | dispatch to the sensor |
| large | — | model invalid for this regime | abstain and say why |

Every industrial copilot on the market conflates rows two and three. That is the
origin of alarm fatigue: operators stop trusting a system that cries wolf about
machines whose sensors are broken. Splitting those two rows is worth more in a
real plant than another point of AUC, and no product we surveyed does it.

### 4.2b The hard limit, found by building it

A local level estimator **adapts** to a bias step, because for a genuine process
a level change is exactly what it should track. So "the torque sensor shifted by
25 N·m" and "the torque genuinely shifted by 25 N·m" produce identical
innovations. They are formally indistinguishable from that channel alone.

This is a standard identifiability result in FDI, not an implementation
shortcoming, and it has a sharp consequence: **CUSUM never fires on a
single-channel bias.** An earlier version of the fault-injection harness
appeared to catch torque bias at +31 cycles — that was a coincident baseline
false alarm on a different channel, which is exactly how this class of mistake
survives into production.

So the system declares what it cannot see:

> Bias and slow drift on `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`
> are not detectable: these channels have no redundancy partner. Freezes,
> dropouts and invalid values on them are still caught.

By contrast the two thermocouples guard each other, and a 6 K decoupling is
caught **on the first sample at 82 σ**.

That declaration is the useful engineering output. If a plant wants bias
protection on torque, it needs a second measurement or a physical relation
involving it — a procurement decision, and this is the module that can name it.

### 4.3 What it lets the system say

Measured output from the running system, four cycles after a torque channel was
frozen mid-stream:

> **M-07 — ABSTAIN.** `torque_nm` unusable (frozen: 5 identical readings,
> against a limit of 5 derived from this channel's own measured repeat rate).
> This is an instrument fault, not a process fault — dispatch to the sensor.
> Verify the named channel before acting on this machine.

One page for that fault across 150 frozen cycles; 297 further alerts suppressed.
No copilot says this today. They say a number.

Note what makes it defensible: **no threshold in the module was chosen.** The
stuck limit is a chi-square quantile at a stated false-positive rate; the CUSUM
limit is inverted from a target average run length via Siegmund's approximation
(validated against the textbook value — k = 0.5, h = 5 gives 465 cycles); the
gate is a normal quantile.

The noise model is not declared either. It is identified per channel by method
of moments on the first differences — `Var(Δz) = q + 2r`,
`Cov(Δz_t, Δz_{t-1}) = −r`. On this dataset that recovers torque's documented
σ = 10 N·m as **10.039 from the data alone**.

That mattered more than expected. The first implementation *did* hard-code the
noise figures, with a comment claiming they were measured. They were invented,
and the temperature values were six times too large — which flagged **94% of
healthy cycles as frozen sensors**. A constant chosen but described as derived
is the precise failure this project exists to prevent, and it still got in. It
was caught by measuring, not by review.

Measured false-alarm performance on 9,600 healthy cycles: **16 episodes, one per
600 cycles**, comfortably inside the ~61 the stated budget predicts. Every
injected fault class is caught: hard freeze at +4 cycles, dropout at +3, NaN at
+3, garbage string at +3, thermal decoupling at +0.

### 4.4 Speaking the plant's own language

The four outcomes map exactly onto **NAMUR NE 107**, the standard that process
industries already use for device health — *Failure*, *Function check*,
*Out of specification*, *Maintenance required*. Emitting NE 107 status alongside
each margin means the copilot's diagnostic vocabulary is one an instrumentation
technician already reads on their handheld, and it routes into existing asset
management systems without translation.

That is a small implementation detail and a large credibility signal: it shows
the system was designed by someone who knows the plant floor has standards, not
only that ML has benchmarks.

### 4.5 The inference-layer counterpart: falsify your own ranking

For items 18 and 19, one cheap general guard covers both — a **permutation
null**. Before reporting that a group is worst, shuffle the group labels a few
thousand times and ask how often chance alone produces a spread this extreme. If
the observed ranking sits inside the permuted null, the honest output is *"no
detectable difference between assets; the spread is what randomness looks like
at this sample size"*.

It needs no distributional assumption, works for any grouping, costs
milliseconds at our data sizes, and turns the single most dangerous artifact we
produce — the ranked asset list — into one that has passed a falsification test.

It also composes with what we already have: the premise gate tests claims the
*user* brings. The permutation null tests claims *we* would otherwise volunteer.

---

## 5 · The thesis, restated

Where we started:

> Compute the signed distance to the failure boundary, not the probability of
> failure. Margins invert, forecast, and compose.

That remains right, and it is the reason the arithmetic is trustworthy. But it
is incomplete, because it says nothing about where the numbers came from.

Where this takes us:

> **A margin is only as real as the sensor that produced it. So compute the
> margin, and compute — from the same physical model, at the same instant — the
> evidence that the sensor is telling the truth. Report both, and when the
> second one fails, refuse to report the first.**

Three questions, answered in order, every tick:

1. **Is the signal real?** — parity residuals, noise-floor collapse, staleness,
   range. Fails → NE 107 status, name the channel, abstain on that machine only.
2. **Is the number right?** — Analysis IR, dimensional types, verified
   narration. Already built.
3. **Is the claim supported?** — premise gates, permutation nulls, confound
   detection. Mostly built; add the null.

A system that answers all three, and says which one failed when one does, is not
a better chatbot over a maintenance database. It is an instrument.

Instruments get trusted. Chatbots get audited.

---

## 6 · Honest scope

- This does not make us better than Siemens or Cognite as a *product*. They own
  the OT connector moat, real vibration and thermographic modalities, and years
  of deployment learning. The claim is narrower and defensible: on the axes the
  brief names — latency, context engineering, hallucination reduction — plus the
  instrument/process split, this answers better.
- Parity relations require a physical model. We have one because the AI4I
  failure modes are documented arithmetic. A plant with undocumented equipment
  needs the relations discovered or elicited first — that is real work, and it
  is the honest limit of the approach.
- Items 3 (slow drift) and 21 (unforecastable failures) are not fully solvable.
  Drift at onset is information-theoretically hard; a fraction of failures have
  no precursor. The correct engineering response is to *bound and report* them,
  not to claim coverage.
