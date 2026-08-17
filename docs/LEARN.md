# Argus, explained from zero

*A learning document. It assumes you know nothing about what was built and ends
with you able to defend every claim in it.*

The fifteen numbered docs beside this one were written **during** construction -
they are design records, arguing for decisions as they were made. This one is
written **backwards**, from the finished system, in the order a person learns
best: problem, idea, mechanism, evidence, limits.

---

## Part 0 - The one-paragraph version

Read this now, and again at the end. It should mean four times as much the
second time.

> Industrial failure prediction is normally a classifier: features in,
> probability out. Argus does something different. Every failure mode in this
> process is a **documented inequality** - a line the machine must not cross.
> So instead of estimating *how likely* failure is, it computes the **signed
> distance to that line**. That number is called the margin. It is exact, it has
> units, it can be inverted into an instruction ("reduce torque to 51.5 N·m"),
> and it can be projected forward into a deadline ("crossing in 11 cycles"). A
> language model never produces a number; it only chooses which calculation to
> run and reads the result back in English. A verifier checks that every numeral
> in the answer came from the calculation.

---

## Part 1 - The problem, and why the obvious answer is wrong

### 1.1 What the brief asked for

Build a copilot over industrial machine data that can (a) understand machine
behaviour, (b) analyse history, (c) investigate failures, (d) handle follow-up
questions. Explicitly no credit for the best LLM, the prettiest frontend, or the
most frameworks. Credit for **latency, context engineering, hallucination
reduction**, and an argument for scaling to 1,000 factories.

That last sentence is the whole design brief in disguise. It says: *the
interesting problem is not intelligence, it is trust and cost.*

### 1.2 The obvious solution

Ninety percent of submissions to a brief like this look like:

```
CSV → pandas → train XGBoost → wrap in LangChain → "ask me anything"
```

It works. It scores ~0.90 F1 on AI4I. It demos well. And it is unusable in a
factory, for three reasons that took this project a long time to articulate:

**One: a probability is not an instruction.** A model says "0.83 probability of
failure". The operator's next question is *"so what do I change?"* and the model
cannot answer, because probability is not invertible. There is no operation that
takes 0.83 and returns "reduce torque by 4 N·m".

**Two: a probability is not auditable.** When the model is wrong, there is no
sentence explaining why. Gradient attribution tells you which feature moved the
number; it does not tell you which physical limit was approached. In a regulated
plant, "the model said so" is not a maintenance justification.

**Three: probabilities from different models do not compose.** A 0.3 from a
thermal model and a 0.3 from a wear model are not the same quantity. You cannot
rank them, average them, or take a minimum. Which means you cannot build a fleet
view - the single most useful screen in the product - out of them.

### 1.3 The observation everything else follows from

Read the AI4I documentation and you find the failure modes are not patterns
discovered from data. They are **rules the data generator was built from**:

| Mode | The documented rule |
|---|---|
| **HDF** - heat dissipation failure | ΔT < 8.6 K **and** rpm < 1380 |
| **PWF** - power failure | torque × ω < 3,500 W **or** > 9,000 W |
| **OSF** - overstrain failure | wear × torque > 11,000 / 12,000 / 13,000 (L/M/H) |
| **TWF** - tool wear failure | random, in a wear window of 200–240 min |
| **RNF** - random failure | random, 0.1%, by construction |

We transcribed each rule and scored it against the published labels on all
10,000 rows:

| Mode | Fires | Labelled | False positives | False negatives |
|---|---:|---:|---:|---:|
| HDF | 115 | 115 | **0** | **0** |
| PWF | 95 | 95 | **0** | **0** |
| OSF | 98 | 98 | **0** | **0** |

Zero errors, with no training whatsoever. (308 rule firings across the three
modes; they overlap on 21 cycles, so they cover 287 *distinct* failed cycles.)

This is where most people stop and say "well, that's just the synthetic dataset,
it doesn't generalise." That objection is worth taking seriously, and Part 6
answers it properly. The short answer: **real plants have documented limits
too** - they are in the equipment manual, the ISO standard, the process
safety review. What is fake about AI4I is not that limits exist; it is only that
*these particular* limits were used to generate the labels.

### 1.4 The evidence that a density model cannot do this

Here is the number that made this project's mind up.

**86.4% of the failures in AI4I (293 of 339) are within 3σ of the mean on every
single one of the five measured channels.** Individually, nothing looks unusual. Temperature is normal. Torque is
normal. Speed is normal. Wear is normal. An anomaly detector - autoencoder,
isolation forest, one-class SVM - sees an ordinary point, because it is looking
at the *marginal* distribution of each channel.

But the derived product `wear × torque` is 11,003, and the limit is 11,000.

The failure is not in the data's density. It is in a **relationship between
variables that crosses a line**. No amount of unsupervised learning finds that,
because the information is not in the distribution - it is in the physics.

---

## Part 2 - The idea: margin, not probability

### 2.1 Definition

For a rule of the form `quantity ≤ limit`, the **margin** is:

```
margin = limit − quantity
```

Signed. Negative means the line has been crossed. In the units of the quantity.

For comparison across different modes, it is **normalised** by its own
threshold:

```
normalised margin = (limit − quantity) / limit
```

Now a thermal margin and a torque margin are both dimensionless fractions of
their own limit, and can be put on the same axis.

The system's headline number for a machine is:

```
worst margin = min(HDF margin, PWF margin, OSF margin)
```

### 2.2 Why this specific choice matters - three properties

This is the part to have memorised, because it is the actual intellectual
content of the project.

**Property 1: it inverts.** A margin is an algebraic expression, so it can be
solved for any variable in it. The overstrain rule is `wear × torque ≤ 11,000`.
If wear is 200 minutes and the margin has gone negative, you can *solve for the
torque that restores it*: `torque ≤ 11,000 / 200 = 55 N·m`. That is not a
recommendation generated by a language model. It is division. This is what makes
the product prescriptive rather than merely predictive, and it is exactly what a
probability cannot do.

**Property 2: it is continuous, so it forecasts.** A probability of failure
jumps around. A margin is a smooth quantity you can watch approach zero. Fit a
slope to the last N cycles of margin, extrapolate to the axis, and you get a
**time to crossing in cycles** - a deadline, not a risk score. That is what
drives the alerts: *"Overstrain projected in 11.4 cycles."*

**Property 3: it is associative under `min()`, so it scales.** Because every
margin is normalised to the same dimensionless scale, `min()` over machines is
meaningful. Fifteen machines collapse into one ranked list; a thousand factories
collapse into one number per factory. The fleet screen exists **because of the
representation choice**, not as a feature bolted on top of it.

### 2.3 The floating-point subtlety (a good detail to know)

A margin is a difference of measured quantities near 300 K. In IEEE754 double
precision, that subtraction carries rounding error of order `ε × 300 ≈ 7×10⁻¹⁴`.

That is not hypothetical here. **128 rows in AI4I have a thermal delta of
exactly 8.6 K** - precisely on the limit. Float subtraction places 43 of them
below and 85 at or above, purely according to which decimal pair got subtracted:

```
306.9 − 298.3 = 8.599999999999966   → fires
308.6 − 300.0 = 8.600000000000023   → does not fire
```

Both are 8.6 K. The rule is deciding on representation error, not physics.

So `physics.py` defines a **boundary tolerance** of a few ULPs of the largest
operand, and margins inside it are marked *degenerate* - the sign is not
determined by the arithmetic, so the system does not assert one.

There is an honest footnote here worth repeating out loud, because it is the
kind of thing that earns credibility: the published AI4I labels were *also*
generated in floating point, so our audit reports 115/115 exact because we
reproduce UCI's float artifact. "Exact" therefore means *exact against a
float-computed ground truth*, not against the real number.

---

## Part 3 - How a question becomes an answer

This is the request path. Follow one question all the way through.

```
   "How often does L-03 fail?"
            │
            ▼
  ┌────────────────────┐
  │ 1. UNKNOWN CHECK   │  Is the subject something we measure at all?
  └────────────────────┘  "bearing temperature" → refuse, and say why
            │
            ▼
  ┌────────────────────┐  Four tiers, cheapest first:
  │ 2. PLANNER         │    cache → grammar → exemplar → LLM
  └────────────────────┘  Output: an Analysis Plan (JSON), never prose
            │
            ▼
  ┌────────────────────┐  Does every name resolve? Are units coherent?
  │ 3. IR VALIDATION   │  Invalid plan → refuse. Nothing executes yet.
  └────────────────────┘
            │
            ▼
  ┌────────────────────┐  Deterministic SQL / arithmetic over DuckDB.
  │ 4. EXECUTION       │  Produces an Evidence Bundle of SLOTS.
  └────────────────────┘
            │
            ▼
  ┌────────────────────┐  Language model writes sentences, and may only
  │ 5. NARRATION       │  reference slot ids - it never sees raw numbers.
  └────────────────────┘
            │
            ▼
  ┌────────────────────┐  Every numeral in the text must trace to a slot.
  │ 6. VERIFIER        │  Unsourced figure → the answer is refused.
  └────────────────────┘
```

### 3.1 The Analysis IR - the most important design decision

The planner does not emit an answer. It emits a **plan**: a validated JSON
object naming an operation, filters, metrics, and grouping.

```json
{
  "op": "rate",
  "filters": [{"field": "machine_id", "op": "=", "value": "L-03"}],
  "group_by": []
}
```

Three things follow from this, and they are the reason the architecture is
shaped this way:

- **The model chooses, it does not compute.** The space of things the model can
  produce is a small, enumerable set of operations. It cannot produce a number,
  so it cannot produce a *wrong* number.
- **The plan is checkable before anything runs.** `ir.py` validates every field
  name against the semantic layer and every comparison for dimensional
  coherence. A plan referencing a column that does not exist is rejected with a
  suggestion, not executed into a confusing error.
- **The plan is the cache key and the audit record.** Same plan, same answer,
  forever - and the replay handle in every response lets you reproduce it.

### 3.2 The four planning tiers - this is the latency story

The brief asked for latency. Here is the answer, and note that it is an
*architectural* answer rather than a faster-GPU answer:

| Tier | What it is | Cost |
|---|---|---|
| **0 - Cache** | Normalised question shape → stored plan shape | ~microseconds |
| **1 - Grammar** | Regex + semantic layer, deterministic | ~1 ms |
| **2 - Exemplar** | Nearest-neighbour over verified past plans | ~10 ms |
| **3 - LLM** | Constrained JSON decoding, enums from the semantic layer | ~1 s |

On the 74-question eval set the realised distribution is **59 grammar, 3
exemplar, 12 refused, 0 LLM**. Measured latency: **p50 = 3.1 ms, p95 = 66.7 ms**.

The insight worth stating in an interview: *most questions in a real industrial
setting are not novel.* They are the same twenty questions asked in slightly
different words. Tier 0 and Tier 1 exist to make the common case free, and the
LLM exists as a fallback for genuine novelty. Cost is then driven by the
diversity of **question shapes**, not by the number of factories - which is
precisely the 1,000-factory scaling argument.

### 3.3 Grounding - how hallucination is actually prevented

Most systems "reduce hallucination" by prompting. This one makes it structurally
difficult, in three layers:

**Layer 1: the model cannot see numbers.** Execution produces an Evidence Bundle
of named slots:

```
rate.overall  = 0.0286   unit=fraction   n=1188   quality=exact
count.failed  = 34       unit=count      n=1188   quality=exact
```

The narrator's prompt contains the slot **ids**, not the values. It writes
"`{rate.overall}` of cycles failed". Substitution happens afterwards, in code.

**Layer 2: the verifier is adversarial.** After rendering, every numeral in the
final text is extracted and matched against the slot values. An unsourced figure
means the answer is **refused**, not flagged. There is a test file dedicated to
trying to defeat this - including spelling numbers out as words, which was a
real bypass that had to be closed.

**Layer 3: refusal is a first-class outcome.** The system declines when the
subject is not measured. This sounds obvious and is the hardest part to get
right, which is the subject of the next section.

### 3.4 Silent substitution - the failure mode this project is really about

If you remember one war story from this codebase, make it this one.

Ask: **"show me the bearing temperature"**. There is no bearing sensor in this
process. The system returned 20 rows of data. Every guarantee held perfectly:
the plan validated, the arithmetic was exact, every numeral traced to a slot,
the verifier passed. **The answer was correct and about a different sensor than
the question.**

Worse: **"how much does a replacement tool cost"** → air temperature statistics.
The token "tool" matched `tool_wear`, no subject resolved, and a fallback
described every metric it had - leading with ambient air.

This class of bug is invisible to every correctness gate, because the answer *is*
correct. It is only visible if you ask whether the answer is about the thing that
was asked. That is why the project has a **risk-coverage harness** rather than
just an accuracy score.

The same bug recurred, in a worse place, while building the operations console:

> **"How often does L-03 fail?"** → *3.39% (339 of 10,000)*

That is the entire fleet. The machine-id pattern required a noun in front of it
(`machine L-03`), so a bare `L-03` never matched, the filter was never added -
and **a filter that is never added cannot be seen to be missing**. The scoped
and unscoped answers are both well-formed sentences with correct numbers in
them. It now reads `2.86% (34 of 1,188)`, and there is a test file arguing the
case at length.

---

## Part 4 - How a sensor reading becomes an alert

That was the question path. This is the streaming path, and it is where most of
the engineering rigour lives.

```
   raw reading
       │
       ▼
  ┌─────────────┐   Kalman filter per channel + fault tests.
  │ OBSERVER    │   Is this number trustworthy at all?
  └─────────────┘
       │  posterior estimate + uncertainty
       ▼
  ┌─────────────┐   Which operating mode is the machine in?
  │ REGIME      │   Changeover → reset the observers.
  └─────────────┘
       │
       ▼
  ┌─────────────┐   Margins as INTERVALS, not points.
  │ MARGIN      │   Interval straddles zero → ABSTAIN.
  └─────────────┘
       │
       ▼
  ┌─────────────┐   Persistence, hysteresis, robust median.
  │ ALERTING    │   Alert carries lead time AND the fix.
  └─────────────┘
```

### 4.1 The observer - interrogate before computing

The single most important line in `stream.py` is a comment:

> *This used to be six bare `float()` calls. Everything below it - signed
> margins, interval arithmetic, three-state verdicts - is rigorous arithmetic,
> and all of it was being performed on unexamined numbers. The more careful the
> downstream, the more confidently the system asserted a conclusion drawn from a
> dead sensor.*

That is the trap. Rigour applied to a frozen sensor produces *confident garbage*.

So every channel gets a scalar Kalman filter (local-level model) and three fault
tests:

- **Stuck** - a χ² test on innovation energy. A sensor reading exactly 40.0 for
  50 cycles is not a stable machine; it is a dead wire.
- **Drift** - CUSUM, with the alarm threshold set by inverting Siegmund's ARL
  approximation so the false-alarm rate is *chosen*, not discovered.
- **Dropout** - availability floor at 0.98 for intermittent channels.

The categories map onto **NAMUR NE 107**, the standard device-status taxonomy,
so the output is something a real control system already knows how to consume.

### 4.2 Noise identified from data, not invented

The observers need per-channel noise variances. The first version *invented*
them, with comments claiming they were measured. Result: **94% false alarms**.

They are now identified by method of moments, from the differenced signal:

```
Var(Δz)            = q + 2r
Cov(Δz_t, Δz_{t-1}) = −r
```

Two equations, two unknowns: process noise `q` and measurement noise `r`, solved
per channel from the data itself.

The validation is the satisfying part. AI4I documents torque as **N(40, 10)**.
Nothing in the identification pipeline is told that. It recovers **σ = 10.039**.

A related category error, worth knowing because it is a genuinely subtle bug:
tool wear is a **counter**, not a level. It ramps up and resets to zero on a tool
change. Tested as a level, that sawtooth reads as enormous process noise -
inflating `q` to a standard deviation of 21. Channels now carry a `ChannelKind`
and counters are differenced before testing.

### 4.3 Three-state verdicts - abstention as a design principle

Once uncertainty is propagated, a margin is not a number. It is an **interval**.
Which gives three outcomes rather than two:

| Interval | Verdict |
|---|---|
| Entirely above zero | **SAFE** |
| Entirely below zero | **ALERT** |
| **Straddles zero** | **ABSTAIN** |

Abstention is not a failure. It is the system saying *"given how much I trust
this instrument right now, I cannot determine the sign."* Uncertainty propagates
as covariance: a stale channel has a wide posterior, which widens the margin
interval, which makes it straddle zero, which yields abstain. **No special-casing
anywhere** - it falls out of the arithmetic.

You can see this in the console. During the first ~50 cycles of a replay the
observers are still calibrating and the activity rail fills with *"margin
withheld: the interval straddles zero given current input uncertainty."* That is
the system being honest about warm-up rather than guessing.

### 4.4 Alert discipline

On a full 10,000-cycle replay: **306 alerts raised, 4,805 suppressed.**

That ratio is the product. An industrial alerting system that fires 5,000 times
gets muted within a week, at which point its detection rate is irrelevant.
Suppression comes from persistence requirements, hysteresis on recovery (an
intermittent channel would otherwise re-arm and page on every cycle - 517 alerts
where there was one fault), and a robust median track so a single outlier sample
cannot fire.

---

## Part 5 - Where the machine learning actually is

Be precise about this, because "is there any ML in it?" is the first question a
sceptical interviewer asks. The honest answer is a table.

| Component | Method | Learned from data? |
|---|---|---|
| Failure boundaries | Documented rules | No - transcribed, then audited |
| Rule **discovery** (new plant) | Separate-and-conquer tree induction | **Yes** |
| Noise variances | Method of moments | **Yes** |
| Drift detection | CUSUM + Siegmund ARL inversion | **Yes** (thresholds) |
| Regime detection | Sequential leader clustering, Mahalanobis radius | **Yes** |
| Hazard curve | Isotonic regression (PAVA) | **Yes** |
| RUL | Inverse-Gaussian first passage + split conformal | **Yes** (calibration) |
| Threshold tuning | Outcome feedback from closed work orders | **Yes** |
| Question planning | 4-tier, LLM only at tier 3 | Partly |
| **Any number in any answer** | Arithmetic | **Never generated** |

So: substantial statistical learning, in the places where learning is the right
tool. **Zero** learning in the place where a wrong answer is dangerous.

### 5.1 Rule discovery - the scaling answer

The obvious objection to "we transcribed documented rules" is: *what about a
plant that has no documented rules?*

`scripts/onboard.py` is the answer. Point it at a CSV and a label column, and it:

1. Profiles each channel and classifies it **counter vs level**
2. Identifies noise by method of moments
3. **Excludes target leakage** - AI4I ships the per-mode flags beside the label,
   and a learner handed those "discovers" that failure occurs when the failure
   flag is set
4. Induces rules by **separate-and-conquer**: fit, remove the explained
   failures, refit for the next mode

Point 4 matters. A single depth-3 decision tree reached **8.3% coverage**,
because one tree partitions the space and the modes compete for splits. Removing
each mode's failures and refitting surfaces the next one, reaching **>75%**.

And it recovers the real limits without being told them: rpm exactly, overstrain
to within 0.1 of 11,000, power to within 0.5, thermal delta to within 1.0 K.

Crucially, it **grades its own output**. Rules with zero false alarms are marked
`verified`; imperfect ones are marked `candidate` and say *"an engineer must
review this"*. And it leaves TWF and RNF **uncovered** - they are random by
construction, no threshold rule can find them, and a tool claiming 100% coverage
there would be lying. The gap is the honest answer.

### 5.2 RUL - a real predictive model

Remaining useful life is genuine probabilistic modelling. Overstrain margin
degrades as wear accrues, and wear accrues as a Wiener process with drift
(verified: Pearson r = 0.986 between wear and cycle index). Time to first
passage of a level by such a process is **inverse-Gaussian** - a standard PHM
formulation, giving a full predictive distribution from measured parameters
without training.

The nominal IG quantiles are then corrected by **split conformal prediction**,
which is distribution-free and finite-sample exact.

This is also a good cautionary tale. The conformal calibration queried a table
that does not exist, behind a bare `except Exception`, so the correction was
silently **0.0** - indistinguishable from a correction that was computed and came
out small. The API advertised a "90% conformal interval" and shipped the raw
uncalibrated quantile. Fixed, the correction is **17.4 cycles**, and H-01's
interval widened from **46–51** to **29–68**.

The lesson generalises: *a swallowed exception does not remove a feature; it
leaves a false claim standing in place of a working one.*

---

## Part 6 - Answering the hard objections

Rehearse these. They are what a good interviewer will ask.

**"This only works because AI4I is synthetic with known rules."**

Partly fair, and the honest framing is: the *rules* are synthetic, the
*existence* of rules is not. Real plants have documented limits - equipment
manuals, ISO standards, process safety reviews, HAZOP studies. What Argus needs
is not "a synthetic dataset"; it needs "a process where somebody wrote down what
must not happen", which is nearly every regulated industrial process. And for
plants where nobody did, `onboard.py` induces candidate rules and **marks them
as candidates**. The system also runs end-to-end against a second, entirely
different process definition with no code change - that test exists specifically
to answer this objection.

**"Why not just use a neural network?"**

We built one - a 5→3→5 residual autoencoder written directly in numpy, under
100 parameters, running in microseconds - and benchmarked it against a PCA
baseline (`scripts/bench_neural.py`). It did not beat the physics. Not because
neural networks are bad, but because **86.4% of failures are within 3σ on every
channel**: the information is not in the density, so a density model has nothing
to find.

We also checked whether *sequence* models applied, and they do not. AI4I has
1,490 tool-life segments with a **median length of six cycles and not one
reaching twenty** - the longest monotone wear run in the whole dataset is 16
cycles. There are no degradation trajectories to learn from, so an LSTM or
temporal transformer fitted here would be fitting noise and any RUL curve it
produced would be an artefact.

Both negative results were kept rather than quietly deleted, because knowing
what does not work - and why - is part of the argument.

**"How does it compare to a trained classifier?"**

Against published AI4I baselines - Random Forest ≈ 0.882 F1, XGBoost ≈ 0.901 -
the margin engine achieves **precision 1.0000, recall 0.8466, F1 0.9169**, with
**zero false alarms** and **no training**. The recall gap is entirely TWF and
RNF, which are random by construction and which no method can predict. That is
worth saying plainly: we do not beat the baselines by being cleverer at the
learnable part; we match or exceed them while being *exactly right* on the
deterministic part and *honestly silent* on the irreducible part.

**"What if the sensors are wrong?"**

That is what Part 4 is entirely about, and it was the first version's biggest
flaw.

**"Isn't the LLM still a hallucination risk?"**

It cannot emit a number. It selects an operation from an enumerated set, and the
verifier refuses any answer containing a figure that does not trace to a computed
slot.

---

## Part 7 - Scaling to 1,000 factories

The brief asked for a systems argument. Here it is in five points.

1. **Cost scales with question diversity, not factory count.** Cache keys
   normalise away entities - `"why did cycle 9016 fail"` and `"why did cycle
   4045 fail"` are one key. One shared cache serves every site. (This is also
   what caused a real bug: because the key erases entities, the *value* must not
   contain them. It stored the operating point, so every envelope question
   returned the first one's setpoint. The invariant was written in the module
   docstring and violated three lines below it.)

2. **The LLM is a fallback, not a dependency.** 0 of 74 eval questions reached
   tier 3. A site with no connectivity still answers via grammar and cache.

3. **A new process is a file, not a deploy.** Every physical constant is read
   from a YAML process model rather than compiled in. Onboarding factory #2 is
   writing a definition, not editing Python.

4. **Edge-deployable.** No GPU, no API key, no external calls. DuckDB is
   embedded; the whole thing runs on a plant PC, which matters because factory
   networks are air-gapped far more often than cloud architectures assume.

5. **Storage is split by lifecycle.** The analytical warehouse (DuckDB) is
   rebuilt from source and read-only at serve time. The work-order ledger
   (SQLite, WAL) is appended to and must survive rebuilds. Conflating them was a
   real bug - DuckDB takes an exclusive lock, so a second worker or an
   overlapping deploy dies at startup.

---

## Part 8 - The evidence, and what it does not say

Every number here is regenerated by `make verify`, which fails the build if any
figure drifts.

| Measure | Value |
|---|---|
| Deterministic rule accuracy | 308 firings, 308 labelled, **0 FP, 0 FN** |
| Detection vs baselines | P **1.0000** / R **0.8466** / F1 **0.9169** |
| Question coverage | **98.4%** (73 of 74) |
| Soundness - every numeral sourced | **100%** |
| Silent failures | **0** |
| Refusal precision | **91.7%** |
| Latency p50 / p95 | **3.1 ms / 66.7 ms** |
| Alerts raised / suppressed | **306 / 4,805** |
| Tests | **569** |

**What these numbers do not say**, which you should volunteer before you are
asked:

- Accuracy is against float-computed labels, so "exact" is narrower than it sounds.
- TWF and RNF are unpredictable and are not predicted. The recall gap is them.
- `machine_id` and `shift` are a **synthetic overlay** - AI4I is a pool of
  cycles, not per-machine time series. Answers depending on them carry a warning
  saying so.
- The coverage harness is 74 questions written by the same people who built the
  system. It is a real instrument, not an independent benchmark.

---

## Part 9 - How to explain it out loud

**In 30 seconds.** *"Most predictive maintenance gives you a probability. We
compute the signed distance to the failure boundary instead. That's a number
with units, so you can invert it into an instruction - 'reduce torque to 51.5
N·m' - and project it into a deadline - 'crossing in 11 cycles'. The language
model never produces a number; it picks which calculation to run and reads the
result back."*

**In two minutes.** Add: the four-tier planner and why p50 is 3 ms; the Evidence
Bundle and the verifier; three-state verdicts with abstention when the
instrument cannot be trusted; and the fact that margins normalise, which is what
makes a fleet view possible at all.

**If challenged.** Go straight to the honest limits in Part 8 before being
pushed there. The strongest thing about this system is not any single number -
it is that it knows what it does not know, and there is machinery enforcing
that: the refusal path, abstention, candidate-vs-verified rule grading, and the
warnings on synthetic columns.

**The story to tell if you only tell one.** Silent substitution. *"The worst bug
we found wasn't a wrong number - it was a perfectly correct answer about a
different sensor than the one the engineer asked about. Every guarantee held.
That's the failure mode nobody tests for, because the answer is right. It's why
we built a risk-coverage harness instead of just measuring accuracy."*

---

## Appendix - Reading the code in the right order

If you want to actually understand the implementation, read in this sequence.
Roughly two hours.

| # | File | Why |
|---|---|---|
| 1 | `copilot/physics.py` | The process model. Everything else serves this. |
| 2 | `copilot/ir.py` | The Analysis IR and its validation. |
| 3 | `copilot/planner/grammar.py` | Tier 1, and where machine scoping lives. |
| 4 | `copilot/evidence.py` + `copilot/verify.py` | Slots and the adversarial check. |
| 5 | `copilot/observer.py` | Kalman, χ², CUSUM, NAMUR. |
| 6 | `copilot/stream.py` | Interval margins, three-state verdicts, alerting. |
| 7 | `copilot/rul.py` | Inverse-Gaussian first passage + conformal. |
| 8 | `copilot/static/console.html` | The operations console. |
| 9 | `evals/coverage.py` | The risk-coverage harness. |
| 10 | `scripts/onboard.py` | Rule discovery - the scaling claim, executable. |

Then read the test files named after bugs - `test_silent_substitution.py`,
`test_machine_scope.py`, `test_prescription_integrity.py`. Each one opens with a
prose account of a real defect, what it produced, and why nothing caught it.
They are the most honest documentation in the repository.
