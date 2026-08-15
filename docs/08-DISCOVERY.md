# 08 — Knowledge Discovery: Learning the Boundary, Not the Label

> **The fair criticism:** our exactness is an artifact of UCI documenting its own
> rules. Real bearings do not come with a threshold spec sheet.
>
> **The answer:** the rules are recoverable from data alone. Demonstrated below,
> on this dataset, to within 0.01–3.4 %.

---

## 1. Reframe: learn the boundary, not the label

Conventional PdM learns `P(failure | x)` — a quantity that does not compose under
aggregation, does not invert to a setpoint, and cannot be audited by an engineer.

We learn the **boundary itself**: the functional form `f(x)` and the threshold
`θ` such that `f(x) > θ` constitutes failure. That output is a *knowledge base
entry*, not a model artifact.

---

## 2. The experiment

Hide the documented rules. Attempt recovery from data alone.

```bash
python scripts/discover_rules.py
```

| Mode | F1, **raw sensors** | F1, **dimensional quantities** | Recovered θ | Documented | Error |
|---|---:|---:|---|---|---:|
| PWF | 0.393 | **1.000** | 3496.1 / 9001.2 W | 3500 / 9000 | 0.11 % / **0.014 %** |
| HDF | 0.475 | 0.939 | ΔT 8.65 K; ω 144.51 rad/s → **1379.97 rpm** | 8.6 K; 1380 rpm | 0.58 % / **0.002 %** |
| OSF | 0.398 | 0.891 | 10,998.5 min·N·m (L) | 11,000 | **0.014 %** |

Nobody told the model that 1380 rpm mattered. It found **144.51 rad/s**, which is
1379.97 rpm.

Per-variant OSF bracketing (interval between max-negative and min-positive):

| Variant | bracket | midpoint | documented | error | n |
|---|---|---:|---:|---:|---:|
| L | [10994, 11003] | 10,998 | 11,000 | **0.01 %** | 6,000 |
| M | [11920, 12337] | 12,128 | 12,000 | 1.07 % | 2,997 |
| H | [11873, 13235] | 12,553 | 13,000 | 3.43 % | 1,003 |

The H error is **informative, not embarrassing**: fewer H products → wider
bracket → more uncertainty. The data itself tells you how much to trust the
threshold. This is exactly why a KB entry must carry a **confidence interval**,
never a point estimate.

---

## 3. Why it works — and it is not the model

Look at the two F1 columns. Same algorithm, same data. Raw sensors: **0.39–0.48,
useless.** Dimensionally-constructed quantities: **0.89–1.00.**

The unlock is **knowing the units**.

```
torque [N·m] × angular velocity [rad/s]  =  power [W]        ← forced by units
process [K]  −  air [K]                  =  ΔT [K]           ← same-unit difference
wear [min]   × torque [N·m]              =  strain [min·N·m] ← forced by units
```

You do not *search* for these. You *derive* them. Dimensional analysis collapses
the hypothesis space from "all functions of five variables" to "the handful of
dimensionally coherent combinations" — which is what makes physics discovery
tractable rather than a symbolic-regression fishing expedition.

**And units are free in a real plant**: OPC-UA tag metadata, instrument ranges,
and P&ID line lists all carry them.

### 3.1 Pipeline

```
sensor tags + units (OPC-UA metadata)
        │
        ▼
  DIMENSIONAL CONSTRUCTION            discovery/dimensional.py
  enumerate unit-coherent combinations
  (same-unit differences, products, ratios, π-groups)
        │
        ▼
  BOUNDARY ESTIMATION                 discovery/threshold.py
  shape-constrained / monotonic fit
  bracket θ, profile-likelihood CI
        │
        ▼
  CANDIDATE KB ENTRY
  {expr, θ̂, CI, support n, F1, provenance}
        │
        ▼
  ┌──────────────────┐
  │ ENGINEER REVIEW  │  ← the only point requiring judgment
  └──────────────────┘
        │ confirmed
        ▼
  VERSIONED KB ENTRY  (author, date, evidence, supersedes)
        │
        ▼
  SHADOW MODE → cut over → Gate 3 monitors it forever
```

---

## 4. Split discovery from evaluation

```
OFFLINE — slow, uncertain, heavy       ONLINE — fast, exact, auditable
  dimensional construction               margin evaluation
  boundary estimation + CI               0.22 µs, arithmetic
  hierarchical pooling across fleet      composable, verifiable
           │                                      ▲
           └──► candidate ──► ENGINEER ───────────┘
                              confirms, KB versions
```

This is strictly better than the conventional arrangement because it places the
**epistemically hard** question (*what is the rule?*) offline where uncertainty is
tolerable, keeps the **operationally critical** question (*how close are we?*)
online where it must be exact, and puts the human at the single point where
judgment is genuinely required.

---

## 5. Uncertain thresholds change nothing structurally

When θ is estimated rather than given, θ carries a CI, so the **margin becomes an
interval** — which is *already* the representation ([06-RELIABILITY.md](06-RELIABILITY.md) §2.1).

| Component | Change required |
|---|---|
| KB entry | gains `theta_ci` field |
| Margin evaluation | already interval-valued |
| Alerting | already three-state; wider θ → more ABSTAIN |
| Forecast | conformal calibration on the interval |
| Rollup | `min` over intervals, still associative |
| Copilot, planner, verifier | **none** |

Graceful degradation into the uncertain world, not a collapse. This is the
strongest property of the design.

---

## 6. This inverts the data-disadvantage argument

The objection: *incumbent vendors have orders of magnitude more data.* True — but
it assumes data must become **model weights**, which decay, need retraining, and
cannot be inspected.

Make data become **confirmed KB entries** instead, and it is permanent,
auditable, transferable, and **compounding**:

```
global prior  →  asset class  →  site  →  individual asset
```

A threshold learned across 200 machines of one class becomes a hierarchical prior
for machine 201, then updates on local data. That is data efficiency by pooling —
and it is precisely the "cross-line learning" capability, with real statistical
machinery underneath rather than a slogan.

> You do not need more data than the incumbent. You need a representation in
> which each observation contributes **permanently** rather than being amortised
> into a weight matrix that is discarded at the next retrain.

---

## 7. Honest limits

- **Cold start is real.** Discovery needs failures. A new asset class with zero
  failure history yields nothing, and hierarchical priors help only once a
  sibling exists.
- **The F1 figures above are in-sample.** Legitimate for *bracketing a
  deterministic boundary* — the interval [max-negative, min-positive] is a valid
  estimate — but productisation requires cross-validated brackets and proper
  confidence intervals, not point estimates. This is specified, not yet built.
- **Not every mode has a closed form.** Spectral signatures will not reduce to
  `f(x) > θ`. Those need learned health indicators; the margin abstraction then
  applies to the learned indicator, which is standard PHM. The architecture
  survives; the exactness does not.
- **Engineer confirmation costs the scarcest resource in a plant.** Active
  learning ranks candidates by information gain, but does not make it free.

---

**Next:** [09-STREAMING.md](09-STREAMING.md) — real-time inference and forecasting.
