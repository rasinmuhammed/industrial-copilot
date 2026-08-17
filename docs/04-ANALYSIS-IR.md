# 04 - Analysis IR and the Operator Registry

> The model emits a **validated data structure**, never SQL and never prose
> containing numbers. This document is the contract.

---

## 1. Why an IR instead of text-to-SQL

| Property | Text-to-SQL | Analysis IR |
|---|---|---|
| Hallucinated column | Runtime error, or silently wrong join | **Rejected at validation** |
| Failure mode | Wrong answer, looks right | "No plan matched" - *detectable* |
| Backend portability | Re-tune prompts per dialect | Recompile the same object |
| Cacheable across tenants | No (filters embedded in text) | Yes (plan ≠ filter) |
| Auditable | Diff SQL strings | Diff structured plans |
| Measured accuracy | **~72–76 %** (BIRD SOTA) | Validation is binary |

The decisive point is the *failure mode*. Text-to-SQL fails by producing
plausible wrong answers. A closed registry fails by producing **no answer**,
which is recoverable - escalate, ask a clarifying question, or use the guarded
escape hatch.

---

## 2. Plan schema

```python
class Filter(BaseModel):
    field: str                       # must ∈ semantic_layer.metrics|dimensions
    op: Literal["=","!=","<","<=",">",">=","in","between","is_null"]
    value: float | int | str | list

class Cohort(BaseModel):
    name: str
    filters: list[Filter] = []

class Binning(BaseModel):
    field: str
    method: Literal["quantile","width","explicit"] = "quantile"
    bins: int | list[float] = 5

class AnalysisPlan(BaseModel):
    op: OpName                       # closed enum - see §3
    cohorts: list[Cohort] = []
    filters: list[Filter] = []       # applied to all cohorts
    metrics: list[str] = []          # must ∈ semantic_layer.metrics
    dimensions: list[str] = []       # must ∈ semantic_layer.dimensions
    group_by: list[str] = []
    bin: Binning | None = None
    time_grain: Literal["hour","shift","day"] | None = None
    limit: int = Field(50, le=500)
    confidence: float = Field(0.95, ge=0.5, le=0.999)
    effect_size: Literal["cohens_d","rate_ratio","risk_diff"] | None = None
    params: dict = {}                # op-specific, schema-checked per op
    # meta
    verify_premise: bool = True
    explain: bool = True
```

### 2.1 Validation pipeline

Order matters - cheapest rejection first.

1. **Structural** - Pydantic types and enums.
2. **Vocabulary** - every `field`/`metric`/`dimension` exists in the semantic
   layer. *This is the hallucination barrier.*
3. **Dimensional** - comparisons and arithmetic are unit-coherent. `torque_nm < 8.6`
   with a `K` literal is rejected. Prevents the °C-vs-K class of bug.
4. **Cardinality** - `group_by` on a high-cardinality field without a limit is rejected.
5. **Statistical viability** - an op requiring cohort comparison with an empty
   cohort is rejected before execution.
6. **Op-specific** - `params` validated against the op's own schema.

On failure: **one** repair attempt with the validation error fed back. If it
fails again, the system refuses and explains what it could not resolve. It never
guesses.

### 2.2 Worked example

> *"Compare operating conditions of machines that failed versus those that didn't."*

```json
{
  "op": "compare",
  "cohorts": [
    {"name": "failed",  "filters": [{"field": "failure", "op": "=", "value": 1}]},
    {"name": "healthy", "filters": [{"field": "failure", "op": "=", "value": 0}]}
  ],
  "metrics": ["torque_nm","rotational_speed_rpm","tool_wear_min",
              "temp_delta_k","power_w"],
  "effect_size": "cohens_d",
  "confidence": 0.95
}
```

Note what is **absent**: no table name, no join, no SQL, no aggregate function.
The op knows how to compare; the plan says only *what*.

---

## 3. Operator registry

Twelve operators. Closed set. Each is independently unit-tested against
`evals/reference.py`, an implementation written without reference to the engine.

| Op | Answers | Key outputs |
|---|---|---|
| `describe` | "What are conditions on M-03?" | n, mean, sd, min, p25, median, p75, max per metric, with units |
| `rate` | "Failure rate by variant?" | count, rate, **Wilson CI**, per group |
| `compare` | "Failed vs healthy?" | per-metric means, Δ, Cohen's *d*, CI, **collinearity warnings** |
| `trend` | "Is torque climbing?" | slope + CI per grain, changepoints (CUSUM) |
| `drivers` | "What separates failures?" | ranked std. mean diff + mutual information, **confounder flags** |
| `root_cause` | "Why did this fail?" | firing modes, margin to *each* boundary, crossing point |
| `counterfactual` | "If torque dropped 5 Nm?" | mode firings before/after, margin deltas |
| `envelope` | "What's the safe window?" | feasible region in a 2-D control plane; **minimal corrective change** |
| `forecast` | "When do we cross?" | E[cycles], inverse-Gaussian interval, mode at risk |
| `records` | "Show me those rows." | capped raw rows, drill-down evidence |
| `data_quality` | "Can I trust this?" | orphan failures, RNF roll-up, KB drift, invariant status |
| `sql_explore` | anything unanticipated | **guarded escape hatch** - see §5 |

### 3.1 `root_cause` - the flagship

Input: a row, a cohort, or a hypothetical operating point.

Output per deterministic mode:

```
mode        fired   margin              crossing point
HDF         no      +2.4 K, +171 rpm    -
PWF         no      +2740 W below ceil  -
OSF         YES     −1433 min·N·m       crossed at wear = 189 min
TWF         window  wear 214 ∈ [200,240]  P(fail) ≈ 5.4 % per cycle
```

Contract:
- Reports **every** firing mode, never one (23 rows fire ≥2 modes).
- Reports the **crossing point** where the trajectory is known.
- TWF returns a **probability**, never a certainty - it is stochastic.
- RNF is **never** attributed; it is by definition unpredictable.
- Orphan failures return `cause_undetermined` with the margin evidence. This is a
  *verified correct* answer (see [01-DATASET.md](01-DATASET.md) §4.1), not a
  fallback.

### 3.2 `envelope` - prescription by inversion

Because constraints are analytic, solve for the control variable rather than
searching:

```
Given:  wear = 214 min, type = L, current torque = 58.1 N·m, rpm = 1412
Find:   minimal Δ restoring all margins ≥ 0

→ torque ≤ 51.4 N·m       (overstrain limit 11,000 / 214)
→ Δtorque = −6.7 N·m
→ resulting power = 7,595 W  ✓ inside [3500, 9000]
→ all five margins positive
```

Also returns the **feasible region** in the rpm × torque plane, which is what the
Envelope Explorer renders. This is a computed region, not a classifier decision
surface - no other approach can draw it correctly.

### 3.3 `compare` and `drivers` - automatic confounder detection

`r(rpm, torque) = −0.8750` in this dataset. Therefore **every** rpm analysis is
confounded by torque. Both ops:

1. Compute the requested separation statistics.
2. Compute pairwise correlation among reported metrics.
3. Emit a `collinearity` warning wherever `|r| > 0.7` between a reported driver
   and another metric that also separates the cohorts.
4. The narrator is **required** to surface the warning.

Example emitted warning:

> `rotational_speed_rpm` and `torque_nm` are strongly inversely coupled
> (r = −0.875). Differences attributed to speed may be driven by torque.

Almost no copilot does this, and this dataset punishes its absence.

### 3.4 Premise verification (Gate 1)

When a question embeds a comparative claim - *"why are we seeing **more** failures
at high rpm?"* - the planner sets `verify_premise: true`. The engine tests the
claim **before** answering:

```
premise:  failure rate increases with rotational speed
test:     rate by rpm quintile
result:   12.17 % → 1.60 % → 0.60 % → 0.45 % → 2.24 %
verdict:  REFUTED - U-shaped; 5.4× higher at LOW speed
```

The narrator must lead with the refutation, then explain the true mechanism. A
system that answers the question as asked confabulates. This is the flagship eval
case.

---

## 4. Statistical contracts

Non-negotiable, enforced in code:

| Rule | Rationale |
|---|---|
| Every rate carries a **Wilson** interval | Normal approximation is wrong at low *p*; base rate here is 3.39 % |
| **Refuse** a point estimate when CI half-width > 50 % of the estimate | H-variant filters reach single-digit *n* quickly |
| Every effect size carries a CI | A *d* without an interval is not a finding |
| `n` accompanies every aggregate | Non-negotiable for engineer trust |
| No causal language from `compare`/`drivers` | These are associational ops; wording is constrained |
| Synthetic dimensions tagged in-answer | `ts`, `machine_id`, `shift` are overlays |

---

## 5. `sql_explore` - the guarded escape hatch

A closed registry has a coverage ceiling. Rather than pretend otherwise:

**Fires only when** no structured op matches with sufficient confidence.

**Constraints:**
- Read-only view; no DDL, no writes, no attach.
- Row budget (10k) and wall-clock budget (2 s), enforced.
- Output passes through the **same** PCN verifier.
- Answer is **labelled** `exploratory - not a certified analysis`.
- Query is logged with the question for registry-gap review.

This closes the expressiveness gap without weakening the hot path, and the log
becomes the backlog for new first-class ops.

---

## 6. Backend portability

The same plan compiles to three targets. No prompt or eval changes.

| Target | Scope | Compiler |
|---|---|---|
| DuckDB | laptop, single site | `ops/*.py` (built) |
| ClickHouse / Timescale | fleet, hot rollups | recompile aggregates onto tile tables |
| Flink SQL | live stream | windowed predicates over the event stream |

Because the model never wrote SQL, changing the backend never touches the agent.
This is where text-to-SQL copilots die in production.

---

**Next:** [05-GROUNDING.md](05-GROUNDING.md) - how a number reaches the engineer.
