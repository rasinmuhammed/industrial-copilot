# 07 - Context Engineering, Routing, and Latency

The brief names **latency** and **context engineering** as explicit criteria.
Both are properties of this path, not of the model chosen to walk it.

---

## 1. Flat token budget

The naive follow-up implementation appends every turn to a growing transcript.
Cost and latency climb turn over turn, and the model starts resolving pronouns
against stale context. We compress the conversation into a **typed object**:

```python
class SessionState(BaseModel):
    focus:         Focus | None        # M-03, or a cohort, or None
    filters:       list[Filter]        # sticky scope
    last_plan:     AnalysisPlan | None # for "what about X instead?"
    last_evidence: str | None          # handle → "show me those rows"
    metrics_seen:  list[str]
    turn_digest:   deque[str]          # maxlen=6, one line per turn
    synthetic_used: set[str]           # which overlays the answer relied on
```

**Turn 40 costs the same as turn 2.** That is the property that matters when this
runs across a plant rather than in a demo.

### 1.1 Prompt assembly - byte-stable prefix

Ordered so everything cacheable sits above everything volatile:

```
[ cache breakpoint ────────────────────────────────── ]
  1. system prompt            ~600 tok   never changes
  2. semantic layer digest   ~1400 tok   changes on schema version
  3. operator catalogue      ~1000 tok   changes on registry version
[ ─────────────────────────────────────────────────── ]
  4. session state             ~200 tok   volatile
  5. question                   ~40 tok   volatile
```

Nothing dynamic appears above a breakpoint. Cache reads cost 10 % of base input,
so the ~3,000-token prefix costs ~$0.0003 rather than $0.003.

### 1.2 Follow-ups resolve structurally

*"What about the H variants?"* is not a new question. It **mutates one filter on
the previous plan**:

```
previous:  {op: rate, group_by: [shift], filters: [{product_type = L}]}
mutation:  filters[0].value: L → H
```

No re-planning, no model call - it usually lands on the grammar tier. This is
both a latency win and a correctness win: the analysis stays identical, so the
comparison is valid.

### 1.3 Anti-poisoning

A wrong resolution in turn 3 must not silently propagate. Two defenses:

- Session state is **displayed** in the CLI/UI and directly editable.
- Every answer **restates its resolved scope**: *"For machine M-03, L variant,
  since 8 Jan (synthetic timeline)…"* - so a bad resolution is visible
  immediately rather than three turns later.

---

## 2. Three-tier router

```
question
   │
   ├─ Tier 0  PLAN CACHE ─────────────── hit? → plan          ~0 ms
   │          key = sha256(normalised question + session shape)
   │          normalisation: lowercase, synonym→canonical,
   │          entity→placeholder. Tenant-independent.
   │
   ├─ Tier 1  GRAMMAR PLANNER ────────── match? → plan        ~1 ms
   │          deterministic patterns over the semantic layer
   │          covers the common plant question vocabulary
   │          emits confidence; low confidence escalates
   │
   └─ Tier 2  LLM PLANNER ────────────── plan                ~400 ms
              structured output against the plan schema
              cached prefix; one repair attempt on validation failure
```

### 2.1 Why the cache is tenant-independent

`"which machines are closest to their overstrain limit?"` normalises to the same
key for every factory. Only the *filter* differs, and the filter comes from
session state, not the plan. **One shared plan cache serves 1,000 sites** - which
is what keeps inference cost sublinear in fleet size.

### 2.2 Speculative execution

While the Tier-2 model streams, the router **speculatively executes** the Tier-1
grammar guess. If the returned plan matches, the evidence bundle is already
warm - the model's latency is hidden behind work that was going to happen anyway.
Cost is one wasted sub-10 ms query on a miss.

### 2.3 Escalation policy

| Condition | Action |
|---|---|
| Grammar confidence ≥ threshold | use grammar plan |
| Grammar confidence below threshold | escalate to LLM |
| Plan fails validation | repair ×1, then refuse |
| Question is multi-part | decompose into a plan list |
| No op fits | `sql_explore` (labelled exploratory) or refuse |

**Refusal is a first-class outcome.** *"I can't answer that from this data"* is
correct for RNF root cause, and for anything requiring data the dataset lacks.

---

## 3. Cost model

Per fully-LLM question on Haiku 4.5 ($1/$5 per MTok; cache reads $0.10):

| Component | Tokens | Cost |
|---|---:|---:|
| Cached prefix | 3,000 @ cache-read | $0.0003 |
| Question + session | 300 in | $0.0003 |
| Plan output | 150 out | $0.0008 |
| Narration | 800 in / 250 out | $0.0021 |
| **Total** | | **~$0.0034** |

**≈ 300 questions per dollar.** A 50-question demo costs ~$0.17. Ten full eval
iterations of a 40-question golden set: ~$1.40, or ~$0.70 via the Batch API.

Zero-cost paths:

| Tier | Cost | Notes |
|---|---|---|
| Deterministic (default) | **$0** | No account. Fully evaluable. |
| Cerebras free tier | **$0** | 30 RPM, 60k TPM, 1M tok/day, native JSON schema |
| Ollama local | **$0** | Air-gapped |

---

**Next:** [08-DISCOVERY.md](08-DISCOVERY.md) - learning the rules instead of being given them.
