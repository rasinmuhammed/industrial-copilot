# 11 — Scaling to 1,000 Factories

> 1,000 factories × ~2,000 machines × 1 Hz ≈ **2 million samples/second**,
> ~173 billion samples/day. Nothing queries that per question.

The brief does not require building this. It requires justifying how the
architecture evolves. This document is that justification.

---

## 1. The load-bearing property: margins compose, probabilities do not

`min()` is associative. The worst approach to a boundary over a window is the
`min` of the worst approaches of its sub-windows. Tiles therefore merge
**losslessly**:

```
MARGIN  —  min() is associative, so tiles merge without loss
 raw samples      1 s        1 min      1 h       1 day       query
 +412 W  ┐
 +180 W  ├─min─► +180 W ─► +96 W ─► −58 W ─► −58 W ─► answered from tile
 +901 W  ┘

PROBABILITY  —  mean() is not a probability; the event is gone by level 2
 0.02  ┐
 0.71  ├─mean─► 0.25  ─► 0.09  ─► 0.04  ─► 0.03  ─► ✗ must rescan raw
 0.01  ┘
```

*"How close did this machine come to overload last quarter?"* is one lookup
against a daily tile. The 0.71 spike in the lower track is unrecoverable — to get
it back you must query 173 billion rows a day, forever.

**The choice of aggregate is the scale decision.** Everything else follows.

---

## 2. Target topology

```
EDGE  (per cell / line)
  PLC · OPC-UA → edge agent
    · unit-tagged normalisation
    · derived quantities + interval margins   0.44 µs, stateless
    · invariant evaluation (Gate 2)
    · rule predicates
    emits:  1 s tiles {min margin, mean, n, quality}  +  crossing EVENTS
    buffers locally on link loss
         │  MQTT / Kafka — margins and events, never raw
         ▼
REGIONAL STREAM
  Kafka → Flink
    · windowed rules, persistence, debounce
    · trajectory forecast (first passage)
    · watermarking for late/out-of-order
         ├── alerts topic
         ├── rollups  → ClickHouse / Timescale   (hot, sharded by tenant)
         └── raw      → Iceberg / Parquet        (cold, rarely read)
         ▼
QUERY TIER
  semantic layer + IR compiler
    hot rollups by default; cold raw only when explicitly requested
         ▼
COPILOT TIER  (stateless, horizontally scaled)
  router → validated plan → executor → PCN verifier → answer
```

---

## 3. Why each layer holds

### 3.1 Computation moves to the edge, for free

Margin evaluation is **stateless O(1) arithmetic** on the current sample — no
window, no join, no model artifact. It runs on a gateway in microseconds.

Measured (`make bench`):

| | rate | headroom vs one 2,000-machine site |
|---|---:|---:|
| scalar margins | 4.9 M events/sec/core | **2,469×** |
| interval margins (production path) | 2.3 M events/sec/core | **1,144×** |

> **The entire 1,000-factory fleet — 2.0 M events/sec — requires 0.9 CPU cores
> for margin evaluation.**

That is the whole reasoning core for two million machines, on less than one core.
It is not an optimisation result; it is what happens when there is no model to
evaluate.

There is **no model to distribute** to 1,000 sites, no version skew to manage,
and no retraining cadence. Contrast a per-site ML model: 1,000 artifacts,
1,000 drift monitors, 1,000 retraining schedules.

### 3.2 Only margins and events cross the WAN

Raw stays at the edge or in cold object storage. What travels is 1-second tiles
plus boundary-crossing events — orders of magnitude less than raw telemetry, and
sized by *information content* rather than sample rate.

### 3.3 The Analysis IR is the portability layer

One plan object, three compilers:

| Backend | Scope |
|---|---|
| DuckDB | laptop, single site |
| ClickHouse / Timescale | fleet, hot rollups |
| Flink SQL | live stream |

Prompts, evals, and the agent never change when the backend does. Had the model
emitted SQL, you would re-tune prompts per dialect per backend — which is where
text-to-SQL copilots die in production.

### 3.4 Plans cache across tenants

*"Which machines are closest to their overstrain limit?"* is the **same plan** for
every factory. Only the filter differs, and the filter comes from session state,
not the plan.

**One shared plan cache serves 1,000 sites.** LLM cost is therefore driven by the
diversity of *question shapes*, not by fleet size — sublinear, and in practice
close to flat.

### 3.5 The knowledge base shards and inherits

```
global  →  asset class  →  site  →  individual asset
```

A threshold confirmed at one site becomes a prior elsewhere, adapted to local
duty cycle and ambient. Every entry carries provenance, author, and version. This
is the cross-line learning mechanism and the compounding asset
([08-DISCOVERY.md](08-DISCOVERY.md) §6).

---

## 4. What must be built that is not built

Honest inventory. Each is understood work, not research.

| Gap | Required | Difficulty |
|---|---|---|
| **Temporal operators** | Windowed aggregates, rate-of-change, persistence ("true for N samples"). AI4I modes are per-sample; real modes are not. | Medium — CEP-shaped |
| **Multi-tenancy** | Row-level isolation, per-tenant KB scoping, quota | Medium |
| **Rollup materialisation** | Tile tree with incremental merge + backfill | Medium |
| **Late/out-of-order** | Watermarks, tile revision on late arrival | Medium |
| **Per-site calibration** | Unit and offset reconciliation across identical assets | Medium |
| **Alarm rationalisation** | ISA-18.2 prioritisation, flood suppression, shelving | Medium |
| **Edge deployment** | Agent packaging, offline buffering, OTA KB updates | Medium |
| **Spectral modes** | Learned health indicators feeding the margin abstraction | Hard |
| **RBAC + audit** | Who asked what, who confirmed which threshold | Medium |

---

## 5. Cost shape at scale

| Component | Scaling | Note |
|---|---|---|
| Margin evaluation | linear in samples, **negligible** | 2 M/s needs < 1 core |
| Storage — tiles | linear, small | ~seconds-resolution scalars |
| Storage — raw | linear, large | cold object storage, rarely read |
| Query | **sublinear** | tiles, not raw |
| LLM inference | **sublinear** | shared plan cache; tail only |
| KB maintenance | linear in *asset classes*, not assets | the key economy |

The last row matters most. Conventional PdM cost scales with the number of
**assets** (a model per asset, retrained). Ours scales with the number of **asset
classes** (a rule set per class, confirmed once and inherited). At 2 million
machines across a few hundred classes, that is a difference of four orders of
magnitude.

---

## 6. What we do not claim

- These are design arguments, not measurements. Nothing here has run at 2 M
  events/sec across real infrastructure.
- The edge agent is specified, not built.
- Multi-tenancy is specified, not built.
- The prototype demonstrates the *properties* that make the evolution credible —
  stateless margins, portable IR, composable rollups — not the evolution itself.

---

**Next:** [12-ASSUMPTIONS.md](12-ASSUMPTIONS.md)
