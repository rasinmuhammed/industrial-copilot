# 03 - Architecture

---

## 1. Layered view

Mapped onto the four-layer industrial reference model (plant → ingest → reasoning
→ action), so the prototype maps directly onto a production platform.

```
┌──────────────────────────────────────────────────────────────────────────┐
│ L1  PLANT                                                                 │
│     PLC/SCADA · MES · Historian · IoT          [AI4I CSV replay]          │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ raw samples
┌────────────────────────────────▼─────────────────────────────────────────┐
│ L2  INGEST & NORMALISE                                    copilot/ingest │
│     · schema normalisation (BOM, unit suffixes)                          │
│     · dimensional typing - units are first-class, mismatch is rejected   │
│     · derived physics:  ΔT · power · overstrain                          │
│     · MARGIN COMPUTATION - 5 signed scalars, stateless, 0.22 µs          │
│     · invariant evaluation (I1–I4)                                       │
│     · synthetic overlays: ts, machine_id, shift        [flagged SYNTHETIC]│
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ margins + events
┌────────────────────────────────▼─────────────────────────────────────────┐
│ L3  REASONING CORE                                                        │
│  ┌────────────────┐ ┌────────────────┐ ┌──────────────┐ ┌─────────────┐  │
│  │ Knowledge base │ │ Margin         │ │ Trajectory   │ │ Setpoint    │  │
│  │ rules + prov.  │ │ evaluator      │ │ forecaster   │ │ solver      │  │
│  │ knowledge/     │ │ DIAGNOSIS      │ │ ANALYSIS     │ │ PRESCRIPTION│  │
│  └────────────────┘ └────────────────┘ └──────────────┘ └─────────────┘  │
│  ┌───────────────────────────────┐ ┌───────────────────────────────────┐ │
│  │ Analysis IR                   │ │ Evidence bundle                   │ │
│  │ validate · compile · execute  │ │ typed rows + full provenance      │ │
│  │ DuckDB │ ClickHouse │ Flink   │ │ every figure: unit, filter, n     │ │
│  └───────────────────────────────┘ └───────────────────────────────────┘ │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ evidence
┌────────────────────────────────▼─────────────────────────────────────────┐
│ L4  ACTION                                                                │
│     Copilot chat · Envelope explorer · Alert stream · [PLC writeback]     │
└──────────────────────────────────────────────────────────────────────────┘

Data flows up.  Decisions flow down.  Dashed/bracketed = roadmap, not prototype.
```

**One core, two consumers.** Chat and closed-loop control run *identical*
arithmetic. Only delivery and the approval gate differ. That is why the copilot
is a legitimate prototype of a control system rather than a reporting tool
bolted on beside one.

---

## 2. Request lifecycle

Where latency and hallucination are controlled. Neither depends on model quality.

```
  Engineer's question
          │
          ▼
  ┌───────────────────┐   ┌──────────────────────────┐
  │ Reference         │◄──│ Typed session state      │
  │ resolver          │   │ focus · filters · last   │
  │ "that machine"    │   │ evidence · turn digest   │
  │   → M-03          │   │ FLAT TOKEN BUDGET        │
  └─────────┬─────────┘   └──────────────────────────┘
            ▼
  ┌─────────────────────────────────────────────────────┐
  │ THREE-TIER ROUTER - escalate only on miss           │
  │  ┌───────────┐ →  ┌───────────────┐ → ┌───────────┐ │
  │  │Plan cache │    │Grammar planner│   │LLM planner│ │
  │  │  ~0 ms    │    │    ~1 ms      │   │  ~400 ms  │ │
  │  └───────────┘    └───────────────┘   └───────────┘ │
  └─────────────────────────┬───────────────────────────┘
                            ▼
                  ┌───────────────────┐
                  │ Analysis Plan     │
                  │      (JSON)       │
                  └─────────┬─────────┘
                            ▼
  ┌─────────────────────────────────────────┐      ┌──────────────────────┐
  │ Schema validator                        │◄─────│ Semantic layer + KB  │
  │ every field ∈ semantic layer            │      │ the ONLY permitted   │
  │ units dimensionally coherent            │      │ vocabulary           │
  └─────────┬───────────────────────┬───────┘      └──────────────────────┘
            │ valid                 │ invalid → repair ×1 → else refuse
            ▼
  ┌─────────────────────────────────────────┐
  │ Typed executor → DuckDB        < 10 ms  │
  │ → evidence bundle + provenance          │
  └─────────┬───────────────────────────────┘
            ▼
  ┌─────────────────────────────────────────┐
  │ Narrator - prose with {{slot}} refs     │
  │ NEVER writes a digit                    │
  └─────────┬───────────────────────────────┘
            ▼
  ┌─────────────────────────────────────────┐
  │ PCN verifier - fail-closed              │
  │ bare numeral → REJECT → regenerate      │
  └─────────┬───────────────────────────────┘
            ▼
     Verified answer + evidence + replay handle
```

### 2.1 Latency budget

| Tier | Path | Target p50 | Coverage |
|---|---|---|---|
| 0 | plan cache hit | < 1 ms | repeat questions |
| 1 | grammar planner | < 5 ms | common plant vocabulary |
| 2 | LLM planner (cached prefix) | < 600 ms | novel questions |
| - | execution | < 10 ms | all |
| - | narration (template) | < 1 ms | offline mode |
| - | narration (LLM) | < 500 ms | online mode |

Measured tier distribution on the golden set is reported by
`make eval` - see [10-EVALS.md](10-EVALS.md).

---

## 3. The four gates

A copilot that cannot doubt itself will eventually be confidently wrong.

| Gate | Doubt | Mechanism | Doc |
|---|---|---|---|
| 1 | Is the question's **premise** true? | Premise verification before answering | [04](04-ANALYSIS-IR.md) |
| 2 | Is the **instrument** honest? | Physics invariant monitors | [06](06-RELIABILITY.md) |
| 3 | Is the **rule** still valid? | KB calibration monitor | [06](06-RELIABILITY.md) |
| 4 | Is every **number** sourced? | Proof-Carrying Numbers | [05](05-GROUNDING.md) |

Question → input → knowledge → output. A complete epistemic perimeter.

---

## 4. Repository layout

```
industrial-copilot/
├── README.md                     deliverable: architecture, setup, run
├── Makefile                      build · verify · eval · serve · demo
├── pyproject.toml
├── data/
│   ├── ai4i2020.csv
│   └── warehouse.duckdb          generated
├── copilot/
│   ├── config.py                 all tunables; nothing else reads env
│   ├── ingest.py                 CSV → DuckDB + physics + margins
│   ├── knowledge/
│   │   ├── failure_modes.yaml    the SOP: documented rules + provenance
│   │   ├── semantic_layer.yaml   the only permitted vocabulary
│   │   └── __init__.py           cached loaders
│   ├── ir.py                     Pydantic Analysis Plan + validation
│   ├── ops/
│   │   ├── registry.py           op dispatch, signatures
│   │   ├── describe.py  rate.py  compare.py  trend.py
│   │   ├── drivers.py   root_cause.py  counterfactual.py
│   │   ├── envelope.py  forecast.py    records.py  data_quality.py
│   │   └── sql_explore.py        guarded escape hatch
│   ├── planner/
│   │   ├── router.py             3-tier dispatch + speculative execution
│   │   ├── grammar.py            deterministic NL → plan
│   │   ├── llm.py                Claude / Cerebras / Ollama adapters
│   │   └── cache.py              normalised-key plan cache
│   ├── session.py                typed conversational state
│   ├── evidence.py               evidence bundle + slot IDs + provenance
│   ├── narrate.py                slot-only narration
│   ├── verify.py                 PCN verifier (fail-closed)
│   ├── reliability/
│   │   ├── invariants.py         Gate 2
│   │   ├── kb_monitor.py         Gate 3
│   │   └── intervals.py          interval-valued margins → ABSTAIN
│   ├── stream.py                 replay + online scorer + alerts
│   ├── engine.py                 orchestrator
│   ├── api.py                    FastAPI + SSE
│   └── cli.py                    terminal chat
├── discovery/
│   ├── dimensional.py            unit-coherent feature construction
│   ├── threshold.py              boundary estimation + CIs
│   └── audit.py                  KB vs data reconciliation
├── evals/
│   ├── golden.yaml               questions + programmatic assertions
│   ├── reference.py              INDEPENDENT reference implementation
│   ├── run_evals.py
│   └── reports/
├── scripts/
│   ├── verify_dataset.py         reproduces every figure in doc 01
│   ├── discover_rules.py         re-derives thresholds from data alone
│   └── bench.py                  throughput benchmark
├── tests/
└── docs/                         this documentation set
```

---

## 5. Technology choices

| Choice | Why | Alternative rejected |
|---|---|---|
| **DuckDB** | In-process OLAP, zero setup, columnar, ~ms on 10k rows; same SQL dialect family as the fleet-scale target | Postgres (server dependency), pandas (no query planner) |
| **Pydantic v2** | Plan validation *is* the anti-hallucination mechanism; needs to be declarative and fast | Hand-rolled validation (unauditable) |
| **numpy** | Vectorised margins at 419 M samples/sec | Pure Python (20× slower) |
| **FastAPI + SSE** | Streaming alerts and token streaming over one protocol | WebSockets (unneeded bidirectionality) |
| **YAML knowledge base** | Rules must be reviewable and diffable by an engineer, not buried in code | Python constants (not reviewable) |
| **No agent framework** | LangChain/LlamaIndex add indirection, latency, and hidden prompts. The brief explicitly discourages framework stacking. | LangChain |
| **No vector DB** | 10k structured rows. Retrieval is a WHERE clause. | Chroma/FAISS |

### 5.1 Model provider abstraction

One interface, three implementations, selected by config:

| Provider | Cost | Use |
|---|---|---|
| **None (deterministic)** | $0 | Default. Grammar planner + template narrator. Fully evaluable with zero credentials. |
| **Cerebras** | $0 (free tier: 30 RPM, 60k TPM, 1M tok/day) | Native JSON-schema structured output - ideal for the planner |
| **Anthropic** | ~$0.0034/question (Haiku 4.5, cached prefix) | Best phrasing; used to measure real latency percentiles |
| **Ollama** | $0, local | Air-gapped operation |

Graceful degradation to deterministic is a **systems-engineering feature**, not a
fallback: the system degrades to *less fluent*, never to *wrong*.

---

## 6. Data contracts

### 6.1 `observations` (materialised at ingest)

```
udi                        INTEGER
product_id                 VARCHAR
product_type               VARCHAR       -- L | M | H
air_temperature_k          DOUBLE  [K]
process_temperature_k      DOUBLE  [K]
rotational_speed_rpm       DOUBLE  [rpm]
torque_nm                  DOUBLE  [N·m]
tool_wear_min              DOUBLE  [min]
machine_failure            TINYINT
twf hdf pwf osf rnf        TINYINT

-- derived physics
temp_delta_k               DOUBLE  [K]
power_w                    DOUBLE  [W]
overstrain_min_nm          DOUBLE  [min·N·m]
osf_threshold_min_nm       DOUBLE  [min·N·m]

-- margins: signed distance to boundary; negative == violated
temp_delta_margin_k        DOUBLE  [K]
speed_margin_rpm           DOUBLE  [rpm]
power_low_margin_w         DOUBLE  [W]
power_high_margin_w        DOUBLE  [W]
overstrain_margin_min_nm   DOUBLE  [min·N·m]
wear_to_window_min         DOUBLE  [min]
worst_normalised_margin    DOUBLE  [-]

-- rule firings, RECOMPUTED from the KB, never copied from labels
hdf_rule pwf_rule osf_rule twf_window  BOOLEAN

-- synthetic overlays  [SYNTHETIC]
ts                         TIMESTAMP
machine_id                 VARCHAR
shift                      VARCHAR
```

Rule columns are recomputed rather than copied so that **divergence between rules
and labels is measurable** - that is what the KB calibration monitor consumes.

### 6.2 Evidence bundle

```python
EvidenceBundle:
    plan_hash:     str            # replay handle
    kb_version:    str
    data_version:  str            # fingerprint; invalidates answer cache
    slots:         dict[str, Slot]
    rows:          list[dict]     # capped at max_rows_returned
    provenance:    Provenance     # filters, row counts, SQL, elapsed
    warnings:      list[Warning]  # small-n, collinearity, SYNTHETIC use
    abstained:     list[str]      # quantities withheld as unresolvable

Slot:
    id:      str        # "failed.torque_nm.mean"
    value:   float | int | str
    unit:    str
    n:       int
    ci:      tuple[float, float] | None
```

Every number the engineer ever sees is a `Slot`. There is no other path.

---

**Next:** [04-ANALYSIS-IR.md](04-ANALYSIS-IR.md) - the plan schema and every operator.
