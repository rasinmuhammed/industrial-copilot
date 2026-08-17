# 01 - Dataset Analysis

Every figure in this document was computed from `data/ai4i2020.csv` (10,000 rows).
Reproduce all of it with:

```bash
make verify
```

Source: [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

---

## 1. Raw schema

| Column | Clean name | Unit | Notes from the dataset page |
|---|---|---|---|
| `UDI` | `udi` | - | Row index 1..10000 |
| `Product ID` | `product_id` | - | Letter prefix = quality variant |
| `Type` | `product_type` | L/M/H | Low 50 %, Medium 30 %, High 20 % |
| `Air temperature [K]` | `air_temperature_k` | K | Random walk, σ = 2 K around 300 K |
| `Process temperature [K]` | `process_temperature_k` | K | Air + 10 K, σ = 1 K |
| `Rotational speed [rpm]` | `rotational_speed_rpm` | rpm | Derived from a 2860 W power target + noise |
| `Torque [Nm]` | `torque_nm` | N·m | N(40, 10), truncated at 0 |
| `Tool wear [min]` | `tool_wear_min` | min | H/M/L add 5/3/2 min per process |
| `Machine failure` | `machine_failure` | 0/1 | Roll-up label |
| `TWF HDF PWF OSF RNF` | lowercase | 0/1 | Independent mode flags |

The published header carries a UTF-8 BOM and bracketed unit suffixes. Ingest
normalises both; units are retained in the semantic layer rather than discarded.

## 2. Class balance - verified

```
rows                10,000
product_type        L 6,000  |  M 2,997  |  H 1,003
machine_failure        339   (3.39 %)
TWF 46   HDF 115   PWF 95   OSF 98   RNF 19
```

## 3. Rule verification - the central result

Rules transcribed from the dataset page's *Additional Variable Information*,
then scored against the published labels. Derived quantities:

```
temp_delta_k      = process_temperature_k − air_temperature_k
power_w           = torque_nm × rotational_speed_rpm × 2π / 60
overstrain_min_nm = tool_wear_min × torque_nm
```

| Mode | Predicate | Fires | Labelled | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|
| HDF | `temp_delta_k < 8.6 AND rpm < 1380` | 115 | 115 | 115 | **0** | **0** |
| PWF | `power_w < 3500 OR power_w > 9000` | 95 | 95 | 95 | **0** | **0** |
| OSF | `overstrain > {L:11000, M:12000, H:13000}` | 98 | 98 | 98 | **0** | **0** |
| TWF | `tool_wear_min BETWEEN 200 AND 240` | 790 | 46 | 43 | 747 | 3 |

`HDF ∨ PWF ∨ OSF` vs `machine_failure`: **TP 287, FP 0, FN 52.**

Three deterministic modes are recovered **exactly** - zero false positives and
zero false negatives across 10,000 rows. TWF is genuinely stochastic: 43 of the
790 rows inside the documented window are labelled failures, an in-window rate of
**5.4 %**. Three TWF rows fall outside the window and are reported as such.

> **Design consequence.** Root-cause attribution is arithmetic, not
> classification. A probabilistic model here would be strictly worse: it
> replaces a computable quantity with an estimate and cannot invert to a
> setpoint.

## 4. Label-integrity findings *not* stated on the dataset page

### 4.1 Nine orphan failures

Nine rows carry `machine_failure = 1` with none of TWF/HDF/PWF/OSF set.

| UDI | type | ΔT | rpm | torque | wear | power | OSF margin | RNF |
|---:|:--:|---:|---:|---:|---:|---:|---:|:--:|
| 1438 | H | 11.1 | 1439 | 45.2 | 40 | 6811 | 11192 | 0 |
| 2750 | M | 9.5 | 1685 | 28.9 | 179 | 5099 | 6827 | 0 |
| 4045 | M | 9.0 | 1419 | 47.7 | 20 | 7088 | 11046 | 0 |
| 4685 | M | 8.2 | 1421 | 44.8 | 101 | 6667 | 7475 | 0 |
| 5537 | M | 9.5 | 1363 | 54.0 | 119 | 7708 | 5574 | 0 |
| 5942 | L | 10.1 | 1438 | 48.5 | 78 | 7303 | 7217 | 0 |
| 6479 | L | 9.3 | 1663 | 29.1 | 145 | 5068 | 6780 | 0 |
| 8507 | L | 11.2 | 1710 | 27.3 | 163 | 4889 | 6550 | 0 |
| 9016 | L | 10.9 | 1431 | 49.7 | 210 | 7448 | 563 | 0 |

**Tested for hidden structure and found none.** Mean rule-level worst margin is
**0.153** for orphans versus **0.180** for healthy rows - the same neighbourhood,
with orphans if anything slightly *safer* than a failure population should look.
None carries RNF.

Therefore *"cause cannot be determined"* is a **verified correct answer**, not an
evasion. The copilot reports these explicitly and excludes them from attribution.

### 4.2 RNF does not roll up

19 rows set `RNF = 1`; **only 1** also sets `machine_failure`. The published RNF
flag does not contribute to the failure label. All rates are computed against
`machine_failure`, with RNF reported separately and labelled as such.

### 4.3 Documentation/column mismatch on TWF

The dataset page describes 120 tool events (69 replacements, 51 failures). The
`TWF` column flags **46** rows. Every statement is computed from the column; the
discrepancy is surfaced as a `data_quality` finding.

## 5. Multi-mode failures

23 failures fire two or more modes simultaneously:

```
HDF only 106 | PWF only 80 | OSF only 78 | TWF only 43
PWF+OSF 11 | HDF+OSF 6 | HDF+PWF 3 | TWF+OSF 2 | TWF+PWF+OSF 1
none (orphans) 9
```

A single-label multiclass classifier is **structurally incapable** here - it must
pick one mode and is wrong on the rest by construction. The rule engine reports
every firing mode with its individual margin.

## 6. The brief's example question is a false premise

> *"Why are we seeing more failures at high rotational speeds?"*

| rpm band | n | failure rate | HDF | PWF | OSF | TWF |
|---|---:|---:|---:|---:|---:|---:|
| 1168–1405 | 1981 | **12.17 %** | 115 | 52 | 83 | 9 |
| 1405–1470 | 2003 | 1.60 % | 0 | 11 | 12 | 9 |
| 1470–1541 | 2009 | 0.60 % | 0 | 1 | 3 | 8 |
| 1541–1644 | 2002 | 0.45 % | 0 | 0 | 0 | 9 |
| 1644–2887 | 2005 | **2.24 %** | 0 | 31 | 0 | 11 |

Failures are **5.4× more common at low speed**. The relationship is U-shaped, not
monotonic.

**The real high-speed mechanism** is a power *stall*, not an overload:

```
r(rpm, torque)        = −0.8750        (log-log: −0.9434)
power                  mean 6280 W, σ 1067 W, range 1148–10470 W
stall  (<3500 W)  n=31   mean rpm 2638   mean torque 10.6
overload (>9000 W) n=64  mean rpm 1341   mean torque 66.9
```

rpm and torque are inversely coupled around the 2860 W design point. High speed
therefore implies *low* torque, and power drops beneath the 3500 W floor.

This is the flagship eval case. A correlation-based or RAG system confabulates a
story supporting the premise. The copilot must refute it with computed evidence
and then explain the true mechanism.

**It is also a confounding trap.** At r = −0.875, *any* analysis of "failures vs
rpm" is confounded by torque. The `compare` and `drivers` ops therefore detect
and report collinearity automatically - see [04-ANALYSIS-IR.md](04-ANALYSIS-IR.md).

## 7. The near-miss surface

Distance is computed **per rule**, not per condition - this distinction is a real
modelling trap. HDF is conjunctive (both conditions required), so its binding
constraint is the *larger* normalised margin; PWF fires on either side, so its
binding constraint is the *smaller*. Treating all five conditions as independent
makes healthy rows appear to have violated a boundary.

The corrected definition is **self-validating**:

```
healthy rows with a negative rule-level margin:  0      (exactly as it must be)
failed  rows with a negative rule-level margin:  287    (= the deterministic count)
```

| Healthy rows within… | count | vs 339 total failures |
|---|---:|---:|
| 2 % of a boundary | 164 | 0.5× |
| 5 % of a boundary | 579 | 1.7× |
| 10 % of a boundary | 2,019 | 6.0× |

Nearly **six hundred healthy cycles came within 5 % of failing**. The binary label
sees none of them; the margin sees every one. That is the quantified justification
for the entire design - and unlike the naive per-condition version, this count
contains no false members.

## 8. Tool wear is a real degradation trajectory

Initially assumed rows were independent snapshots. **That is wrong**, and the
correction materially strengthens the forecasting work.

```
consecutive wear deltas:  2.0 min ×5927 | 3.0 min ×2963 | 5.0 min ×990
tool resets (delta < 0):  119
```

The deltas are exactly the documented L/M/H accrual rates (2/3/5 min) in the
documented proportions. 119 resets against the documentation's *"tool is replaced
120 times."*

**Interpretation:** a single tool wearing across grade-mixed production, with
real replacements. There is a genuine degradation path, so time-to-crossing
forecasts can be validated against observed events rather than only on a
synthetic stream. See [09-STREAMING.md](09-STREAMING.md).

## 9. Physics invariants - quantities that must hold

Used by Gate 2 to distinguish a broken instrument from a changing process
([06-RELIABILITY.md](06-RELIABILITY.md)).

| # | Invariant | Measured | Status |
|---|---|---|---|
| I1 | `process_temp > air_temp` always | 0 violations / 10,000 | ✅ |
| I2 | ΔT ~ N(10, 1) | μ = 10.001, σ = 1.001 | ✅ |
| I3 | rpm ↔ torque inversely coupled | r = −0.8750 | ✅ |
| I4 | mean power stable | 6,280 W | ✅ |

I2 matches the documented generative process to three decimal places.

## 10. What the dataset does *not* contain

| Missing | Consequence | Our handling |
|---|---|---|
| Timestamps | No real temporal questions | `udi` → 2-min takt from a fixed epoch. Every time-based answer is labelled `SYNTHETIC`. |
| Machine identity | Per-product log, not per-machine | Virtual fleet assigned round-robin within variant. Flagged in schema and in answers. |
| Vibration / spectra / thermography | The real PdM modalities are absent | Stated as a limitation; not simulated. |
| Maintenance actions | No intervention outcomes | Alert self-audit designed but not demonstrable here. |

Full detail: **[12-ASSUMPTIONS.md](12-ASSUMPTIONS.md)**.

---

**Next:** [02-TIME-SERIES.md](02-TIME-SERIES.md) - why language models fail on this
data, and the structural answer.
