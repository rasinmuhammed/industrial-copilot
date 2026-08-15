# 12 — Assumptions and Limitations

> The brief says: *"Assume whatever is necessary, but state your assumptions."*
>
> Synthetic overlays are the most common quiet failure in submissions like this:
> a candidate invents timestamps, then answers *"what happened Tuesday?"* as
> though Tuesday were real. Every overlay below is marked `SYNTHETIC` in the
> semantic layer, and **any answer depending on one says so in the response
> itself.**

---

## 1. Data assumptions

### A1 — Timestamps do not exist *(SYNTHETIC)*

The dataset has only `UDI`, a row index.

**Assumption:** `UDI` is a production cycle index at a fixed **2-minute takt**,
anchored at `2024-01-01T00:00:00Z`, yielding ≈ 13.9 days of continuous production.

**Why:** it makes time-grain questions demonstrable.
**Risk:** none if labelled; misleading if not. Every time-based answer carries the caveat.
**Config:** `takt_seconds`, `epoch`.

### A2 — Machine identity does not exist *(SYNTHETIC)*

AI4I is a per-*product* log, not a per-*machine* one.

**Assumption:** a virtual fleet of 5 machines per quality variant (15 total),
assigned round-robin by `UDI` within variant.

**Why:** *"what has been happening with this machine?"* is acceptance criterion 1
and is otherwise unanswerable.
**Risk:** per-machine differences are artifacts of the assignment, not physics.
The copilot must never present a machine-to-machine difference as meaningful.
**Config:** `virtual_machines_per_type`.

### A3 — Shifts are derived *(SYNTHETIC)*

8-hour shifts A/B/C from the synthetic timestamp. Inherits A1's caveat entirely.

### A4 — Product quality variant is the real asset class

`Type` (L/M/H) determines the OSF threshold, so it is the **one dimension with
genuine engineering meaning** in the published data. Treated as asset class for
KB scoping.

### A5 — Tool wear is a single-tool trajectory

Verified: deltas are exactly 2/3/5 min matching documented L/M/H accrual, with
119 resets against the documented "tool replaced 120 times."

**Assumption:** one tool wearing across grade-mixed production, replaced 119–120
times. Justifies treating wear as a degradation path for forecasting.
**Risk:** low — this is data-supported, not invented.

---

## 2. Knowledge assumptions

### A6 — The knowledge base is authoritative over the prose

Where the dataset page and the published columns disagree (TWF: 120 events in
prose, 46 in the column), computation follows the **column**, and the discrepancy
is surfaced as a `data_quality` finding rather than reconciled silently.

### A7 — Documented thresholds are treated as given *for the prototype*

We use 8.6 K, 1380 rpm, 3500/9000 W, 11k/12k/13k as authoritative.

**This is the load-bearing assumption of the whole project**, and the one a
reviewer should press on. Mitigation: `scripts/discover_rules.py` re-derives them
from data alone to within 0.01–3.4 % ([08-DISCOVERY.md](08-DISCOVERY.md)),
demonstrating the approach does not *depend* on the documentation. But in a real
plant, threshold acquisition becomes the primary engineering work.

### A8 — RNF is unpredictable by construction

0.1 % background, parameter-independent. **The system refuses RNF root-cause
questions.** This is correct behaviour, and it is tested.

### A9 — Orphan failures are genuinely unexplained

Nine rows, tested for structure and found to have none (worst-margin 0.057 vs
0.068 healthy; none carries RNF). Reported as `cause_undetermined` with evidence.
**A verified correct answer, not a fallback.**

---

## 3. Deployment assumptions

### A10 — Units are available in a real deployment

Discovery depends on knowing sensor units. Assumed available from OPC-UA tag
metadata, instrument ranges, or P&ID line lists. Reasonable — these exist in
every instrumented plant — but it is a dependency.

### A11 — Failure labels eventually arrive

The KB calibration monitor needs failures. Assumed to arrive from a CMMS with a
lag of days. The monitor is designed for delayed labels; it degrades to slower
detection, not to incorrect detection.

### A12 — Sensor uncertainty is knowable

Interval margins need interval widths. Assumed from declared instrument accuracy,
falling back to observed dispersion, falling back to a wide default that
correctly yields ABSTAIN.

---

## 4. Limitations we are not solving

| # | Limitation | Consequence | Path |
|---|---|---|---|
| L1 | No vibration, spectra, thermography, oil analysis | The real PdM modalities are absent from AI4I | Learned health indicators feeding the same margin abstraction |
| L2 | Per-sample predicates only | Cannot express "RMS rising 3σ over 6 h" | Temporal operators in the rule language |
| L3 | Drift defenses tested on **injected** drift | Not field-validated | Pilot deployment |
| L4 | Forecast assumes stationary drift | Step change in duty cycle invalidates it | Gate 3 catches it after the fact |
| L5 | Discovery F1 figures are in-sample | Bracketing is valid; generalisation is unproven | Cross-validated brackets + CIs |
| L6 | Cold start | New asset class with no failures yields nothing | Hierarchical priors once a sibling exists |
| L7 | No closed-loop actuation | Advisory only | Needs PHA and interlock design |
| L8 | Single-node prototype | Scale is argued, not measured | [11-SCALE.md](11-SCALE.md) |
| L9 | Engineer confirmation is a bottleneck | Discovery needs scarce expert time | Active learning by information gain |
| L10 | No RBAC, tenancy, or audit UI | Not production-deployable | Standard work, out of scope |

---

## 5. What we explicitly do not claim

- **Not better than Siemens Senseye, Cognite, or Augury as a product.** They have
  the OT connector moat, domain breadth from real machine-hours, the sensor
  modalities that matter, and deployment learning. Not close.
- **Not novel in its components.** Model-based diagnosis, PHM health indicators,
  inverse-Gaussian first passage, interval arithmetic, and Proof-Carrying Numbers
  all have clear prior art. The contribution is the composition, plus three
  pieces for which no prior art was found (KB calibration monitor,
  invariant-based sensor/process discrimination, the four-gate perimeter).
- **Not proven in production.** Every empirical figure is from this dataset.

**What we do claim:** a better *answer engine* on this dataset, on the axes the
brief names — latency, context engineering, hallucination reduction — with an
architecture aimed at the documented 80 % failure mode rather than the 20 %
everyone else optimises.

---

**Next:** [13-BUILD-PLAN.md](13-BUILD-PLAN.md)
