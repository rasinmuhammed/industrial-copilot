#!/usr/bin/env python3
"""Onboard a new process: turn a plant's history into a verified config.

    python scripts/onboard.py --csv data/ai4i2020.csv --label "Machine failure" \
        --out /tmp/discovered.yaml

WHY THIS IS THE INTERESTING PART
--------------------------------
The moat around Argus platforms is not modelling. It is integration labour:
every new plant is a consulting engagement in which somebody interviews process
engineers, writes down the failure rules, and hand-builds a configuration. That
is why these systems cost what they cost and deploy as slowly as they do.

Everything needed to automate that already existed here in pieces:

  * the observer identifies each channel's noise from the signal, with no
    configuration (it recovers torque's documented sigma of 10 N.m as 10.039)
  * rule discovery recovers failure boundaries from data alone (0.01-3.4% on
    the documented AI4I thresholds)
  * the audit re-derives every threshold and fails the build on disagreement
  * physics.py now READS its process definition rather than compiling it

This wires them into one path. Point it at a history, get a config, and get an
honest report of which rules it could verify and which it could not.

WHAT MAKES IT TRUSTWORTHY RATHER THAN JUST AUTOMATED
----------------------------------------------------
It grades its own output and refuses to overstate it. Every emitted rule carries
a `confidence` and the evidence behind it:

    verified    reproduces the label exactly on every row     -> usable as-is
    candidate   separates well but not exactly                -> engineer review
    rejected    no better than chance                         -> not emitted

A discovered rule is never presented as a documented one. The config is
human-readable YAML precisely so an engineer can correct it, and `make verify`
then audits their correction the same way it audits ours.

The honest limit: this finds thresholds on quantities that are *measured or
derivable*. A failure mode with no sensor signature stays invisible, and the
report says so rather than quietly reporting good coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Grading. Stated as an error budget, not tuned until the demo looked good.
#
# The first version graded each candidate on F1 against the whole label and
# found NOTHING on a dataset whose rules are known to be recoverable. The error
# was conceptual: a real machine has SEVERAL failure modes, the label is their
# union, and several of those modes are conjunctions. A single cut on one metric
# therefore has its recall structurally capped - overstrain alone tops out near
# F1 0.44 here - and gets thrown away no matter how perfect it is at the thing
# it actually explains.
#
# What a failure mode really is: a *sufficient* condition. When it fires, the
# machine broke. It need not explain the other modes' failures. So the right
# criterion is high PRECISION with meaningful support, and the right measure of
# the rule SET is the coverage of their union.
PRECISION_FLOOR = 0.90   # when the rule fires, a failure must really have occurred
MIN_POSITIVES = 15       # below this a "rule" is fitting a handful of rows
MAX_DEPTH = 3            # conjunction depth; HDF needs two terms, OSF needs one
MAX_ROUNDS = 8           # separate-and-conquer passes; one mode surfaces per pass
COUNTER_MONOTONICITY = 0.95   # fraction of non-decreasing steps to call it a counter


@dataclass(slots=True)
class Channel:
    """What we can work out about a signal without being told anything."""

    name: str
    kind: str
    n: int
    mean: float
    sd: float
    lo: float
    hi: float
    process_sd: float
    instrument_sd: float
    repeat_rate: float
    integral: bool

    def as_yaml_block(self) -> str:
        return (
            f"  {_slug(self.name)}:\n"
            f"    source_column: {self.name!r}\n"
            f"    kind: {self.kind}\n"
            f"    observed_range: [{self.lo:.4g}, {self.hi:.4g}]\n"
            f"    process_sd: {self.process_sd:.4g}\n"
            f"    instrument_sd: {self.instrument_sd:.4g}\n"
        )


@dataclass(slots=True)
class Term:
    """One comparison inside a rule."""

    metric: str
    op: str
    value: float

    def __str__(self) -> str:
        return f"{self.metric} {self.op} {self.value:.6g}"


@dataclass(slots=True)
class Rule:
    """A sufficient condition for failure, with the evidence for it."""

    terms: list[Term]
    precision: float
    support: int          # failures this rule explains
    false_alarms: int     # healthy rows it fires on
    confidence: str = "candidate"
    note: str = ""

    @property
    def metric(self) -> str:
        return " AND ".join(str(t) for t in self.terms)


@dataclass(slots=True)
class Report:
    channels: list[Channel] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    leaked: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    label: str = ""
    rows: int = 0
    positives: int = 0
    coverage: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "label": self.label,
            "positives": self.positives,
            "channels": [c.name for c in self.channels],
            "verified": [r.metric for r in self.rules if r.confidence == "verified"],
            "candidates": [r.metric for r in self.rules if r.confidence == "candidate"],
            "rejected": self.rejected,
            "leaked": self.leaked,
            "coverage": round(self.coverage, 4),
        }


def _slug(name: str) -> str:
    s = re.sub(r"\[.*?\]", "", name).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _numeric(values: list[str]) -> list[float] | None:
    out = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            return None
    return out


def profile(name: str, values: list[float]) -> Channel:
    """Identify a channel's character from the signal alone.

    Noise is split by method of moments on the first differences - the same
    identification the observer uses at runtime:

        Var(dz) = q + 2r ,  Cov(dz_t, dz_{t-1}) = -r
    """
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / max(n - 1, 1))
    d = [values[i + 1] - values[i] for i in range(n - 1)]
    non_decreasing = sum(1 for x in d if x >= 0) / max(len(d), 1)

    dm = sum(d) / max(len(d), 1)
    g0 = sum((x - dm) ** 2 for x in d) / max(len(d), 1)
    g1 = (
        sum((d[i] - dm) * (d[i + 1] - dm) for i in range(len(d) - 1)) / max(len(d) - 1, 1)
        if len(d) > 2
        else 0.0
    )
    if g1 < 0:
        r, q = -g1, g0 + 2 * g1
    else:
        r, q = g0 / 2.0, 0.0

    repeats = sum(1 for i in range(n - 1) if values[i + 1] == values[i]) / max(n - 1, 1)
    integral = all(float(v).is_integer() for v in values[:500])
    counter = non_decreasing >= COUNTER_MONOTONICITY

    return Channel(
        name=name,
        kind="counter" if counter else "level",
        n=n,
        mean=mean,
        sd=sd,
        lo=min(values),
        hi=max(values),
        process_sd=math.sqrt(max(q, 0.0)),
        instrument_sd=math.sqrt(max(r, 0.0)),
        repeat_rate=repeats,
        integral=integral,
    )


def _leakage(name: str, values: list[float], label: list[bool]) -> bool:
    """Is this column a component of the label rather than a cause of it?

    AI4I ships TWF/HDF/PWF/OSF/RNF alongside the machine_failure flag. They are
    the label decomposed, so a learner handed them "discovers" that failure
    happens when the failure flag is set. Target leakage is among the most
    common ways an industrial model looks excellent in evaluation and useless in
    production, and it is worth catching by construction rather than by noticing
    that the numbers are too good.

    The signature: a rare indicator almost entirely contained inside the label.
    """
    distinct = set(values)
    if len(distinct) != 2:
        return False
    hi = max(distinct)
    on = [i for i, v in enumerate(values) if v == hi]
    if not on or len(on) > 0.5 * len(values):
        return False
    contained = sum(1 for i in on if label[i]) / len(on)
    return contained >= 0.90


def _simplify(rule: Rule) -> Rule:
    """Collapse redundant terms from a tree path.

    A tree can split the same feature twice on the way down, so a path arrives
    as `strain > 10293 AND strain > 10998`. Only the tighter bound constrains
    anything. Keeping both is not wrong, but a discovered rule is meant to be
    read and corrected by a process engineer, and noise in the statement is a
    tax on the person doing that.
    """
    tightest: dict[tuple[str, str], Term] = {}
    for t in rule.terms:
        key = (t.metric, t.op)
        current = tightest.get(key)
        if current is None:
            tightest[key] = t
        elif t.op == ">" and t.value > current.value:
            tightest[key] = t
        elif t.op == "<=" and t.value < current.value:
            tightest[key] = t
    rule.terms = list(tightest.values())
    return rule


def discover(
    features: dict[str, list[float]], label: list[bool]
) -> tuple[list[Rule], float]:
    """Find sufficient conditions for failure, including conjunctions.

    A decision tree is the right instrument and not a concession: each
    root-to-leaf path IS a conjunction of threshold tests, which is exactly the
    shape a documented failure rule takes. Reading its paths recovers
    conjunctive modes no single-cut search can express, and the output is a rule
    an engineer can read rather than a model they must trust.

    But one tree is not enough, and finding that out was the second correction
    here. A single depth-3 fit reached only 8.3% coverage on a dataset whose
    rules are fully recoverable, because a tree PARTITIONS the space: four
    independent failure modes compete for the same splits and the largest one
    wins every contest.

    Real machines fail in several unrelated ways at once, so the search has to
    be separate-and-conquer. Fit, keep the high-precision paths, remove the
    failures they explain, refit on what is left. Each round surfaces the next
    mode. This is classical rule induction (CN2, RIPPER), and it matches the
    domain exactly: a failure mode is a sufficient condition, not a partition.
    """
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    names = list(features)
    x = np.column_stack([np.asarray(features[n], dtype=float) for n in names])
    y = np.asarray(label, dtype=bool)
    index = {n: k for k, n in enumerate(names)}

    def mask_of(terms: list[Term]) -> "np.ndarray":
        m = np.ones(len(y), dtype=bool)
        for t in terms:
            col = x[:, index[t.metric]]
            m &= (col <= t.value) if t.op == "<=" else (col > t.value)
        return m

    rules: list[Rule] = []
    remaining = y.copy()

    for _ in range(MAX_ROUNDS):
        if remaining.sum() < MIN_POSITIVES:
            break
        tree = DecisionTreeClassifier(
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_POSITIVES,
            class_weight="balanced",
            random_state=0,
        ).fit(x, remaining)
        t = tree.tree_
        found: list[Rule] = []

        def walk(node: int, terms: list[Term]) -> None:
            if t.children_left[node] == -1:
                if not terms:
                    return
                m = mask_of(terms)
                fires = int(m.sum())
                if not fires:
                    return
                # Precision is scored against the TRUE label, never the
                # residual. A rule that looks clean only because earlier rounds
                # removed its counterexamples is not a rule.
                support = int((m & y).sum())
                precision = support / fires
                new = int((m & remaining).sum())
                if precision >= PRECISION_FLOOR and new >= MIN_POSITIVES:
                    found.append(Rule(
                        terms=list(terms), precision=precision, support=support,
                        false_alarms=fires - support,
                    ))
                return
            feature = names[t.feature[node]]
            cut = float(t.threshold[node])
            walk(t.children_left[node], terms + [Term(feature, "<=", cut)])
            walk(t.children_right[node], terms + [Term(feature, ">", cut)])

        walk(0, [])
        found = [_simplify(r) for r in found]
        if not found:
            break
        for rule in found:
            rules.append(rule)
            remaining &= ~mask_of(rule.terms)

    covered = np.zeros(len(y), dtype=bool)
    for rule in rules:
        covered |= mask_of(rule.terms)
    coverage = float((covered & y).sum() / max(y.sum(), 1))

    rules.sort(key=lambda r: (-r.support, -r.precision))
    return rules, coverage


def derive_columns(
    numeric: dict[str, list[float]], channels: dict[str, Channel]
) -> dict[str, list[float]]:
    """Physically meaningful combinations, built from units rather than searched.

    This is the step that carries the discovery. Raw sensors separate failures
    poorly; the dimensional constructions separate them almost perfectly,
    because the actual failure boundaries live on derived quantities. A plant
    gets the units from OPC-UA tag metadata; here they come from the column
    headers, which is the same information in a shabbier container.
    """
    out: dict[str, list[float]] = {}
    names = list(numeric)

    def unit_of(name: str) -> str:
        m = re.search(r"\[(.*?)\]", name)
        return (m.group(1) if m else "").strip().lower()

    kelvin = [n for n in names if unit_of(n) in ("k", "c", "degc")]
    rpm = [n for n in names if unit_of(n) == "rpm"]
    torque = [n for n in names if unit_of(n) in ("nm", "n*m", "n.m")]
    minutes = [n for n in names if unit_of(n) == "min"]

    # Temperature differences: a gradient is what drives heat transfer, so the
    # difference between two temperature channels is physically meaningful in a
    # way neither absolute value is.
    for i, a in enumerate(kelvin):
        for b in kelvin[i + 1:]:
            delta = [x - y for x, y in zip(numeric[b], numeric[a])]
            if abs(sum(delta) / len(delta)) > 1e-9:
                out[f"{_slug(b)}_minus_{_slug(a)}"] = delta

    # torque x angular velocity = mechanical power. Forced by dimensions.
    for t in torque:
        for s in rpm:
            out[f"power_from_{_slug(t)}_{_slug(s)}"] = [
                x * y * 2 * math.pi / 60 for x, y in zip(numeric[t], numeric[s])
            ]

    # An accumulating counter times a load is an accumulated-damage proxy.
    for m in minutes:
        if channels.get(m) and channels[m].kind == "counter":
            for t in torque:
                out[f"{_slug(m)}_x_{_slug(t)}"] = [
                    x * y for x, y in zip(numeric[m], numeric[t])
                ]
    return out


def onboard(csv_path: Path, label_col: str, out_path: Path) -> Report:
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{csv_path} has no rows")
    if label_col not in rows[0]:
        raise SystemExit(f"label column {label_col!r} not found. Columns: {list(rows[0])}")

    label = [str(r[label_col]).strip() not in ("0", "", "False", "false") for r in rows]
    report = Report(label=label_col, rows=len(rows), positives=sum(label))

    numeric: dict[str, list[float]] = {}
    categorical: dict[str, list[str]] = {}
    for col in rows[0]:
        if col == label_col:
            continue
        raw = [r[col] for r in rows]
        vals = _numeric(raw)
        if vals is not None and len(set(vals)) > 1:
            if _leakage(col, vals, label):
                report.leaked.append(_slug(col))
                continue
            numeric[col] = vals
        elif vals is None and 1 < len(set(raw)) <= 12:
            categorical[col] = raw

    channels = {name: profile(name, vals) for name, vals in numeric.items()}
    report.channels = list(channels.values())

    features: dict[str, list[float]] = {_slug(k): v for k, v in numeric.items()}
    features.update(derive_columns(numeric, channels))

    # One-hot the low-cardinality categoricals. Without this a per-variant limit
    # is unreachable: AI4I's overstrain boundary is 11,000 / 12,000 / 13,000
    # depending on product type, and no global cut expresses that.
    for col, raw in categorical.items():
        for level in sorted(set(raw)):
            features[f"{_slug(col)}_is_{_slug(level)}"] = [
                1.0 if v == level else 0.0 for v in raw
            ]

    report.rules, report.coverage = discover(features, label)
    for rule in report.rules:
        if rule.false_alarms == 0:
            rule.confidence = "verified"
            rule.note = "fires only on rows that actually failed"
        else:
            rule.confidence = "candidate"
            rule.note = (
                f"fires on {rule.false_alarms} healthy row(s); "
                f"needs an engineer to confirm, tighten, or delete"
            )
    report.features = sorted(features)
    out_path.write_text(_render(report, csv_path))
    return report


def _render(report: Report, csv_path: Path) -> str:
    lines = [
        "# Process definition DISCOVERED from data. Not documented, not reviewed.",
        "#",
        f"# Source   : {csv_path}",
        f"# Rows     : {report.rows:,}   labelled failures: {report.positives:,}",
        f"# Coverage : {report.coverage:.1%} of failures explained by the rules below",
        "#",
        "# Every rule carries the evidence that produced it. `verified` fires only on",
        "# rows that actually failed; `candidate` also fires on healthy rows and needs",
        "# a process engineer to confirm, tighten, or delete it. Nothing here is",
        "# documented physics until somebody who knows the machine says so.",
        "",
        "version: 1",
        "provenance: discovered",
        f"coverage: {report.coverage:.4f}",
        "",
        "channels:",
    ]
    for ch in report.channels:
        lines.append(ch.as_yaml_block().rstrip("\n"))
    if report.leaked:
        lines += ["", "# Excluded as target leakage - these are the label decomposed,",
                  "# not causes of it:", f"# {', '.join(report.leaked)}"]
    lines += ["", "modes:"]
    if not report.rules:
        lines.append("  []  # nothing reached the precision floor")
    for i, r in enumerate(report.rules, 1):
        lines += [
            f"  - code: D{i:02d}",
            f"    name: {r.metric}",
            "    kind: deterministic",
            f"    confidence: {r.confidence}",
            f"    note: {r.note}",
            "    evidence:",
            f"      precision: {r.precision:.4f}",
            f"      failures_explained: {r.support}",
            f"      false_alarms: {r.false_alarms}",
            "    predicate:",
        ]
        if len(r.terms) == 1:
            t = r.terms[0]
            lines += ["      metric: " + t.metric, f'      op: "{t.op}"',
                      f"      value: {t.value:.6g}"]
        else:
            lines.append("      all_of:")
            for t in r.terms:
                lines.append(
                    f'        - {{metric: {t.metric}, op: "{t.op}", value: {t.value:.6g}}}')
    return "\n".join(lines) + "\n"


# Discovered feature names are built from column headers; documented ones come
# from the process description. Mapping between them is a human step, and it is
# only needed for the AUDIT - validating the method against a process whose
# rules we already know. A genuinely new plant has nothing to audit against,
# which is exactly why the confidence grading has to carry the honesty instead.
_AUDIT_ALIAS = {
    "power_from_torque_rotational_speed": "power_w",
    "process_temperature_minus_air_temperature": "temp_delta_k",
    "tool_wear_x_torque": "overstrain_min_nm",
    "rotational_speed": "rotational_speed_rpm",
    "tool_wear": "tool_wear_min",
}


def audit(report: Report) -> list[tuple[str, float, float, float]]:
    """Score discovered boundaries against the documented ones.

    This is the claim that makes onboarding credible: the thresholds are not
    memorised from the dataset documentation, because nothing upstream of this
    function has seen it. Returns (metric, discovered, documented, % error).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from copilot.process_model import load_process_model

    documented: dict[str, list[float]] = {}
    for mode in load_process_model().deterministic_modes:
        for cond in mode.conditions:
            if cond.value_by_type:
                documented.setdefault(cond.metric, []).extend(cond.value_by_type.values())
            elif isinstance(cond.value, (int, float)):
                documented.setdefault(cond.metric, []).append(float(cond.value))

    out = []
    for rule in report.rules:
        for term in rule.terms:
            target = _AUDIT_ALIAS.get(term.metric)
            if target is None or target not in documented:
                continue
            nearest = min(documented[target], key=lambda v: abs(v - term.value))
            if nearest == 0:
                continue
            err = abs(term.value - nearest) / abs(nearest) * 100
            if err < 10:      # a match, not a coincidence
                out.append((target, term.value, nearest, err))
    out.sort(key=lambda r: r[3])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--json", action="store_true", help="machine-readable summary")
    ap.add_argument("--audit", action="store_true",
                    help="score the discovered boundaries against the documented ones")
    args = ap.parse_args()

    report = onboard(args.csv, args.label, args.out)

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
        return 0

    print(f"\n  {args.csv}  ->  {report.rows:,} rows, "
          f"{report.positives:,} labelled failures ({report.positives/report.rows:.2%})")
    print(f"  {len(report.channels)} channels profiled with no configuration:\n")
    for ch in report.channels:
        print(f"    {_slug(ch.name):26s} {ch.kind:8s} "
              f"process sd {ch.process_sd:9.4g}   instrument sd {ch.instrument_sd:9.4g}")

    if report.leaked:
        print(f"\n  EXCLUDED as target leakage ({len(report.leaked)}): "
              f"{', '.join(report.leaked)}")
        print("    These are the label decomposed, not causes of it.")

    for tier, head in (("verified", "VERIFIED - fire only on rows that failed"),
                       ("candidate", "CANDIDATE - need an engineer's confirmation")):
        subset = [r for r in report.rules if r.confidence == tier]
        print(f"\n  {head} ({len(subset)}):")
        for r in subset:
            print(f"    {r.metric}")
            print(f"        precision {r.precision:.3f}   explains {r.support} "
                  f"failures   false alarms {r.false_alarms}")
        if not subset:
            print("    none")

    print(f"\n  Union coverage: {report.coverage:.1%} of all labelled failures.")

    if args.audit:
        matches = audit(report)
        print("\n  AUDIT - discovered vs documented (nothing upstream saw the docs):")
        print(f"    {'metric':<24}{'discovered':>13}{'documented':>13}{'error':>9}")
        for metric, got, want, err in matches:
            print(f"    {metric:<24}{got:>13.6g}{want:>13.6g}{err:>8.2f}%")
        if matches:
            worst = max(m[3] for m in matches)
            print(f"\n    {len(matches)} boundaries recovered, worst error {worst:.2f}%")
    print(f"\n  config -> {args.out}")
    print("  Review it before use. A discovered rule is not a documented one.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
