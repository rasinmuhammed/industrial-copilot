#!/usr/bin/env python3
"""Reproduce every figure in docs/01-DATASET.md directly from the CSV.

Deliberately dependency-light and independent of copilot/: if this and the
engine ever disagree, that is a finding, not a nuisance. Exits non-zero if any
documented claim fails to reproduce.

    python scripts/verify_dataset.py
"""

from __future__ import annotations

import csv
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "data" / "ai4i2020.csv"
OSF_THRESHOLD = {"L": 11000.0, "M": 12000.0, "H": 13000.0}
RAD_PER_RPM = 2 * math.pi / 60

failures: list[str] = []


def check(label: str, actual, expected, tol: float = 0.0) -> None:
    if isinstance(expected, float) or isinstance(actual, float):
        ok = abs(float(actual) - float(expected)) <= tol
    else:
        ok = actual == expected
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {label:<52} {actual!s:>12}  (expect {expected})")
    if not ok:
        failures.append(label)


def load() -> list[dict]:
    if not CSV.exists():
        sys.exit(f"Dataset not found at {CSV}. See README > Setup.")
    rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
    for r in rows:
        r["air"] = float(r["Air temperature [K]"])
        r["proc"] = float(r["Process temperature [K]"])
        r["rpm"] = float(r["Rotational speed [rpm]"])
        r["tq"] = float(r["Torque [Nm]"])
        r["wear"] = float(r["Tool wear [min]"])
        r["dT"] = r["proc"] - r["air"]
        r["power"] = r["tq"] * r["rpm"] * RAD_PER_RPM
        r["strain"] = r["wear"] * r["tq"]
        r["osf_th"] = OSF_THRESHOLD[r["Type"]]
        r["mf"] = int(r["Machine failure"])
        for m in ("TWF", "HDF", "PWF", "OSF", "RNF"):
            r[m] = int(r[m])
        # Rule-level distance: HDF is conjunctive (binding = larger margin),
        # PWF is disjunctive (binding = smaller), OSF is a single condition.
        r["hdf_d"] = max((r["dT"] - 8.6) / 8.6, (r["rpm"] - 1380) / 1380)
        r["pwf_d"] = min((r["power"] - 3500) / 3500, (9000 - r["power"]) / 9000)
        r["osf_d"] = (r["osf_th"] - r["strain"]) / r["osf_th"]
        r["worst"] = min(r["hdf_d"], r["pwf_d"], r["osf_d"])
    return rows


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> int:
    rows = load()
    print(f"Verifying docs/01-DATASET.md against {CSV.name}\n{'=' * 74}")

    # -- §2 class balance -------------------------------------------------
    section("§2  Class balance")
    check("rows", len(rows), 10000)
    types = Counter(r["Type"] for r in rows)
    check("product_type L", types["L"], 6000)
    check("product_type M", types["M"], 2997)
    check("product_type H", types["H"], 1003)
    check("machine_failure", sum(r["mf"] for r in rows), 339)
    for mode, expected in (("TWF", 46), ("HDF", 115), ("PWF", 95), ("OSF", 98), ("RNF", 19)):
        check(mode, sum(r[mode] for r in rows), expected)

    # -- §3 rule verification --------------------------------------------
    section("§3  Rule verification  (the central result)")
    rules = {
        "HDF": lambda r: r["dT"] < 8.6 and r["rpm"] < 1380,
        "PWF": lambda r: r["power"] < 3500 or r["power"] > 9000,
        "OSF": lambda r: r["strain"] > r["osf_th"],
    }
    for mode, pred in rules.items():
        fp = sum(1 for r in rows if pred(r) and not r[mode])
        fn = sum(1 for r in rows if not pred(r) and r[mode])
        check(f"{mode} false positives", fp, 0)
        check(f"{mode} false negatives", fn, 0)

    window = [r for r in rows if 200 <= r["wear"] <= 240]
    check("TWF window size", len(window), 790)
    check("TWF labelled in window", sum(r["TWF"] for r in window), 43)
    check("TWF outside window", sum(r["TWF"] for r in rows) - 43, 3)

    def any_det(r):
        return rules["HDF"](r) or rules["PWF"](r) or rules["OSF"](r)

    check("HDF|PWF|OSF vs failure: TP", sum(1 for r in rows if any_det(r) and r["mf"]), 287)
    check("HDF|PWF|OSF vs failure: FP", sum(1 for r in rows if any_det(r) and not r["mf"]), 0)

    # -- §4 label integrity ------------------------------------------------
    section("§4  Label-integrity findings")
    orphans = [r for r in rows if r["mf"] and not (r["TWF"] or r["HDF"] or r["PWF"] or r["OSF"])]
    check("orphan failures", len(orphans), 9)
    check("orphans carrying RNF", sum(r["RNF"] for r in orphans), 0)
    check("orphan worst-margin mean", round(st.mean(r["worst"] for r in orphans), 3), 0.153, 0.001)
    healthy = [r["worst"] for r in rows if not r["mf"]]
    check("healthy worst-margin mean", round(st.mean(healthy), 3), 0.180, 0.001)
    check("RNF=1 also machine_failure", sum(1 for r in rows if r["RNF"] and r["mf"]), 1)

    # -- §5 multi-mode -----------------------------------------------------
    section("§5  Multi-mode failures")
    combos = Counter(
        "+".join(m for m in ("TWF", "HDF", "PWF", "OSF") if r[m]) or "NONE"
        for r in rows
        if r["mf"]
    )
    check("PWF+OSF", combos["PWF+OSF"], 11)
    check("HDF+OSF", combos["HDF+OSF"], 6)
    check("HDF+PWF", combos["HDF+PWF"], 3)
    check("multi-mode total", sum(v for k, v in combos.items() if "+" in k), 23)

    # -- §6 the false premise ----------------------------------------------
    section("§6  The brief's example question is a false premise")
    speeds = sorted(r["rpm"] for r in rows)
    edges = [speeds[int(len(speeds) * p)] for p in (0, 0.2, 0.4, 0.6, 0.8)] + [max(speeds) + 1]
    rates = []
    for i in range(5):
        band = [r for r in rows if edges[i] <= r["rpm"] < edges[i + 1]]
        rates.append(100 * sum(r["mf"] for r in band) / len(band))
    check("failure rate, lowest rpm quintile %", round(rates[0], 2), 12.17, 0.01)
    check("failure rate, highest rpm quintile %", round(rates[4], 2), 2.24, 0.01)
    check("low/high ratio", round(rates[0] / rates[4], 1), 5.4, 0.1)

    def pearson(a, b):
        ma, mb = st.mean(a), st.mean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
        return num / den

    r_st = pearson([r["rpm"] for r in rows], [r["tq"] for r in rows])
    check("r(rpm, torque)", round(r_st, 4), -0.8750, 0.0001)
    stall = [r for r in rows if r["power"] < 3500]
    check("stall count", len(stall), 31)
    check("stall mean rpm", round(st.mean(r["rpm"] for r in stall)), 2638, 1)
    check("stall mean torque", round(st.mean(r["tq"] for r in stall), 1), 10.6, 0.1)

    # -- §7 near-miss surface ----------------------------------------------
    section("§7  The near-miss surface")
    for thr, expected in ((0.02, 164), (0.05, 579), (0.10, 2019)):
        n = sum(1 for r in rows if not r["mf"] and 0 < r["worst"] < thr)
        check(f"healthy rows within {int(thr * 100)}% of a boundary", n, expected)
    # Self-validation: the rule-level margin must be negative on exactly the
    # deterministically-explained failures, and on nothing else.
    check("healthy rows with negative rule margin",
          sum(1 for r in rows if not r["mf"] and r["worst"] < 0), 0)
    check("failed rows with negative rule margin",
          sum(1 for r in rows if r["mf"] and r["worst"] < 0), 287)

    # -- §8 degradation trajectory -----------------------------------------
    section("§8  Tool wear is a real degradation trajectory")
    deltas = [rows[i + 1]["wear"] - rows[i]["wear"] for i in range(len(rows) - 1)]
    hist = Counter(d for d in deltas if d >= 0)
    check("wear delta 2.0 min", hist[2.0], 5927)
    check("wear delta 3.0 min", hist[3.0], 2963)
    check("wear delta 5.0 min", hist[5.0], 990)
    check("tool resets", sum(1 for d in deltas if d < 0), 119)

    # -- §9 physics invariants ---------------------------------------------
    section("§9  Physics invariants  (Gate 2 foundation)")
    check("I1  process > air, violations", sum(1 for r in rows if r["proc"] <= r["air"]), 0)
    check("I2  mean dT", round(st.mean(r["dT"] for r in rows), 3), 10.001, 0.001)
    check("I2  sd dT", round(st.stdev(r["dT"] for r in rows), 3), 1.001, 0.001)
    check("I3  r(rpm, torque)", round(r_st, 4), -0.8750, 0.0001)
    check("I4  mean power W", round(st.mean(r["power"] for r in rows)), 6280, 1)

    # -- summary ------------------------------------------------------------
    print("\n" + "=" * 74)
    if failures:
        print(f"FAILED - {len(failures)} claim(s) did not reproduce:")
        for f in failures:
            print(f"  · {f}")
        return 1
    print("PASSED - every documented figure reproduces exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
