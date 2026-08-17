"""Independent reference implementation.

Deliberately imports NOTHING from `copilot`. It reads the CSV with the standard
library and numpy and recomputes expected answers from scratch.

That independence is the whole point. If the eval harness asserted against the
engine's own output, or against stored snapshots the engine once produced, a bug
present in both would pass silently. Two implementations written separately from
the same specification will not usually be wrong in the same way.

Where this file and the engine disagree, that is a finding to investigate, not a
tolerance to widen.
"""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "ai4i2020.csv"

OSF_THRESHOLD = {"L": 11000.0, "M": 12000.0, "H": 13000.0}
WEAR_RATE = {"L": 2.0, "M": 3.0, "H": 5.0}
RAD_PER_RPM = 2 * math.pi / 60
HDF_TEMP, HDF_SPEED = 8.6, 1380.0
PWF_LOW, PWF_HIGH = 3500.0, 9000.0
TWF_WINDOW = (200.0, 240.0)


@lru_cache(maxsize=1)
def data() -> dict[str, Any]:
    """Load and derive everything once."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"dataset not found at {CSV_PATH}")
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8-sig")))

    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    flag = lambda k: np.array([int(r[k]) for r in rows])  # noqa: E731

    air = col("Air temperature [K]")
    proc = col("Process temperature [K]")
    rpm = col("Rotational speed [rpm]")
    torque = col("Torque [Nm]")
    wear = col("Tool wear [min]")
    ptype = np.array([r["Type"] for r in rows])

    temp_delta = proc - air
    power = torque * rpm * RAD_PER_RPM
    strain = wear * torque
    threshold = np.array([OSF_THRESHOLD[t] for t in ptype])

    # Rule-level distances. HDF is conjunctive so its binding constraint is the
    # LARGER normalised margin; PWF is disjunctive so it is the smaller.
    hdf_distance = np.maximum((temp_delta - HDF_TEMP) / HDF_TEMP, (rpm - HDF_SPEED) / HDF_SPEED)
    pwf_distance = np.minimum((power - PWF_LOW) / PWF_LOW, (PWF_HIGH - power) / PWF_HIGH)
    osf_distance = (threshold - strain) / threshold

    return {
        "n": len(rows),
        "udi": np.array([int(r["UDI"]) for r in rows]),
        "product_type": ptype,
        "air_temp_k": air,
        "process_temp_k": proc,
        "rotational_speed_rpm": rpm,
        "torque_nm": torque,
        "tool_wear_min": wear,
        "temp_delta_k": temp_delta,
        "power_w": power,
        "overstrain_min_nm": strain,
        "osf_threshold": threshold,
        "failure": flag("Machine failure"),
        "twf": flag("TWF"),
        "hdf": flag("HDF"),
        "pwf": flag("PWF"),
        "osf": flag("OSF"),
        "rnf": flag("RNF"),
        "hdf_rule": (temp_delta < HDF_TEMP) & (rpm < HDF_SPEED),
        "pwf_rule": (power < PWF_LOW) | (power > PWF_HIGH),
        "osf_rule": strain > threshold,
        "twf_window": (wear >= TWF_WINDOW[0]) & (wear <= TWF_WINDOW[1]),
        "worst_margin": np.minimum(np.minimum(hdf_distance, pwf_distance), osf_distance),
    }


def _mask(**filters: Any) -> np.ndarray:
    d = data()
    m = np.ones(d["n"], dtype=bool)
    for key, value in filters.items():
        m &= d[key] == value
    return m


# --------------------------------------------------------------------------
# Expected answers
# --------------------------------------------------------------------------


def overall_failure_rate() -> float:
    d = data()
    return float(d["failure"].sum()) / d["n"] * 100.0


def failure_count() -> int:
    return int(data()["failure"].sum())


def rate_by_product_type() -> dict[str, float]:
    d = data()
    out = {}
    for variant in ("L", "M", "H"):
        m = d["product_type"] == variant
        out[variant] = float(d["failure"][m].sum()) / int(m.sum()) * 100.0
    return out


def count_by_product_type() -> dict[str, int]:
    d = data()
    return {v: int((d["product_type"] == v).sum()) for v in ("L", "M", "H")}


def rate_by_rpm_quintile() -> list[float]:
    """Quintile edges by the same convention the engine uses (DuckDB
    quantile_cont, which is linear interpolation between order statistics)."""
    d = data()
    rpm, failure = d["rotational_speed_rpm"], d["failure"]
    edges = [float(np.quantile(rpm, q, method="linear")) for q in (0, 0.2, 0.4, 0.6, 0.8)]
    edges.append(float(rpm.max()) + 1)
    rates = []
    for i in range(5):
        m = (rpm >= edges[i]) & (rpm < edges[i + 1])
        rates.append(float(failure[m].sum()) / int(m.sum()) * 100.0)
    return rates


def cohort_mean(metric: str, *, failed: bool) -> float:
    d = data()
    m = d["failure"] == (1 if failed else 0)
    return float(d[metric][m].mean())


def cohort_n(*, failed: bool) -> int:
    d = data()
    return int((d["failure"] == (1 if failed else 0)).sum())


def cohens_d(metric: str) -> float:
    d = data()
    a = d[metric][d["failure"] == 1]
    b = d[metric][d["failure"] == 0]
    na, nb = len(a), len(b)
    pooled = ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2)
    return float((a.mean() - b.mean()) / math.sqrt(pooled))


def correlation(x: str, y: str) -> float:
    d = data()
    return float(np.corrcoef(d[x], d[y])[0, 1])


def mode_counts() -> dict[str, int]:
    d = data()
    return {m.upper(): int(d[m].sum()) for m in ("twf", "hdf", "pwf", "osf", "rnf")}


def rule_agreement() -> dict[str, tuple[int, int]]:
    """(false positives, false negatives) per deterministic mode."""
    d = data()
    out = {}
    for mode in ("hdf", "pwf", "osf"):
        rule, label = d[f"{mode}_rule"], d[mode].astype(bool)
        out[mode.upper()] = (int((rule & ~label).sum()), int((~rule & label).sum()))
    return out


def orphan_failures() -> int:
    d = data()
    m = (d["failure"] == 1) & (d["twf"] == 0) & (d["hdf"] == 0) & (d["pwf"] == 0) & (d["osf"] == 0)
    return int(m.sum())


def multi_mode_failures() -> int:
    d = data()
    fired = d["twf"] + d["hdf"] + d["pwf"] + d["osf"]
    return int(((d["failure"] == 1) & (fired > 1)).sum())


def deterministically_explained() -> int:
    d = data()
    any_rule = d["hdf_rule"] | d["pwf_rule"] | d["osf_rule"]
    return int((any_rule & (d["failure"] == 1)).sum())


def rnf_rollup() -> tuple[int, int]:
    d = data()
    return int(d["rnf"].sum()), int(((d["rnf"] == 1) & (d["failure"] == 1)).sum())


def twf_in_window() -> tuple[int, int]:
    """(rows inside the window, of which labelled TWF)."""
    d = data()
    return int(d["twf_window"].sum()), int((d["twf_window"] & (d["twf"] == 1)).sum())


def near_miss_counts() -> dict[float, int]:
    d = data()
    healthy = d["failure"] == 0
    return {
        t: int(((d["worst_margin"] > 0) & (d["worst_margin"] < t) & healthy).sum())
        for t in (0.02, 0.05, 0.10)
    }


def row_by_udi(udi: int) -> dict[str, Any]:
    d = data()
    idx = int(np.where(d["udi"] == udi)[0][0])
    return {
        key: (value[idx] if isinstance(value, np.ndarray) else value)
        for key, value in d.items()
        if isinstance(value, np.ndarray)
    }


def osf_crossing_wear(torque: float, product_type: str = "L") -> float:
    return OSF_THRESHOLD[product_type] / torque


def osf_cycles_to_crossing(wear: float, torque: float, product_type: str = "L") -> float:
    margin = OSF_THRESHOLD[product_type] - wear * torque
    drift = WEAR_RATE[product_type] * torque
    return margin / drift


def counterfactual_mode_counts(torque_delta: float) -> dict[str, int]:
    """Recompute rule firings from perturbed BASE variables.

    Written independently of the engine, and in particular it recomputes power
    and strain from the new torque rather than editing either directly - the
    coupling is part of the specification, not an implementation detail.
    """
    d = data()
    torque = np.maximum(0.0, d["torque_nm"] + torque_delta)
    power = torque * d["rotational_speed_rpm"] * RAD_PER_RPM
    strain = d["tool_wear_min"] * torque
    hdf = (d["temp_delta_k"] < HDF_TEMP) & (d["rotational_speed_rpm"] < HDF_SPEED)
    pwf = (power < PWF_LOW) | (power > PWF_HIGH)
    osf = strain > d["osf_threshold"]
    return {
        "HDF": int(hdf.sum()),
        "PWF": int(pwf.sum()),
        "OSF": int(osf.sum()),
        "any": int((hdf | pwf | osf).sum()),
    }


def safe_torque_band(rpm: float, wear: float, product_type: str = "L") -> tuple[float, float]:
    omega = rpm * RAD_PER_RPM
    lo = PWF_LOW / omega
    hi = PWF_HIGH / omega
    if wear > 0:
        hi = min(hi, OSF_THRESHOLD[product_type] / wear)
    return lo, hi


def invariants() -> dict[str, float]:
    d = data()
    return {
        "I1_violations": float((d["process_temp_k"] <= d["air_temp_k"]).sum()),
        "I2_mean_temp_delta": float(d["temp_delta_k"].mean()),
        "I2_sd_temp_delta": float(d["temp_delta_k"].std(ddof=1)),
        "I3_rpm_torque_corr": correlation("rotational_speed_rpm", "torque_nm"),
        "I4_mean_power": float(d["power_w"].mean()),
    }


# Registry so golden.yaml can name a reference by string.
REFERENCES = {
    "overall_failure_rate": overall_failure_rate,
    "failure_count": failure_count,
    "rate_by_product_type": rate_by_product_type,
    "count_by_product_type": count_by_product_type,
    "rate_by_rpm_quintile": rate_by_rpm_quintile,
    "cohort_mean": cohort_mean,
    "cohort_n": cohort_n,
    "cohens_d": cohens_d,
    "correlation": correlation,
    "mode_counts": mode_counts,
    "rule_agreement": rule_agreement,
    "orphan_failures": orphan_failures,
    "multi_mode_failures": multi_mode_failures,
    "deterministically_explained": deterministically_explained,
    "rnf_rollup": rnf_rollup,
    "twf_in_window": twf_in_window,
    "near_miss_counts": near_miss_counts,
    "osf_crossing_wear": osf_crossing_wear,
    "osf_cycles_to_crossing": osf_cycles_to_crossing,
    "counterfactual_mode_counts": counterfactual_mode_counts,
    "safe_torque_band": safe_torque_band,
    "invariants": invariants,
}


def resolve(spec: str | dict[str, Any]) -> Any:
    """Resolve a reference spec from the golden set.

    "overall_failure_rate"                      -> call it
    {"fn": "cohort_mean", "args": {...}}        -> call with arguments
    {"fn": "rate_by_rpm_quintile", "index": 0}  -> index the result
    {"fn": "rate_by_product_type", "key": "L"}  -> key the result
    """
    if isinstance(spec, str):
        return REFERENCES[spec]()
    fn = REFERENCES[spec["fn"]]
    value = fn(**spec.get("args", {}))
    if "index" in spec:
        value = value[spec["index"]]
    if "key" in spec:
        value = value[spec["key"]]
    return value
