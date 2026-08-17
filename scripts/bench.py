#!/usr/bin/env python3
"""Throughput benchmark for margin evaluation.

Backs the performance claims in docs/09-STREAMING.md and the README. The point
is not that the code is fast - it is that there is no model to run at inference
time, so speed is structural.

    python scripts/bench.py
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

CSV = Path(__file__).resolve().parent.parent / "data" / "ai4i2020.csv"
OSF_THRESHOLD = {"L": 11000.0, "M": 12000.0, "H": 13000.0}
RAD_PER_RPM = 2 * math.pi / 60

# A 2,000-machine site sampling at 1 Hz.
SITE_EVENTS_PER_SEC = 2_000


def margins(air, proc, rpm, torque, wear, osf_th):
    """The complete online computation. Five signed scalars, no state."""
    dT = proc - air
    power = torque * rpm * RAD_PER_RPM
    strain = wear * torque
    return (
        dT - 8.6,
        rpm - 1380.0,
        power - 3500.0,
        9000.0 - power,
        osf_th - strain,
    )


def interval_margins(air, proc, rpm, tq_lo, tq_hi, wear, osf_th):
    """Interval-valued variant: uncertain torque -> margin bounds."""
    lo = margins(air, proc, rpm, tq_hi, wear, osf_th)
    hi = margins(air, proc, rpm, tq_lo, wear, osf_th)
    return lo, hi


def load():
    if not CSV.exists():
        sys.exit(f"Dataset not found at {CSV}. See README > Setup.")
    rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
    return (
        np.array([float(r["Air temperature [K]"]) for r in rows]),
        np.array([float(r["Process temperature [K]"]) for r in rows]),
        np.array([float(r["Rotational speed [rpm]"]) for r in rows]),
        np.array([float(r["Torque [Nm]"]) for r in rows]),
        np.array([float(r["Tool wear [min]"]) for r in rows]),
        np.array([OSF_THRESHOLD[r["Type"]] for r in rows]),
    )


def timed(fn, iterations):
    fn()  # warm
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) / iterations


def main() -> int:
    air, proc, rpm, torque, wear, osf_th = load()
    n = len(air)

    print("=" * 74)
    print("MARGIN EVALUATION THROUGHPUT")
    print("=" * 74)

    vec = timed(lambda: margins(air, proc, rpm, torque, wear, osf_th), 200)
    vec_rate = n / vec
    print(f"\n  vectorised    {vec * 1e3:>8.3f} ms / {n:,} samples"
          f"   -> {vec_rate / 1e6:>8.1f} M samples/sec/core")

    scalar = timed(lambda: margins(298.1, 308.6, 1551.0, 42.8, 0.0, 11000.0), 20_000)
    scalar_rate = 1 / scalar
    print(f"  per-event     {scalar * 1e6:>8.2f} us / sample"
          f"          -> {scalar_rate / 1e3:>8.0f} k events/sec/core")

    iv = timed(lambda: interval_margins(298.1, 308.6, 1551.0, 40.0, 45.0, 0.0, 11000.0), 20_000)
    iv_rate = 1 / iv
    print(f"  interval      {iv * 1e6:>8.2f} us / sample"
          f"          -> {iv_rate / 1e3:>8.0f} k events/sec/core")

    print("\n" + "-" * 74)
    print("HEADROOM")
    print("-" * 74)
    print(f"\n  A 2,000-machine site at 1 Hz needs {SITE_EVENTS_PER_SEC:,} events/sec.\n")
    print(f"    scalar path    {scalar_rate / SITE_EVENTS_PER_SEC:>8.0f}x headroom on ONE core")
    print(f"    interval path  {iv_rate / SITE_EVENTS_PER_SEC:>8.0f}x headroom on ONE core")

    fleet = 1_000 * SITE_EVENTS_PER_SEC  # 1,000 factories
    print(f"\n  1,000 factories = {fleet / 1e6:.1f} M events/sec total.")
    print(f"    cores required (interval path): {fleet / iv_rate:.1f}")

    print("\n" + "=" * 74)
    print("Speed is not the result of optimisation. There is no model to run at")
    print("inference time - intelligence lives in the knowledge base, built")
    print("offline. Online is arithmetic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
