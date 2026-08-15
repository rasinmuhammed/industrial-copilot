"""CSV -> DuckDB, with the physics of the process materialised as columns.

The central design decision of this project lives here. The AI4I failure modes
are not statistical patterns to be learned; they are documented deterministic
rules over derived quantities. We compute those quantities once, at ingest, so
that every downstream question -- "why did it fail", "how close are we to the
limit", "what if torque drops 5 Nm" -- becomes an indexed scan rather than an
inference.

Materialised here:
  * the three physical derivations (temp_delta, power, overstrain)
  * the per-variant overstrain threshold
  * a signed margin to every rule boundary   <- makes root cause quantitative
  * per-row mode firing flags recomputed from the rules (not copied from labels)
  * a synthetic timeline and virtual fleet overlay (clearly flagged as such)
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb

from copilot.config import Settings, settings

# The published CSV has a BOM and bracketed unit suffixes in its headers.
RAW_TO_CLEAN = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "product_type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

RAD_PER_RPM = 2 * math.pi / 60


def _build_sql(cfg: Settings) -> str:
    select_raw = ",\n        ".join(f'"{raw}" AS {clean}' for raw, clean in RAW_TO_CLEAN.items())
    n_mach = cfg.virtual_machines_per_type
    return f"""
    CREATE OR REPLACE TABLE observations AS
    WITH raw AS (
        SELECT {select_raw}
        FROM read_csv(?, header = true, auto_detect = true)
    ),
    physics AS (
        SELECT
            *,
            -- Derived physical quantities (documented derivations).
            process_temperature_k - air_temperature_k                    AS temp_delta_k,
            torque_nm * rotational_speed_rpm * {RAD_PER_RPM}             AS power_w,
            tool_wear_min * torque_nm                                    AS overstrain_min_nm,
            CASE product_type WHEN 'L' THEN 11000.0
                              WHEN 'M' THEN 12000.0
                              WHEN 'H' THEN 13000.0 END                  AS osf_threshold_min_nm
        FROM raw
    ),
    enriched AS (
        SELECT
            *,
            -- Signed margins to every documented boundary. Negative == violated.
            temp_delta_k - 8.6                                           AS temp_delta_margin_k,
            rotational_speed_rpm - 1380.0                                AS speed_margin_rpm,
            power_w - 3500.0                                             AS power_low_margin_w,
            9000.0 - power_w                                             AS power_high_margin_w,
            osf_threshold_min_nm - overstrain_min_nm                     AS overstrain_margin_min_nm,
            200.0 - tool_wear_min                                        AS wear_to_window_min,

            -- Mode predicates RECOMPUTED from the knowledge base, never copied
            -- from the published labels. Divergence between these and the
            -- labels is what the rule audit measures.
            (temp_delta_k < 8.6 AND rotational_speed_rpm < 1380)         AS hdf_rule,
            (power_w < 3500 OR power_w > 9000)                           AS pwf_rule,
            (overstrain_min_nm > osf_threshold_min_nm)                   AS osf_rule,
            (tool_wear_min BETWEEN 200 AND 240)                          AS twf_window,

            -- SYNTHETIC OVERLAYS. Documented in README > Assumptions.
            TIMESTAMP '{cfg.epoch}' + (udi - 1) * INTERVAL {cfg.takt_seconds} SECOND AS ts,
            product_type || '-' ||
                lpad(CAST(((udi - 1) % {n_mach}) + 1 AS VARCHAR), 2, '0')  AS machine_id
        FROM physics
    ),
    scored AS (
        SELECT
            *,
            CASE WHEN hour(ts) < 8 THEN 'A' WHEN hour(ts) < 16 THEN 'B' ELSE 'C' END AS shift,

            -- Distance to each RULE firing, normalised by its own threshold.
            -- The aggregation differs by rule structure, and getting this wrong
            -- is a real modelling error: HDF needs BOTH conditions violated, so
            -- its binding constraint is the LARGER margin. PWF fires on EITHER
            -- side, so its binding constraint is the SMALLER one.
            greatest(temp_delta_margin_k / 8.6,
                     speed_margin_rpm / 1380.0)              AS hdf_distance,
            least(power_low_margin_w / 3500.0,
                  power_high_margin_w / 9000.0)              AS pwf_distance,
            overstrain_margin_min_nm / osf_threshold_min_nm  AS osf_distance
        FROM enriched
    )
    SELECT
        *,
        -- One "how close to failing is this cycle" number. Self-validating:
        -- exactly the 287 deterministically-explained failures are negative,
        -- and no healthy row is.
        least(hdf_distance, pwf_distance, osf_distance) AS worst_normalised_margin
    FROM scored
    """


def build(csv_path: Path | None = None, db_path: Path | None = None) -> Path:
    """(Re)build the warehouse. Idempotent."""
    cfg = settings()
    csv_path = Path(csv_path or cfg.csv_path)
    db_path = Path(db_path or cfg.db_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Download ai4i2020.csv from "
            "https://archive.ics.uci.edu/dataset/601 and place it there, "
            "or set COPILOT_CSV."
        )
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    try:
        con.execute(_build_sql(cfg), [str(csv_path)])
        # Indexes matter once this is a real fleet table; harmless at 10k rows.
        for col in ("machine_id", "product_type", "ts", "machine_failure"):
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_obs_{col} ON observations({col})")
        n = con.execute("SELECT count(*) FROM observations").fetchone()[0]
        con.execute("CREATE OR REPLACE TABLE meta AS SELECT ? AS rows, now() AS built_at", [n])
    finally:
        con.close()
    return db_path


def connect(db_path: Path | None = None, *, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open the warehouse, building it first if it does not exist."""
    cfg = settings()
    db_path = Path(db_path or cfg.db_path)
    if not db_path.exists():
        build(db_path=db_path)
    return duckdb.connect(str(db_path), read_only=read_only)


if __name__ == "__main__":  # pragma: no cover
    path = build()
    con = duckdb.connect(str(path), read_only=True)
    rows = con.execute("SELECT count(*) FROM observations").fetchone()[0]
    fails = con.execute("SELECT count(*) FROM observations WHERE machine_failure = 1").fetchone()[0]
    print(f"built {path}  rows={rows}  failures={fails}")
