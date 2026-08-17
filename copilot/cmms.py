"""CMMS (Computerised Maintenance Management System) integration.

Architecture note
-----------------
The ``WorkOrder`` schema here is intentionally identical to what SAP PM,
IBM Maximo, or a bespoke CMMS sends via REST or OData. Connecting a real
CMMS is replacing the ``_synthetic_source`` generator with an HTTP/Kafka
adapter — the feedback learning loop (``feedback.py``) is unchanged.

The synthetic generator is labelled ``SYNTHETIC`` everywhere it appears.
It drives from actual failure labels in the AI4I replay so the distribution
of confirmed / denied work orders is historically faithful, not invented.

Storage
-------
Work orders live in the existing DuckDB warehouse (``data/warehouse.duckdb``).
Two tables are created on first use:

    cmms_work_orders   — one row per raised alert
    cmms_kb_log        — one row per KB weight update driven by an outcome

Both tables are append-only; outcomes are written by the ``/cmms/…`` API.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

import duckdb

from copilot.ingest import connect

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


class AlertOutcome(StrEnum):
    """What the technician found when they investigated the alert."""
    CONFIRMED   = "confirmed"    # alert correct, mode matches
    WRONG_MODE  = "wrong_mode"   # machine did fail, but a different mode
    FALSE_ALARM = "false_alarm"  # machine was fine; alert was spurious
    INCONCLUSIVE = "inconclusive"  # could not determine root cause


@dataclass
class WorkOrder:
    """One maintenance work order, from alert to closure.

    This is the canonical exchange format between the copilot and the CMMS.
    ``raised_by`` is ``SYNTHETIC`` for demo-generated orders; a real CMMS
    adapter would set it to the integration account name.
    """
    id: str
    machine_id: str
    udi: int                        # cycle where the alert fired
    alert_mode: str                 # HDF | PWF | OSF | TWF | RNF
    raised_at: str                  # ISO-8601 UTC
    raised_by: str                  # source: "SYNTHETIC" | "copilot-api" | "sap-pm"

    # Filled when the work order is closed
    closed_at: str | None = None
    outcome: AlertOutcome | None = None
    confirmed_mode: str | None = None   # mode the technician attributed
    technician_id: str | None = None
    notes: str | None = None

    def is_open(self) -> bool:
        return self.closed_at is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "machine_id": self.machine_id,
            "udi": self.udi,
            "alert_mode": self.alert_mode,
            "raised_at": self.raised_at,
            "raised_by": self.raised_by,
            "closed_at": self.closed_at,
            "outcome": self.outcome,
            "confirmed_mode": self.confirmed_mode,
            "technician_id": self.technician_id,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


_DDL_WORK_ORDERS = """
CREATE TABLE IF NOT EXISTS cmms_work_orders (
    id              TEXT PRIMARY KEY,
    machine_id      TEXT NOT NULL,
    udi             INTEGER NOT NULL,
    alert_mode      TEXT NOT NULL,
    raised_at       TEXT NOT NULL,
    raised_by       TEXT NOT NULL,
    closed_at       TEXT,
    outcome         TEXT,
    confirmed_mode  TEXT,
    technician_id   TEXT,
    notes           TEXT
)
"""

_DDL_KB_LOG = """
CREATE TABLE IF NOT EXISTS cmms_kb_log (
    id              TEXT PRIMARY KEY,
    work_order_id   TEXT NOT NULL,
    mode            TEXT NOT NULL,
    variant         TEXT NOT NULL,
    delta_weight    REAL NOT NULL,
    reason          TEXT NOT NULL,
    recorded_at     TEXT NOT NULL
)
"""


class CMMSStore:
    """Thin DuckDB-backed repository for work orders and KB weight updates."""

    def __init__(self, db_path: str | None = None) -> None:
        self._conn: duckdb.DuckDBPyConnection = connect(db_path)
        self._conn.execute(_DDL_WORK_ORDERS)
        self._conn.execute(_DDL_KB_LOG)

    # ---- work orders -------------------------------------------------------

    def create(self, wo: WorkOrder) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO cmms_work_orders VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            [wo.id, wo.machine_id, wo.udi, wo.alert_mode,
             wo.raised_at, wo.raised_by,
             wo.closed_at, wo.outcome, wo.confirmed_mode,
             wo.technician_id, wo.notes],
        )

    def close(self, wo_id: str, outcome: AlertOutcome, **kwargs: Any) -> WorkOrder | None:
        wo = self.get(wo_id)
        if wo is None or not wo.is_open():
            return None
        now = _utcnow()
        self._conn.execute(
            """UPDATE cmms_work_orders
               SET closed_at=?, outcome=?, confirmed_mode=?, technician_id=?, notes=?
               WHERE id=?""",
            [now,
             outcome,
             kwargs.get("confirmed_mode"),
             kwargs.get("technician_id"),
             kwargs.get("notes"),
             wo_id],
        )
        return self.get(wo_id)

    def get(self, wo_id: str) -> WorkOrder | None:
        row = self._conn.execute(
            "SELECT * FROM cmms_work_orders WHERE id=?", [wo_id]
        ).fetchone()
        return _row_to_wo(row) if row else None

    def list(self, limit: int = 100, open_only: bool = False) -> list[WorkOrder]:
        q = "SELECT * FROM cmms_work_orders"
        if open_only:
            q += " WHERE closed_at IS NULL"
        q += f" ORDER BY raised_at DESC LIMIT {limit}"
        return [_row_to_wo(r) for r in self._conn.execute(q).fetchall()]

    def summary(self) -> dict[str, Any]:
        row = self._conn.execute("""
            SELECT
                COUNT(*)                                        AS total,
                SUM(CASE WHEN closed_at IS NULL THEN 1 END)    AS open,
                SUM(CASE WHEN outcome='confirmed'   THEN 1 END) AS confirmed,
                SUM(CASE WHEN outcome='false_alarm' THEN 1 END) AS false_alarms,
                SUM(CASE WHEN outcome='wrong_mode'  THEN 1 END) AS wrong_mode,
                SUM(CASE WHEN outcome='inconclusive' THEN 1 END) AS inconclusive
            FROM cmms_work_orders
        """).fetchone()
        total, open_, conf, fa, wm, inc = row
        precision = conf / (conf + fa) if (conf + fa) else None
        return {
            "total": total or 0,
            "open": open_ or 0,
            "confirmed": conf or 0,
            "false_alarms": fa or 0,
            "wrong_mode": wm or 0,
            "inconclusive": inc or 0,
            "precision": round(precision, 3) if precision is not None else None,
        }

    # ---- KB log ------------------------------------------------------------

    def log_kb_update(
        self,
        work_order_id: str,
        mode: str,
        variant: str,
        delta_weight: float,
        reason: str,
    ) -> None:
        self._conn.execute(
            "INSERT INTO cmms_kb_log VALUES (?,?,?,?,?,?,?)",
            [str(uuid.uuid4()), work_order_id, mode, variant,
             delta_weight, reason, _utcnow()],
        )

    def kb_log(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM cmms_kb_log ORDER BY recorded_at DESC LIMIT ?", [limit]
        ).fetchall()
        cols = ["id","work_order_id","mode","variant","delta_weight","reason","recorded_at"]
        return [dict(zip(cols, r)) for r in rows]


# --------------------------------------------------------------------------
# Synthetic generator (SYNTHETIC — labelled on every emitted order)
# --------------------------------------------------------------------------


def generate_from_replay(
    store: CMMSStore,
    limit: int = 30,
    *,
    auto_close_pct: float = 0.70,
) -> list[WorkOrder]:
    """Seed the CMMS with realistic work orders derived from the AI4I replay.

    ``auto_close_pct`` of generated orders are closed immediately with a
    realistic outcome distribution (faithful to 0 FP / 0 FN on deterministic
    modes). The rest remain open, simulating in-progress investigations.

    Every order carries ``raised_by="SYNTHETIC"`` so it is always
    distinguishable from orders that arrive from a real CMMS adapter.
    """
    import random
    rng = random.Random(42)

    conn = connect()
    rows = conn.execute("""
        SELECT udi, product_type, failure, machine_failure,
               hdf_failure, pwf_failure, osf_failure, twf_failure
        FROM   ai4i
        WHERE  (hdf_failure=1 OR pwf_failure=1 OR osf_failure=1 OR machine_failure=1)
        ORDER BY udi
        LIMIT ?
    """, [limit * 2]).fetchall()

    # Map variant → synthetic machine ID
    machine_map = {"L": "L-01", "M": "M-02", "H": "H-03"}
    mode_cols   = [("HDF", 4), ("PWF", 5), ("OSF", 6), ("TWF", 7)]

    created: list[WorkOrder] = []
    for row in rows:
        udi, variant, _failure, machine_fail, hdf, pwf, osf, twf = row
        machine_id = machine_map.get(variant, "L-01")

        active_modes = [m for m, idx in mode_cols if row[idx]]
        if not active_modes:
            active_modes = ["HDF"]  # fallback for machine_fail with no mode

        for mode in active_modes[:1]:   # one WO per cycle for clarity
            wo = WorkOrder(
                id         = _wo_id(udi, mode),
                machine_id = machine_id,
                udi        = udi,
                alert_mode = mode,
                raised_at  = _utcnow(),
                raised_by  = "SYNTHETIC",
            )
            store.create(wo)

            if rng.random() < auto_close_pct:
                if machine_fail:
                    outcome  = AlertOutcome.CONFIRMED
                    conf_mode = mode
                else:
                    # Deterministic modes have 0 FP — but synthetic noise: 3%
                    if rng.random() < 0.03:
                        outcome   = AlertOutcome.FALSE_ALARM
                        conf_mode = None
                    else:
                        outcome   = AlertOutcome.CONFIRMED
                        conf_mode = mode

                store.close(
                    wo.id,
                    outcome,
                    confirmed_mode = conf_mode,
                    technician_id  = f"TECH-{rng.randint(1,5):03d}",
                    notes          = "Synthetic outcome from AI4I replay",
                )

            created.append(store.get(wo.id) or wo)
            if len(created) >= limit:
                break
        if len(created) >= limit:
            break

    return created


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wo_id(udi: int, mode: str) -> str:
    raw = f"SYNTHETIC-{udi}-{mode}"
    return "WO-" + hashlib.sha1(raw.encode()).hexdigest()[:8].upper()


def _row_to_wo(row: tuple) -> WorkOrder:
    return WorkOrder(
        id             = row[0],
        machine_id     = row[1],
        udi            = row[2],
        alert_mode     = row[3],
        raised_at      = row[4],
        raised_by      = row[5],
        closed_at      = row[6],
        outcome        = AlertOutcome(row[7]) if row[7] else None,
        confirmed_mode = row[8],
        technician_id  = row[9],
        notes          = row[10],
    )
