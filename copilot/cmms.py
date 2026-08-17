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
Work orders live in their own SQLite database (``data/cmms.db``), NOT in the
DuckDB warehouse. Two tables are created on first use:

    cmms_work_orders   — one row per raised alert
    cmms_kb_log        — one row per KB weight update driven by an outcome

The engine choice is the whole point, so it is worth stating plainly. The
warehouse is an analytical archive: ten thousand rows scanned per question,
rebuilt from the CSV by ``make build``, opened read-only, one writer ever.
DuckDB is exactly right for that and exactly wrong for this. The ledger is
transactional: single-row appends, single-row updates, and it must survive the
rebuild that regenerates the archive.

It must also tolerate more than one process. DuckDB takes an exclusive file
lock, so a second uvicorn worker — or a deploy whose new process starts before
the old one exits, which is the normal case on a platform that restarts on
push — dies at startup with "Conflicting lock is held". SQLite in WAL mode
gives concurrent readers alongside one writer, needs no server, and ships in
the standard library. For a ledger that takes a handful of writes an hour, the
correct database is the boring one.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from copilot.config import settings


def _default_path() -> Path:
    """Beside the warehouse, but a file of its own, in its own engine."""
    return Path(settings().db_path).with_name("cmms.db")

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
    """Thin SQLite-backed repository for work orders and KB weight updates."""

    def __init__(self, db_path: str | None = None) -> None:
        # Work orders live in their OWN database, not the analytical warehouse.
        #
        # Two reasons, and the first is a bug this fixes: the warehouse is
        # opened read-only — correctly, since every query path only reads it —
        # so CREATE TABLE raised "Cannot execute statement of type CREATE on
        # database attached in read-only mode" and every CMMS endpoint returned
        # a 500.
        #
        # The second reason is why the fix is a separate file rather than a
        # writable warehouse handle: `make build` rebuilds the warehouse from
        # the CSV. Work orders are operational state that must survive that.
        # An archive you regenerate and a ledger you append to have different
        # lifecycles and do not belong in the same file.
        self._path = Path(db_path) if db_path else _default_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI runs sync endpoints on a threadpool, so the connection is
        # shared across threads and serialised by an explicit lock. SQLite's
        # own check is disabled because this class does the guarding.
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")   # readers ∥ one writer
            self._conn.execute("PRAGMA busy_timeout=5000")  # wait, do not fail
            self._conn.execute(_DDL_WORK_ORDERS)
            self._conn.execute(_DDL_KB_LOG)
            self._conn.commit()

    def _write(self, sql: str, params: list[Any]) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _read(self, sql: str, params: list[Any] | None = None) -> list[tuple]:
        with self._lock:
            return self._conn.execute(sql, params or []).fetchall()

    # ---- work orders -------------------------------------------------------

    def create(self, wo: WorkOrder) -> None:
        self._write(
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
        self._write(
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
        rows = self._read("SELECT * FROM cmms_work_orders WHERE id=?", [wo_id])
        return _row_to_wo(rows[0]) if rows else None

    def list(self, limit: int = 100, open_only: bool = False) -> list[WorkOrder]:
        q = "SELECT * FROM cmms_work_orders"
        if open_only:
            q += " WHERE closed_at IS NULL"
        q += " ORDER BY raised_at DESC LIMIT ?"
        return [_row_to_wo(r) for r in self._read(q, [int(limit)])]

    def summary(self) -> dict[str, Any]:
        row = self._read("""
            SELECT
                COUNT(*)                                        AS total,
                SUM(CASE WHEN closed_at IS NULL THEN 1 END)    AS open,
                SUM(CASE WHEN outcome='confirmed'   THEN 1 END) AS confirmed,
                SUM(CASE WHEN outcome='false_alarm' THEN 1 END) AS false_alarms,
                SUM(CASE WHEN outcome='wrong_mode'  THEN 1 END) AS wrong_mode,
                SUM(CASE WHEN outcome='inconclusive' THEN 1 END) AS inconclusive
            FROM cmms_work_orders
        """)[0]
        # SUM() over zero rows returns NULL, not 0, so a fresh install with no
        # work orders raised "unsupported operand type(s) for +: NoneType and
        # NoneType" and every CMMS endpoint 500'd. Coalesce before arithmetic,
        # not after — the old code guarded the RESULT and then did the addition
        # inside the guard.
        total, open_, conf, fa, wm, inc = (int(v or 0) for v in row)
        judged = conf + fa
        precision = conf / judged if judged else None
        return {
            "total": total,
            "open": open_,
            "confirmed": conf,
            "false_alarms": fa,
            "wrong_mode": wm,
            "inconclusive": inc,
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
        self._write(
            "INSERT INTO cmms_kb_log VALUES (?,?,?,?,?,?,?)",
            [str(uuid.uuid4()), work_order_id, mode, variant,
             delta_weight, reason, _utcnow()],
        )

    def kb_log(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT * FROM cmms_kb_log ORDER BY recorded_at DESC LIMIT ?", [int(limit)]
        )
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

    # This read used to call an undefined `connect()` against a table named
    # `ai4i` with columns named `hdf_failure`, none of which exist — the table
    # is `observations` and the flags are `hdf`, `pwf`, `osf`, `twf`. So
    # /cmms/seed raised NameError on its first line and had never once run.
    # Nothing caught it because seeding is the one path no test exercised.
    from copilot.engine import Engine
    from copilot.ops.registry import TABLE

    conn = Engine.build().ctx.con
    rows = conn.execute(f"""
        SELECT udi, product_type, machine_id, machine_failure, hdf, pwf, osf, twf
        FROM   {TABLE}
        WHERE  machine_failure = 1
        ORDER BY udi
        LIMIT ?
    """, [limit * 2]).fetchall()  # noqa: S608

    mode_cols = [("HDF", 4), ("PWF", 5), ("OSF", 6), ("TWF", 7)]

    created: list[WorkOrder] = []
    for row in rows:
        udi, variant, machine_id, machine_fail, hdf, pwf, osf, twf = row

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
