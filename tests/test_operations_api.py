"""The endpoints an operations screen depends on.

Every test here corresponds to a real 500 or a silently-empty response found by
calling the API rather than reading it. The RUL bugs are the instructive ones:
a swallowed exception turned a broken query into the sentence "no machine is at
risk", which is the most dangerous thing this system could say.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from copilot.api import app
from copilot.cmms import WorkOrder


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


class TestEveryOperationsEndpointAnswers:
    @pytest.mark.parametrize("path", [
        "/health", "/fleet", "/rul", "/cmms/work_orders", "/cmms/feedback",
    ])
    def test_it_returns_200(self, client, path):
        assert client.get(path).status_code == 200


class TestRemainingUsefulLife:
    """Two bugs hid behind one `except Exception: return None`."""

    def test_the_fleet_has_rul_estimates(self, client):
        """It returned {"machines": [], "total": 0} because the query said
        `FROM ai4i` and the table is `observations`.

        An operator reading an empty risk list concludes NOTHING IS AT RISK.
        An empty result that means "the query is broken" and one that means
        "nothing is wrong" are indistinguishable to a reader, which makes the
        swallowed exception worse than a crash would have been.
        """
        body = client.get("/rul").json()
        assert body["total"] > 0
        assert len(body["machines"]) == body["total"]

    def test_each_estimate_carries_a_horizon_and_a_status(self, client):
        for machine in client.get("/rul").json()["machines"]:
            assert machine["machine_id"]
            assert machine["status"]
            assert "expected_cycles" in machine

    def test_estimates_are_sorted_worst_first(self, client):
        """The screen shows the top of this list. If the sort is wrong the
        operator is shown the machines with the most time left."""
        cycles = [
            m["expected_cycles"] or 0 for m in client.get("/rul").json()["machines"]
        ]
        assert cycles == sorted(cycles)

    def test_the_thermal_state_is_read_not_assumed(self):
        """The second bug the exception hid: OperatingPoint takes the two
        thermocouples, not a derived delta, and the old code passed a nominal
        10 K. The HDF margin depends on it, so substituting a constant makes a
        figure the operator acts on independent of the machine."""
        import inspect

        from copilot import rul

        source = inspect.getsource(rul.machine_rul)
        assert "temp_delta_k" not in source
        assert "air_temperature_k" in source

    def test_an_unknown_machine_is_404_not_empty(self, client):
        assert client.get("/rul/NOPE-99").status_code == 404

    def test_every_machine_in_the_warehouse_has_an_estimate(self, client):
        """The roster was a hardcoded list of eight while the warehouse holds
        fifteen, so /rul/L-04 answered "machine not found" about a machine
        visible in the fleet rail with live margins beside it."""
        from copilot.ops.registry import TABLE
        from copilot.engine import Engine

        known = {
            r[0] for r in Engine.build().ctx.con.execute(
                f"SELECT DISTINCT machine_id FROM {TABLE}"  # noqa: S608
            ).fetchall()
        }
        estimated = {m["machine_id"] for m in client.get("/rul").json()["machines"]}
        assert estimated == known

    def test_the_conformal_correction_is_actually_computed(self):
        """The calibration query named a table and two columns that do not
        exist, and a bare `except Exception` turned the failure into a
        correction of 0.0 — indistinguishable from a correction that was
        computed and came out small.

        So the endpoint advertised a 90% CONFORMAL interval, the module was
        named for it, and every interval shipped was the raw inverse-Gaussian
        quantile. A swallowed exception here does not remove a feature; it
        leaves a false claim in place of one.
        """
        from copilot.rul import _conformal_correction

        assert _conformal_correction() > 0.0

    def test_the_interval_reflects_the_correction(self, client):
        """The visible consequence: intervals are wider than the nominal model,
        because the model was measurably overconfident at the crossing."""
        for machine in client.get("/rul").json()["machines"]:
            if machine["expected_cycles"] and machine["sd_cycles"]:
                span = machine["ci_hi"] - machine["ci_lo"]
                assert span > 2 * machine["sd_cycles"]


class TestWorkOrders:
    """Operational state, in its own database."""

    def test_a_fresh_install_reports_zero_rather_than_crashing(self, client):
        """SUM() over zero rows returns NULL, so `conf + fa` raised
        "unsupported operand type(s) for +: NoneType and NoneType" and every
        CMMS endpoint 500'd until the first work order existed."""
        summary = client.get("/cmms/work_orders").json()["summary"]
        assert summary["total"] >= 0
        assert summary["precision"] is None or 0.0 <= summary["precision"] <= 1.0

    def test_work_orders_do_not_live_in_the_analytical_warehouse(self):
        """Two reasons. The warehouse is opened read-only, so CREATE TABLE
        raised and every endpoint 500'd; and `make build` rebuilds it, which
        would destroy the ledger. An archive you regenerate and a ledger you
        append to have different lifecycles."""
        from copilot.cmms import CMMSStore
        from copilot.config import settings

        store = CMMSStore()
        assert str(store._path) != str(settings().db_path)
        assert store._path.suffix == ".db"   # SQLite, not the DuckDB archive

    def test_a_second_process_can_open_the_ledger(self, tmp_path):
        """DuckDB takes an exclusive file lock, so the API server holding the
        ledger open made every other process fail with "Conflicting lock is
        held" — including the test suite, and including the second uvicorn
        worker or the overlapping process during a rolling deploy.

        This is the deployment failure the test suite could not see while the
        server was stopped, which is precisely when it matters.
        """
        from copilot.cmms import CMMSStore

        path = tmp_path / "ledger.db"
        first = CMMSStore(str(path))
        second = CMMSStore(str(path))     # would raise IOException under DuckDB

        first.create(WorkOrder(
            id="WO-CONCURRENT", machine_id="L-01", udi=1, alert_mode="OSF",
            raised_at="2026-01-01T00:00:00+00:00", raised_by="first",
        ))
        assert second.get("WO-CONCURRENT") is not None

    def test_the_seeder_runs(self, tmp_path):
        """/cmms/seed called an undefined `connect()` against a table that does
        not exist. It raised NameError on its first statement and had never
        run — the one path no test exercised."""
        from copilot.cmms import CMMSStore, generate_from_replay

        store = CMMSStore(str(tmp_path / "seed.db"))
        created = generate_from_replay(store, limit=10)
        assert len(created) == 10
        assert all(wo.raised_by == "SYNTHETIC" for wo in created)

    def test_seeded_outcomes_are_faithful_to_the_labels(self, tmp_path):
        """Synthetic data may be synthetic; it may not be flattering. The
        outcome distribution is driven by the real failure labels, so precision
        reflects what the detector actually achieves rather than a number
        chosen to look good in a demo."""
        from copilot.cmms import CMMSStore, generate_from_replay

        store = CMMSStore(str(tmp_path / "faithful.db"))
        generate_from_replay(store, limit=25)
        summary = store.summary()
        assert summary["total"] == 25
        assert summary["confirmed"] > 0

    def test_a_work_order_round_trips(self, client, tmp_path):
        created = client.post("/cmms/work_orders", json={
            "machine_id": "L-01", "udi": 9016, "alert_mode": "OSF",
            "raised_by": "test",
        })
        assert created.status_code in (200, 201)
        listing = client.get("/cmms/work_orders").json()
        assert listing["summary"]["total"] >= 1


class TestFleet:
    """The live view the operations screen is built on."""

    def test_every_machine_carries_a_margin_and_a_binding_constraint(self, client):
        machines = client.get("/fleet").json()["machines"]
        assert machines
        for machine in machines:
            assert "worst_margin" in machine
            assert machine["binding"] in {"HDF", "PWF", "OSF", "TWF", None}
            assert machine["state"] in {"normal", "watch", "alert", "abstain"}

    def test_history_is_present_for_a_sparkline(self, client):
        for machine in client.get("/fleet").json()["machines"]:
            assert isinstance(machine.get("history"), list)
