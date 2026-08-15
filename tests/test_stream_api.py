"""Phases 6-7: streaming inference and the HTTP surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from copilot.api import app
from copilot.reliability import Uncertainty
from copilot.stream import AlertKind, StreamScorer, replay


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def scored():
    scorer = StreamScorer()
    alerts = [a for tick in replay(scorer=scorer) for a in tick.alerts]
    return scorer, alerts


class TestStreaming:
    def test_scores_every_cycle(self, scored):
        scorer, _ = scored
        assert scorer.ticks == 10000

    def test_throughput_clears_a_full_site(self, scored):
        """A 2,000-machine site at 1 Hz needs 2,000 events/sec."""
        import time

        scorer = StreamScorer()
        started = time.perf_counter()
        for _ in replay(limit=3000, scorer=scorer):
            pass
        rate = scorer.ticks / (time.perf_counter() - started)
        assert rate > 20_000, f"only {rate:,.0f} ticks/sec"

    def test_alerts_are_rationalised_not_flooded(self, scored):
        """Persistence, latching and a deadband. Without them a worn tool pages
        on every cycle for the rest of its life."""
        scorer, _ = scored
        per_thousand = scorer.alerts_raised / scorer.ticks * 1000
        assert per_thousand < 150, f"{per_thousand:.0f} alerts per 1,000 cycles"
        assert scorer.alerts_suppressed > 0

    def test_predicted_alerts_carry_a_lead_time_and_a_fix(self, scored):
        _, alerts = scored
        predicted = [a for a in alerts if a.kind is AlertKind.PREDICTED]
        assert predicted
        for alert in predicted[:20]:
            assert alert.lead_time_cycles is not None and alert.lead_time_cycles > 0
            assert alert.lead_time_min is not None
            assert alert.interval is not None
            assert alert.interval[0] <= alert.lead_time_cycles <= alert.interval[1]
            assert alert.fix  # a prescription, not just a severity

    def test_crossed_alerts_only_fire_on_real_violations(self, scored):
        _, alerts = scored
        for alert in [a for a in alerts if a.kind is AlertKind.CROSSED]:
            assert alert.margin < 0 or alert.mode == "PWF"

    def test_uncertain_sensor_produces_tickets_not_machine_alerts(self):
        """The whole point of three-state alerting."""
        scorer = StreamScorer(uncertainty=Uncertainty(torque_nm=8.0))
        alerts = [a for tick in replay(limit=3000, scorer=scorer) for a in tick.alerts]
        kinds = {a.kind for a in alerts}
        assert AlertKind.SENSOR_SUSPECT in kinds

    def test_forecast_uses_the_robust_track(self):
        """Projecting from one noisy sample makes the alarm chatter; torque has
        sd 10 N.m on a mean of 40."""
        scorer = StreamScorer()
        alerts = [a for tick in replay(limit=4000, scorer=scorer) for a in tick.alerts]
        predicted = [a for a in alerts if a.kind is AlertKind.PREDICTED]
        # Per machine, successive forecasts should not swing wildly.
        assert len(predicted) < 400

    def test_tick_serialises_for_transport(self, scored):
        scorer = StreamScorer()
        tick = next(iter(replay(limit=1, scorer=scorer)))
        payload = tick.as_dict()
        assert {"udi", "machine", "worst_margin", "verdict", "alerts"} <= payload.keys()


class TestAPI:
    def test_ask_returns_a_verified_answer(self, client):
        response = client.post(
            "/ask",
            json={"question": "Why are we seeing more failures at high rotational speeds?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["verified"] and not body["refused"]
        assert body["op"] == "rate"
        assert any(w["code"] == "premise_refuted" for w in body["warnings"])
        assert body["replay_handle"]

    def test_ask_keeps_session_context(self, client):
        client.delete("/session/pytest")
        client.post("/ask", json={"question": "What's the failure rate for L variants?",
                                  "session_id": "pytest"})
        second = client.post("/ask", json={"question": "What about H?", "session_id": "pytest"})
        assert second.json()["scope"].startswith("H")

    def test_ask_refuses_out_of_scope(self, client):
        body = client.post("/ask", json={"question": "purple monkey dishwasher"}).json()
        assert body["refused"]

    def test_evidence_is_available_on_request(self, client):
        body = client.post(
            "/ask", json={"question": "What's the overall failure rate?", "include_evidence": True}
        ).json()
        assert body["evidence"]["all.failures"]["value"] == 339

    def test_health_reports_both_gates(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["kb_calibration"]["healthy"]
        assert all(i["holds"] for i in body["invariants"])
        assert body["kb_version"] and body["data_version"]

    def test_envelope_returns_a_computed_boundary(self, client):
        body = client.get(
            "/envelope",
            params={"tool_wear_min": 214, "torque_nm": 58.1, "rotational_speed_rpm": 1412},
        ).json()
        assert body["current"]["safe"] == "no"
        assert body["current"]["fired"] == "OSF"
        assert body["safe_torque"]["min"] < body["safe_torque"]["max"]
        assert body["fix"]["action"] == "reduce torque"
        assert len(body["curve"]) >= 8
        for point in body["curve"]:
            assert point["torque_min"] < point["torque_max"]

    def test_alert_stream_emits_sse(self, client):
        import json

        with client.stream("GET", "/stream/alerts", params={"speed": 0, "limit": 900}) as response:
            assert response.status_code == 200
            events = [line for line in response.iter_lines() if line.startswith("data:")]
        assert events
        payload = json.loads(events[0][5:])
        assert {"kind", "machine", "mode"} <= payload.keys()

    def test_console_renders(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Margin Engine" in response.text


class TestEnvelopeProjection:
    """The window is a function of tool wear, so it closes over time. This turns
    'how much room do I have?' and 'how long have I got?' into one picture."""

    def test_window_narrows_as_the_tool_wears(self, client):
        body = client.get(
            "/envelope/projection",
            params={"rotational_speed_rpm": 1300, "tool_wear_min": 100, "horizon_cycles": 150},
        ).json()
        widths = [f["width"] for f in body["frames"]]
        assert widths == sorted(widths, reverse=True)
        assert body["shrink_pct"] > 50

    def test_binding_constraint_switches_from_overload_to_overstrain(self, client):
        """Early in a tool's life the drive limits you; late, the tool does."""
        body = client.get(
            "/envelope/projection",
            params={"rotational_speed_rpm": 1300, "tool_wear_min": 100, "horizon_cycles": 150},
        ).json()
        assert body["binding_switches"]
        assert body["frames"][0]["binding"] == "overload"
        assert body["frames"][-1]["binding"] == "overstrain"

    def test_closure_wear_is_where_the_ceiling_meets_the_floor(self, client):
        import math

        body = client.get(
            "/envelope/projection", params={"rotational_speed_rpm": 1400, "tool_wear_min": 150}
        ).json()
        omega = 1400 * 2 * math.pi / 60
        assert body["closure_wear_min"] == pytest.approx(11000 * omega / 3500, rel=1e-3)
        assert body["floor"] == pytest.approx(3500 / omega, rel=1e-3)

    def test_floor_is_fixed_while_the_ceiling_descends(self, client):
        body = client.get(
            "/envelope/projection",
            params={"rotational_speed_rpm": 1300, "tool_wear_min": 120, "horizon_cycles": 200},
        ).json()
        floors = {f["torque_min"] for f in body["frames"]}
        ceilings = [f["torque_max"] for f in body["frames"]]
        assert len(floors) == 1                      # the stall floor never moves
        assert ceilings == sorted(ceilings, reverse=True)

    def test_explorer_page_is_served(self, client):
        response = client.get("/explorer")
        assert response.status_code == 200
        assert "Operating Envelope" in response.text
        assert "ISA-101" in response.text  # the design rationale is in the source
