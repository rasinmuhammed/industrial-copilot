"""The knowledge base is the process definition, and a new process is a file.

Two claims are tested here, and the second is the one that matters commercially.

  1. The config-driven evaluator agrees exactly with the hand-written fast path
     on all 10,000 rows. Two independent implementations, so a bug has to occur
     twice, identically, to escape — the same discipline as evals/reference.py.

  2. A DIFFERENT process, with different metrics and different boundaries, runs
     correctly through the same code. No subclass, no branch, no redeploy. This
     is what turns "the architecture would scale to 1,000 factories" from an
     assertion in a README into something a reviewer can execute.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import duckdb
import pytest

from copilot.physics import (
    HDF_SPEED_LIMIT,
    HDF_TEMP_LIMIT,
    OSF_THRESHOLD,
    PWF_HIGH,
    PWF_LOW,
    OperatingPoint,
    evaluate,
)
from copilot.process_model import KB_PATH, load_process_model


@pytest.fixture(scope="module")
def model():
    return load_process_model()


@pytest.fixture(scope="module")
def rows():
    con = duckdb.connect()
    con.execute("CREATE VIEW t AS SELECT * FROM read_csv_auto('data/ai4i2020.csv')")
    return con.execute(
        'SELECT Type, "Air temperature [K]", "Process temperature [K]", '
        '"Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]" '
        "FROM t ORDER BY UDI"
    ).fetchall()


class TestSingleSourceOfTruth:
    """The literals are gone. The YAML is authoritative."""

    def test_physics_constants_come_from_the_knowledge_base(self, model):
        assert HDF_TEMP_LIMIT == model.limit("HDF", "temp_delta_k")
        assert HDF_SPEED_LIMIT == model.limit("HDF", "rotational_speed_rpm")
        assert PWF_LOW == model.limit("PWF", "power_w", "<")
        assert PWF_HIGH == model.limit("PWF", "power_w", ">")
        assert OSF_THRESHOLD == model.limits_by_type("OSF", "overstrain_min_nm")

    def test_physics_module_declares_no_boundary_literals(self):
        """Guard against the duplication creeping back in.

        `physics.py` and `failure_modes.yaml` both used to declare 8.6, 1380,
        3500, 9000 and the overstrain limits. Nothing detected divergence, in
        the one module where divergence would corrupt every answer downstream.
        """
        # Scan EXECUTABLE code only. Docstrings legitimately quote thresholds
        # when explaining behaviour — boundary_tolerance() discusses the 8.6 K
        # limit at length — and an earlier version of this test failed on that
        # prose, which would have pressured the next author to write a worse
        # comment rather than a better module.
        import ast

        tree = ast.parse(Path("copilot/physics.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue                                    # a docstring
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                assert node.value not in (
                    8.6, 1380, 1380.0, 3500, 3500.0, 9000, 9000.0,
                    11000, 11000.0, 12000, 12000.0, 13000, 13000.0,
                ), (
                    f"{node.value} is hard-coded in physics.py again; it belongs "
                    f"in failure_modes.yaml, which is the process definition"
                )


class TestGenericEvaluatorAgreesWithTheFastPath:
    """Two implementations, required to agree on every row."""

    def test_fired_modes_match_on_all_ten_thousand_rows(self, model, rows):
        mismatches = []
        for i, (ptype, air, proc, rpm, torque, wear) in enumerate(rows):
            point = OperatingPoint(
                air_temp_k=air, process_temp_k=proc, rotational_speed_rpm=rpm,
                torque_nm=torque, tool_wear_min=wear, product_type=ptype,
            )
            reading = {
                "temp_delta_k": proc - air,
                "rotational_speed_rpm": rpm,
                "power_w": point.power_w,
                "overstrain_min_nm": wear * torque,
            }
            generic = set(model.fires(reading, ptype))
            fast = set(evaluate(point).fired_modes())
            if generic != fast:
                mismatches.append((i, sorted(generic), sorted(fast)))
        assert not mismatches, f"{len(mismatches)} disagreements, first: {mismatches[:3]}"

    def test_margin_sign_marks_exactly_the_firing_conditions(self, model, rows):
        """A condition holds if and only if its margin is negative."""
        for ptype, air, proc, rpm, torque, wear in rows[:2000]:
            reading = {
                "temp_delta_k": proc - air,
                "rotational_speed_rpm": rpm,
                "power_w": torque * rpm * 2 * 3.141592653589793 / 60,
                "overstrain_min_nm": wear * torque,
            }
            margins = model.margins(reading, ptype)
            osf = margins["OSF.overstrain_min_nm"]
            fired = "OSF" in model.fires(reading, ptype)
            assert fired == (osf < 0)


class TestASecondProcessNeedsNoCode:
    """Onboarding factory #2 is a file, not a build."""

    @pytest.fixture(scope="class")
    def other_process(self, tmp_path_factory) -> Path:
        """A hydraulic press: different metrics, different limits, same engine."""
        path = tmp_path_factory.mktemp("kb") / "press.yaml"
        path.write_text(textwrap.dedent("""
            version: 1
            process:
              wear_rate_per_cycle: {STD: 1.0}
              base_variables: [ram_pressure_bar, oil_temp_c, cycle_count]
            modes:
              - code: OVP
                name: Overpressure
                kind: deterministic
                predicate:
                  metric: ram_pressure_bar
                  op: ">"
                  value: 250
                  unit: bar
              - code: OIL
                name: Oil Degradation
                kind: deterministic
                predicate:
                  all_of:
                    - {metric: oil_temp_c, op: ">", value: 80, unit: C}
                    - {metric: cycle_count, op: ">", value: 50000, unit: count}
              - code: SEAL
                name: Seal Wear
                kind: deterministic
                predicate:
                  metric: cycle_count
                  op: ">"
                  value_by_type: {STD: 80000, HD: 120000}
                  unit: count
        """).strip())
        return path

    def test_a_different_process_loads(self, other_process):
        model = load_process_model(other_process)
        assert {m.code for m in model.modes} == {"OVP", "OIL", "SEAL"}
        assert model.limit("OVP", "ram_pressure_bar") == 250.0

    def test_a_different_process_evaluates_correctly(self, other_process):
        model = load_process_model(other_process)
        safe = {"ram_pressure_bar": 180, "oil_temp_c": 60, "cycle_count": 10_000}
        assert model.fires(safe, "STD") == []

        burst = {"ram_pressure_bar": 300, "oil_temp_c": 60, "cycle_count": 10_000}
        assert model.fires(burst, "STD") == ["OVP"]

        # Conjunctive: hot oil alone is not enough, it must also be old.
        hot = {"ram_pressure_bar": 180, "oil_temp_c": 95, "cycle_count": 10_000}
        assert "OIL" not in model.fires(hot, "STD")
        hot_and_old = {"ram_pressure_bar": 180, "oil_temp_c": 95, "cycle_count": 60_000}
        assert "OIL" in model.fires(hot_and_old, "STD")

    def test_per_variant_limits_work_for_a_process_that_never_saw_ai4i(
        self, other_process
    ):
        model = load_process_model(other_process)
        worn = {"ram_pressure_bar": 180, "oil_temp_c": 60, "cycle_count": 100_000}
        assert "SEAL" in model.fires(worn, "STD")      # limit 80,000
        assert "SEAL" not in model.fires(worn, "HD")   # limit 120,000

    def test_margins_are_signed_correctly_for_the_new_process(self, other_process):
        model = load_process_model(other_process)
        reading = {"ram_pressure_bar": 200, "oil_temp_c": 60, "cycle_count": 10_000}
        margins = model.margins(reading, "STD")
        assert margins["OVP.ram_pressure_bar"] == pytest.approx(50.0)  # 250 - 200
        assert margins["SEAL.cycle_count"] == pytest.approx(70_000.0)

    def test_the_ai4i_definition_is_not_special_cased_anywhere(self, other_process):
        """Both processes go through identical code, so neither is privileged."""
        ai4i = load_process_model(KB_PATH)
        press = load_process_model(other_process)
        assert type(ai4i) is type(press)
        assert ai4i.deterministic_modes and press.deterministic_modes


class TestMisconfigurationFailsLoudly:
    """A bad config must not silently produce plausible numbers."""

    def test_unknown_operator_is_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            "version: 1\nmodes:\n  - code: X\n    kind: deterministic\n"
            "    predicate: {metric: p, op: '~=', value: 1}\n"
        )
        model = load_process_model(path)
        with pytest.raises(ValueError, match="unsupported operator"):
            model.fires({"p": 1.0})

    def test_missing_variant_limit_is_named_not_guessed(self, tmp_path):
        path = tmp_path / "bad2.yaml"
        path.write_text(
            "version: 1\nmodes:\n  - code: X\n    kind: deterministic\n"
            "    predicate: {metric: p, op: '>', value_by_type: {A: 1}}\n"
        )
        model = load_process_model(path)
        with pytest.raises(ValueError, match="no p limit declared"):
            model.fires({"p": 5.0}, "B")
