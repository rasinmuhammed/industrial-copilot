"""A hazard that rises with wear, constrained to be physically possible.

The tool-wear mode was reported as one number: 3.6% inside the documented
window. That is the average of a curve running from 0.2% to 16%, and telling an
operator at 235 minutes the same thing as one at 195 is the difference between a
warning and a shrug.
"""

from __future__ import annotations

import duckdb
import pytest

from copilot.config import settings
from copilot.hazard import fit_hazard, isotonic


@pytest.fixture(scope="module")
def wear_curve():
    con = duckdb.connect(str(settings().db_path), read_only=True)
    rows = con.execute(
        "SELECT tool_wear_min, TWF FROM observations WHERE tool_wear_min >= 180"
    ).fetchall()
    return fit_hazard(
        [r[0] for r in rows], [bool(r[1]) for r in rows], lower=180, upper=260
    )


class TestIsotonicRegression:
    """Exact, deterministic, and physically constrained by construction."""

    def test_a_dip_is_pooled_and_everything_else_survives(self):
        """The bug this test exists for: an earlier implementation replicated
        each PAVA block by its WEIGHT rather than by the number of positions it
        spanned. With exposure weights in the hundreds that produced a list
        thousands long, truncated back to the first block — so every fitted
        value came out as that block's mean.

        A flat line. Which passed the monotonicity check, because a constant is
        monotone. The output looked plausible and was entirely wrong.
        """
        observed = [0.0, 0.04, 0.05, 0.09, 0.16, 0.08, 0.50]
        weights = [470, 407, 253, 96, 31, 12, 2]
        fitted = isotonic(observed, weights)

        assert len(fitted) == len(observed)
        assert len({round(v, 6) for v in fitted}) > 1        # not collapsed
        assert fitted[:4] == pytest.approx(observed[:4])     # untouched
        assert fitted[4] == fitted[5]                        # the dip pooled
        assert fitted[4] == pytest.approx(0.138, abs=0.002)

    def test_the_result_is_always_non_decreasing(self):
        import random

        rng = random.Random(4)
        for _ in range(50):
            n = rng.randint(2, 30)
            values = [rng.uniform(0, 1) for _ in range(n)]
            weights = [rng.uniform(1, 500) for _ in range(n)]
            fitted = isotonic(values, weights)
            assert all(a <= b + 1e-12 for a, b in zip(fitted, fitted[1:]))

    def test_an_already_monotone_input_is_left_alone(self):
        values = [0.1, 0.2, 0.3]
        assert isotonic(values, [10, 10, 10]) == pytest.approx(values)

    def test_it_is_deterministic(self):
        """No learning rate, no seed, no local minimum. Fit the same data twice
        and get the same curve, which matters for something an engineer acts on.
        """
        values, weights = [0.3, 0.1, 0.5], [5, 50, 5]
        assert isotonic(values, weights) == isotonic(values, weights)


class TestTheCurveOnRealData:
    def test_the_hazard_actually_rises(self, wear_curve):
        """0.2% at 195 minutes against 14.0% at 235. Reporting the 3.6% average
        throws that away."""
        low = wear_curve.at(195)
        high = wear_curve.at(235)
        assert low.fitted < 0.01
        assert high.fitted > 0.10

    def test_it_never_says_a_tool_gets_safer_with_use(self, wear_curve):
        """A wearout hazard is non-decreasing. This is a fact about the physics,
        so the model is chosen to be incapable of contradicting it — rather than
        fitted freely and checked afterwards."""
        fitted = [p.fitted for p in wear_curve.points]
        assert all(a <= b + 1e-12 for a, b in zip(fitted, fitted[1:]))

    def test_a_thin_band_is_reported_as_thin(self, wear_curve):
        """The top bands hold a handful of cycles. The estimate is still given —
        with an interval that makes the thinness obvious rather than a point
        that hides it."""
        sparse = [p for p in wear_curve.points if not p.reportable]
        assert sparse
        for point in sparse:
            assert point.width > 0.15
        assert "direction, not a figure" in wear_curve.sentence(255)

    def test_a_dense_band_states_its_evidence(self, wear_curve):
        text = wear_curve.sentence(205)
        assert "%" in text and "cycles" in text
        assert "non-decreasing" in text

    def test_it_covers_the_documented_window(self, wear_curve):
        """TWF is documented between 200 and 240 minutes; the curve must speak
        to every part of it."""
        for wear in (200, 210, 220, 230, 239):
            assert wear_curve.at(wear) is not None

    def test_degenerate_input_does_not_raise(self):
        assert fit_hazard([], []).points == ()
        single = fit_hazard([210.0], [True])
        assert single.points and single.points[0].n == 1
