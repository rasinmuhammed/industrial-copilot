"""A learned residual detector for the relationships physics does not describe.

WHY THERE IS ANY NEURAL NETWORK HERE AT ALL
-------------------------------------------
Everything else in this system is deliberately not learned. Margins are
arithmetic, the observer is a Kalman filter, drift detection is a CUSUM from
1954. That is not conservatism: for a documented boundary an exact distance
beats an estimate, and a learned model would replace a known quantity with a
guess.

But the physics only reaches as far as the relations somebody wrote down. This
dataset has two - the thermal coupling and wear monotonicity - and a real plant
has a handful more. Everything outside them is unmodelled, and "unmodelled" is
exactly where a density estimator earns its place: it learns what NORMAL looks
like jointly, across all channels at once, without anyone having to state the
relationship.

WHAT IT IS, AND WHAT IT IS NOT
------------------------------
An autoencoder trained on healthy cycles only. It compresses the standardised
channel vector through a narrow bottleneck and reconstructs it; a point the
model cannot rebuild is a point unlike anything it was trained on. The score is
reconstruction error.

It is NOT a sequence model, and that is a finding rather than a shortcut. AI4I
has 1,490 tool-life segments with a median length of six cycles and **not one**
reaching twenty. There are no degradation trajectories to learn from. An LSTM or
temporal transformer fitted to this data would be fitting noise, and a published
RUL curve from it would be an artefact. What can honestly be learned from
independent cycles is their joint distribution at an instant.

WHY NUMPY AND NOT A FRAMEWORK
-----------------------------
The network is 5→3→5. Importing a deep-learning framework to run roughly fifty
multiply-accumulates would add hundreds of megabytes to an image that currently
holds seven light dependencies, for a model that fits in a few hundred floats.
Forward and backward passes are written out explicitly here - it is thirty lines
of arithmetic, it runs in microseconds, and it deploys anywhere Python does,
including the edge boxes where this would actually live.

HOW IT IS JUDGED
----------------
Against the baselines, honestly. Published evaluation of time-series foundation
models found anomaly-detection performance "does not significantly differ to
simple one-liner baselines", so the burden is on the learned model to beat PCA
and to beat the physics it supplements. `scripts/bench_neural.py` runs that
comparison, and if it loses, that is the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["ResidualDetector", "DetectorScore", "CHANNELS"]

CHANNELS: tuple[str, ...] = (
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)

# ── Error budget, stated rather than tuned, as everywhere else here. ─────────
FALSE_ALARM_RATE = 1e-3      # P(a healthy point is flagged)
BOTTLENECK = 3               # < len(CHANNELS), or it learns the identity
EPOCHS = 400
LEARNING_RATE = 0.05
SEED = 20260817


@dataclass(frozen=True, slots=True)
class DetectorScore:
    """One point's reconstruction error and what it means."""

    error: float
    threshold: float
    per_channel: tuple[float, ...]

    @property
    def anomalous(self) -> bool:
        return self.error > self.threshold

    @property
    def worst_channel(self) -> str:
        """Which channel the model failed hardest to reconstruct.

        Not an attribution - a reconstruction error is a joint property and
        blaming one input is a well-known way to over-read an autoencoder. It is
        a starting point for a human, and the docstring says so because the
        number will otherwise be read as a diagnosis.
        """
        return CHANNELS[int(np.argmax(self.per_channel))]


@dataclass(slots=True)
class ResidualDetector:
    """Autoencoder over the channel vector. Learns normal, flags unlike-normal."""

    bottleneck: int = BOTTLENECK
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    w1: np.ndarray | None = None
    b1: np.ndarray | None = None
    w2: np.ndarray | None = None
    b2: np.ndarray | None = None
    threshold: float = math.inf
    trained_on: int = 0

    # -- training ----------------------------------------------------------
    def fit(self, healthy: np.ndarray, epochs: int = EPOCHS) -> "ResidualDetector":
        """Fit on HEALTHY cycles only.

        Training on everything would teach the model that failures are normal,
        which is the commonest way an anomaly detector is quietly ruined: it
        reconstructs the faults perfectly and flags nothing.
        """
        x = np.asarray(healthy, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(CHANNELS):
            raise ValueError(f"expected (n, {len(CHANNELS)}), got {x.shape}")

        self.mean = x.mean(axis=0)
        # Floored so a constant channel cannot divide by zero and produce a
        # detector that flags everything.
        self.scale = np.maximum(x.std(axis=0), 1e-9)
        z = (x - self.mean) / self.scale

        rng = np.random.default_rng(SEED)
        n_in, n_hid = len(CHANNELS), self.bottleneck
        # Xavier: keeps activations in tanh's responsive range at init, so the
        # first epochs are not spent escaping saturation.
        self.w1 = rng.normal(0, math.sqrt(1.0 / n_in), (n_in, n_hid))
        self.b1 = np.zeros(n_hid)
        self.w2 = rng.normal(0, math.sqrt(1.0 / n_hid), (n_hid, n_in))
        self.b2 = np.zeros(n_in)

        n = len(z)
        for _ in range(epochs):
            h = np.tanh(z @ self.w1 + self.b1)
            out = h @ self.w2 + self.b2
            err = out - z                                  # d(MSE)/d(out)

            g_w2 = h.T @ err / n
            g_b2 = err.mean(axis=0)
            d_h = (err @ self.w2.T) * (1.0 - h * h)        # tanh'
            g_w1 = z.T @ d_h / n
            g_b1 = d_h.mean(axis=0)

            self.w2 -= LEARNING_RATE * g_w2
            self.b2 -= LEARNING_RATE * g_b2
            self.w1 -= LEARNING_RATE * g_w1
            self.b1 -= LEARNING_RATE * g_b1

        # Threshold from the stated false-alarm rate on the training set, not
        # from a round number somebody liked.
        errors = self._errors(z)
        self.threshold = float(np.quantile(errors, 1.0 - FALSE_ALARM_RATE))
        self.trained_on = n
        return self

    # -- inference ---------------------------------------------------------
    def score(self, reading: dict[str, float] | np.ndarray) -> DetectorScore:
        """Score one point. Microseconds: two small matrix products."""
        if self.w1 is None:
            raise RuntimeError("detector is not trained")
        if isinstance(reading, dict):
            vector = np.array([float(reading[c]) for c in CHANNELS])
        else:
            vector = np.asarray(reading, dtype=np.float64)
        z = (vector - self.mean) / self.scale
        h = np.tanh(z @ self.w1 + self.b1)
        out = h @ self.w2 + self.b2
        per_channel = (out - z) ** 2
        return DetectorScore(
            error=float(per_channel.mean()),
            threshold=self.threshold,
            per_channel=tuple(float(v) for v in per_channel),
        )

    def score_many(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return self._errors(z)

    def _errors(self, z: np.ndarray) -> np.ndarray:
        h = np.tanh(z @ self.w1 + self.b1)
        out = h @ self.w2 + self.b2
        return ((out - z) ** 2).mean(axis=1)

    @property
    def parameters(self) -> int:
        if self.w1 is None:
            return 0
        return sum(a.size for a in (self.w1, self.b1, self.w2, self.b2))


@dataclass(slots=True)
class PCABaseline:
    """Linear reconstruction, the baseline the network has to beat.

    Included because it is the honest comparison. Published evaluation of
    time-series foundation models found anomaly detection "does not
    significantly differ to simple one-liner baselines", so a learned nonlinear
    model that does not beat a linear projection has not earned its place.
    """

    components: int = BOTTLENECK
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    basis: np.ndarray | None = None
    threshold: float = math.inf

    def fit(self, healthy: np.ndarray) -> "PCABaseline":
        x = np.asarray(healthy, dtype=np.float64)
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-9)
        z = (x - self.mean) / self.scale
        _u, _s, vt = np.linalg.svd(z, full_matrices=False)
        self.basis = vt[: self.components]
        errors = self.score_many(x)
        self.threshold = float(np.quantile(errors, 1.0 - FALSE_ALARM_RATE))
        return self

    def score_many(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        rebuilt = (z @ self.basis.T) @ self.basis
        return ((rebuilt - z) ** 2).mean(axis=1)
