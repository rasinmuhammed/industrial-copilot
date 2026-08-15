"""Tier 2 — a store of plans that were verified to work.

Every answered question emits a triple:

    (question, plan, did_it_verify?)

That label is **free and objective**. The plan either validated, executed and
passed the numeric verifier, or it did not — no human annotation, no preference
model, no judge. It is the reward signal a planner actually needs, and it costs
nothing to collect.

So: keep the verified ones, embed the question, and retrieve the nearest match
next time. Add one exemplar and it is live on the very next query. This is
continual learning with **no training step at all**, and therefore no
catastrophic forgetting, no eval drift, and nothing to roll back except a row.

What is stored is the plan's *shape*, not the plan: op, metrics, dimensions,
binning, grain. Entity-specific filters are deliberately dropped and re-derived
from the new question, because "why did cycle 9016 fail" and "why did cycle 4045
fail" are the same analysis pointed at different rows.

The embedder is character-and-word hashing rather than a neural encoder: it is
deterministic, dependency-free, runs in microseconds, and needs no model
artifact shipped to every site. A learned encoder can be swapped in behind the
same interface when one is worth the deployment cost.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from copilot.ir import AnalysisPlan, PlanError, parse_plan
from copilot.planner.cache import normalise
from copilot.session import SessionState

__all__ = [
    "rebind",
    "plan_shape",
    "Embedder",
    "HashingEmbedder",
    "Exemplar",
    "ExemplarStore",
    "REUSE_THRESHOLD",
    "EXACT_THRESHOLD",
]

# Thresholds calibrated by measurement, not taste. Over a set of hand-written
# paraphrase pairs and unrelated pairs (scripts/calibrate_exemplars.py):
#
#     paraphrases   0.397 .. 0.863   (min 0.397)
#     unrelated     0.000 .. 0.128   (max 0.128)
#
# leaving a wide separation gap. 0.55 sits ~4.3x above the highest unrelated
# score while capturing the clear paraphrases; weak ones fall through and are
# escalated, which is the right asymmetry. Reusing a wrong shape cannot produce
# a wrong NUMBER — validation and the verifier still run — but it would answer
# the wrong question, and an escalation only costs latency.
REUSE_THRESHOLD = 0.55
# Above this the questions differ only in entities, so the exemplar's metric
# selection is reused verbatim instead of being re-derived from the new wording.
EXACT_THRESHOLD = 0.90

DIM = 512
_WORD = re.compile(r"[a-z0-9_]+")

# Fields that describe the ANALYSIS. Everything else is entity-specific and is
# re-derived from the question being asked now.
#
# `cohorts` belongs here, not with the entities: "failed versus healthy" IS the
# comparison being made. Omitting it produced a shape that could never validate,
# because `compare` requires two cohorts — so every compare exemplar was stored
# and then silently discarded on retrieval.
_SHAPE_FIELDS = ("op", "cohorts", "metrics", "dimensions", "group_by", "bin",
                 "time_grain", "effect_size", "limit", "confidence", "verify_premise")


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


@dataclass(slots=True)
class HashingEmbedder:
    """Hashed character 3-grams plus word unigrams, L2-normalised.

    Character n-grams give robustness to inflection and typos ("speeds" vs
    "speed"); word unigrams give topical signal. Hashing avoids a vocabulary
    that would need to be versioned and shipped alongside the store.
    """

    dim: int = DIM

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        norm = normalise(text) or text.lower().strip()

        for word in _WORD.findall(norm):
            v[hash(f"w:{word}") % self.dim] += 1.0
        padded = f"  {norm}  "
        for i in range(len(padded) - 2):
            v[hash(f"c:{padded[i:i + 3]}") % self.dim] += 0.5

        length = math.sqrt(float(v @ v))
        return v / length if length > 0 else v


@dataclass(slots=True)
class Exemplar:
    """One question whose plan was verified end to end."""

    question: str
    normalised: str
    shape: dict[str, Any]
    op: str
    created: str
    uses: int = 0
    source_tier: str = "grammar"

    def to_json(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "normalised": self.normalised,
            "shape": self.shape,
            "op": self.op,
            "created": self.created,
            "uses": self.uses,
            "source_tier": self.source_tier,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Exemplar:
        return cls(
            question=raw["question"],
            normalised=raw["normalised"],
            shape=raw["shape"],
            op=raw["op"],
            created=raw["created"],
            uses=int(raw.get("uses", 0)),
            source_tier=raw.get("source_tier", "grammar"),
        )


def rebind(
    shape: dict[str, Any], question: str, state: SessionState | None
) -> AnalysisPlan | None:
    """Rebuild a plan from a shape, taking entities from the CURRENT question.

    A shape describes the analysis; filters name the rows. Reusing a stored plan
    verbatim answers the old question with the new question's wording — which is
    exactly the failure that made "why did cycle 2750 fail" return cycle 4045.
    """
    from copilot.ir import OpName
    from copilot.planner.grammar import _extract_filters

    payload = dict(shape)
    try:
        filters = _extract_filters(question.lower(), state, OpName(payload["op"]))
        payload["filters"] = [f.model_dump(mode="json") for f in filters]
    except Exception:
        payload["filters"] = []
    try:
        return parse_plan(payload)
    except PlanError:
        return None


def plan_shape(plan: AnalysisPlan) -> dict[str, Any]:
    """The reusable part of a plan: what analysis, not which rows."""
    dumped = plan.model_dump(mode="json")
    shape = {k: dumped[k] for k in _SHAPE_FIELDS if k in dumped}
    # params carry op configuration (ordering, changes) which IS reusable, but
    # a raw SQL string is not — it is entity-specific by construction.
    params = dict(dumped.get("params") or {})
    params.pop("sql", None)
    if params:
        shape["params"] = params
    return shape


@dataclass(slots=True)
class ExemplarStore:
    """Verified plan shapes, retrieved by question similarity."""

    embedder: Embedder = field(default_factory=HashingEmbedder)
    path: Path | None = None
    capacity: int = 2000
    _exemplars: list[Exemplar] = field(default_factory=list)
    _matrix: np.ndarray | None = None
    lookups: int = 0
    hits: int = 0

    # -- lifecycle ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._exemplars)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def load(self) -> ExemplarStore:
        if self.path and self.path.exists():
            with self.path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self._exemplars.append(Exemplar.from_json(json.loads(line)))
            self._reindex()
        return self

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as fh:
            for ex in self._exemplars:
                fh.write(json.dumps(ex.to_json()) + "\n")

    def clear(self) -> None:
        self._exemplars.clear()
        self._matrix = None
        self.lookups = self.hits = 0

    # -- writing -----------------------------------------------------------

    def record(self, question: str, plan: AnalysisPlan, *, tier: str = "grammar") -> bool:
        """Store a plan shape that verified. Returns True if newly added.

        Only ever called for answers that passed the numeric verifier — an
        unverified plan is not evidence of anything.
        """
        norm = normalise(question)
        if not norm:
            return False
        shape = plan_shape(plan)

        for ex in self._exemplars:
            if ex.normalised == norm:
                ex.shape = shape          # a later verified plan supersedes
                ex.op = plan.op.value
                return False

        self._exemplars.append(
            Exemplar(
                question=question.strip(),
                normalised=norm,
                shape=shape,
                op=plan.op.value,
                created=datetime.now(UTC).isoformat(timespec="seconds"),
                source_tier=tier,
            )
        )
        # Evict the least-used exemplars first: an entry nothing retrieves is
        # not carrying its weight.
        if len(self._exemplars) > self.capacity:
            self._exemplars.sort(key=lambda e: (e.uses, e.created))
            del self._exemplars[: len(self._exemplars) - self.capacity]
        self._reindex()
        return True

    def _reindex(self) -> None:
        if not self._exemplars:
            self._matrix = None
            return
        self._matrix = np.vstack(
            [self.embedder.embed(e.question) for e in self._exemplars]
        )

    # -- reading -----------------------------------------------------------

    def nearest(self, question: str, k: int = 3) -> list[tuple[Exemplar, float]]:
        """Top-k by cosine similarity. Vectors are unit-length, so it is a dot."""
        if self._matrix is None or not self._exemplars:
            return []
        sims = self._matrix @ self.embedder.embed(question)
        order = np.argsort(sims)[::-1][:k]
        return [(self._exemplars[int(i)], float(sims[int(i)])) for i in order]

    def suggest(
        self, question: str, state: SessionState | None = None
    ) -> tuple[AnalysisPlan, float, str] | None:
        """Rebuild a plan from the nearest verified exemplar.

        The exemplar supplies the analysis; the current question supplies the
        entities. A plan that fails validation is discarded rather than repaired
        — the router simply escalates, which is the correct behaviour for a tier
        whose whole promise is that it is cheap.
        """
        self.lookups += 1
        matches = self.nearest(question, k=1)
        if not matches:
            return None

        exemplar, score = matches[0]
        if score < REUSE_THRESHOLD:
            return None

        payload = dict(exemplar.shape)

        # Only a near-identical question justifies reusing its metric selection.
        if score < EXACT_THRESHOLD:
            from copilot.planner.grammar import _extract_metrics

            fresh = _extract_metrics(question.lower())
            if fresh:
                payload["metrics"] = fresh

        plan = rebind(payload, question, state)
        if plan is None:
            return None

        exemplar.uses += 1
        self.hits += 1
        return plan, score, f"reused verified exemplar: {exemplar.question!r} ({score:.2f})"

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict[str, Any]:
        by_op: dict[str, int] = {}
        for ex in self._exemplars:
            by_op[ex.op] = by_op.get(ex.op, 0) + 1
        return {
            "exemplars": len(self._exemplars),
            "lookups": self.lookups,
            "hits": self.hits,
            "hit_rate": round(self.hit_rate, 3),
            "by_op": dict(sorted(by_op.items())),
            "most_used": sorted(
                ({"question": e.question, "uses": e.uses} for e in self._exemplars),
                key=lambda d: d["uses"],
                reverse=True,
            )[:5],
        }

    def export_training_pairs(self) -> list[dict[str, Any]]:
        """Supervised pairs for Phase 9b.

        The exemplar store is not only a fast tier — it is the corpus a planner
        LoRA would be distilled from, already filtered to plans that verified.
        """
        return [
            {"question": e.question, "plan": e.shape, "op": e.op, "uses": e.uses}
            for e in self._exemplars
        ]
