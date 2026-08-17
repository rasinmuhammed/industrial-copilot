"""Tier 2 - a store of plans that were verified to work.

Every answered question emits a triple:

    (question, plan, did_it_verify?)

That label is **free and objective**. The plan either validated, executed and
passed the numeric verifier, or it did not - no human annotation, no preference
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
import zlib
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
    "polarity",
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
# a wrong NUMBER - validation and the verifier still run - but it would answer
# the wrong question, and an escalation only costs latency.
REUSE_THRESHOLD = 0.55
# Above this the questions differ only in entities, so the exemplar's metric
# selection is reused verbatim instead of being re-derived from the new wording.
EXACT_THRESHOLD = 0.90

DIM = 512
_WORD = re.compile(r"[a-z0-9_]+")

# Bag-of-ngram similarity is nearly blind to negation: "why did cycle 9016 fail"
# and "why did cycle 9016 NOT fail" score 0.885, comfortably above the reuse
# threshold. Polarity is therefore checked exactly and separately, because
# retrieving the affirmative plan for a negated question is a false answer, not
# a near miss.
_NEGATION = re.compile(
    r"\b(not|never|without|excluding|except|besides|didn't|doesn't|don't|"
    r"isn't|aren't|wasn't|weren't|no longer|failed to|non-)\b"
)


def polarity(text: str) -> bool:
    """True when the question carries a negation. Compared exactly, not scored."""
    return bool(_NEGATION.search(text.lower()))


# Fields that describe the ANALYSIS. Everything else is entity-specific and is
# re-derived from the question being asked now.
#
# `cohorts` belongs here, not with the entities: "failed versus healthy" IS the
# comparison being made. Omitting it produced a shape that could never validate,
# because `compare` requires two cohorts - so every compare exemplar was stored
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

    THE HASH MUST BE STABLE ACROSS PROCESSES. This used Python's builtin
    hash(), which is randomised per interpreter by PYTHONHASHSEED. Two
    consequences, both silent:

      * the same question embedded in two processes produced different vectors,
        so exemplar retrieval was nondeterministic. Measured: "impact of
        dropping torque 10 Nm" was answered under seeds 0-2 and refused under
        seed 3, which moved end-to-end coverage between 96.8% and 98.4% run to
        run. A benchmark figure that is not reproducible is not a figure.
      * a PERSISTED store would have been quietly corrupt. Vectors written by
        one process would not match vectors computed by the next, so every
        saved exemplar becomes unreachable after a restart - with no error,
        just a system that mysteriously stops learning.

    crc32 is deterministic, C-speed, and adequate for a hashing trick where
    collision quality matters far less than reproducibility.
    """

    dim: int = DIM

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        norm = normalise(text) or text.lower().strip()

        for word in _WORD.findall(norm):
            v[zlib.crc32(f"w:{word}".encode()) % self.dim] += 1.0
        padded = f"  {norm}  "
        for i in range(len(padded) - 2):
            v[zlib.crc32(f"c:{padded[i:i + 3]}".encode()) % self.dim] += 0.5

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
    verbatim answers the old question with the new question's wording - which is
    exactly the failure that made "why did cycle 2750 fail" return cycle 4045.
    """
    from copilot.ir import OpName
    from copilot.planner.grammar import (
        _categorical_premise,
        _extract_changes,
        _extract_filters,
        _extract_operating_point,
    )

    payload = dict(shape)

    # Entities come from the question being asked NOW - params exactly as much
    # as filters. `plan_shape` stores only the configuration half, so anything
    # naming a wear, a torque, a variant or a claim is re-derived here. A key
    # the current question does not mention is simply absent, and the op falls
    # back to its own default rather than inheriting a stranger's operating
    # point.
    lowered = question.lower()
    params = dict(payload.get("params") or {})
    params.update(_extract_operating_point(lowered))
    if (changes := _extract_changes(lowered)):
        params["changes"] = changes
    if payload.get("verify_premise") and (claim := _categorical_premise(lowered)):
        params["premise"] = claim
    if params:
        payload["params"] = params
    else:
        payload.pop("params", None)

    try:
        filters = _extract_filters(lowered, state, OpName(payload["op"]))
        # A premise claim needs the cross-group comparison, so a filter on the
        # claimed field would defeat the very test being run - you cannot ask
        # "is H the worst" while looking only at H. Scoped breakdowns like
        # "failure rate for L variants" legitimately group AND filter, so this
        # is narrowed to the premise case rather than banned outright. Without
        # it the grammar and cache tiers silently disagreed: the first ask
        # refuted the premise and an identical repeat did not.
        claimed = ((payload.get("params") or {}).get("premise") or {}).get("field")
        if claimed:
            filters = [f for f in filters if f.field != claimed]
        payload["filters"] = [f.model_dump(mode="json") for f in filters]
    except Exception:
        payload["filters"] = []
    try:
        return parse_plan(payload)
    except PlanError:
        return None


#: Params that name a specific operating point or quantity from the question,
#: rather than configuring the analysis. These are ENTITIES, and the cache key
#: erases entities: `normalise()` turns every number into `<n>`, so "at 8
#: minutes of wear" and "at 200 minutes of wear" are one key. Storing their
#: values in the shape therefore answers the second question with the first
#: question's operating point.
#:
#: That is what happened. Every phrasing of the envelope question returned the
#: torque window for whichever wear was asked about FIRST in the process - a
#: verified, exact, correctly-computed safe operating range for a machine in a
#: different condition than the one the engineer described. It is the
#: prescriptive capability, which is the part of this product an operator would
#: actually act on, and the plan cache made it answer from stale wear.
#:
#: `sql` was already excluded here for exactly this reason. The reasoning simply
#: had not been carried to the rest of the entity-bearing keys.
_ENTITY_PARAMS = frozenset({
    "sql",
    "udi", "product_type",
    "air_temp_k", "process_temp_k",
    "rotational_speed_rpm", "torque_nm", "tool_wear_min",
    "changes", "premise",
})


def plan_shape(plan: AnalysisPlan) -> dict[str, Any]:
    """The reusable part of a plan: what analysis, not which rows.

    Entity-bearing params are dropped, not stored. `rebind` re-derives them from
    the question being asked now, the same way it re-derives filters.
    """
    dumped = plan.model_dump(mode="json")
    shape = {k: dumped[k] for k in _SHAPE_FIELDS if k in dumped}
    params = {
        k: v for k, v in (dumped.get("params") or {}).items()
        if k not in _ENTITY_PARAMS
    }
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

        Only ever called for answers that passed the numeric verifier - an
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
        - the router simply escalates, which is the correct behaviour for a tier
        whose whole promise is that it is cheap.
        """
        self.lookups += 1
        matches = self.nearest(question, k=1)
        if not matches:
            return None

        exemplar, score = matches[0]
        if score < REUSE_THRESHOLD:
            return None
        # Polarity is a hard gate, not a similarity contribution: a negated
        # question asks about the complement of the stored cohort.
        if polarity(question) != polarity(exemplar.question):
            return None
        # Polarity is a hard gate, not a similarity contribution. A negated
        # question asks about the complement of the stored cohort.
        if polarity(question) != polarity(exemplar.question):
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

        The exemplar store is not only a fast tier - it is the corpus a planner
        LoRA would be distilled from, already filtered to plans that verified.
        """
        return [
            {"question": e.question, "plan": e.shape, "op": e.op, "uses": e.uses}
            for e in self._exemplars
        ]
