"""Tier 0: the plan cache.

The key property is that keys are **tenant-independent**. "Which machines are
closest to their overstrain limit?" normalises to the same key for every
factory; only the filter differs, and the filter comes from session state rather
than the plan. One shared cache therefore serves a whole fleet, which is what
keeps LLM cost driven by the diversity of *question shapes* rather than by the
number of sites.

Normalisation deliberately erases specifics — entity ids, numbers, punctuation —
so "why did cycle 9016 fail" and "why did cycle 4045 fail" share one entry.

Because the KEY erases entities, the stored VALUE must not contain them. The
cache therefore holds a plan *shape* and rebinds filters from the question being
asked now. Storing whole plans instead is a correctness bug, not an
optimisation: it answers the new question with the old question's rows.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass, field

from copilot.ir import AnalysisPlan
from copilot.knowledge import synonym_map

__all__ = ["PlanCache", "normalise"]

_PUNCT = re.compile(r"[^\w\s.-]")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_MACHINE = re.compile(r"\b[LMH]-\d{2}\b", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_FILLER = frozenset(
    {"the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "please",
     "can", "you", "me", "us", "our", "my", "i", "we", "of", "for", "to", "on"}
)


def normalise(question: str) -> str:
    """Reduce a question to its shape.

    Synonyms collapse to canonical metric names, entities and numbers become
    placeholders, filler words go. What remains is the analytical intent.
    """
    text = question.strip().lower()
    text = _MACHINE.sub("<machine>", text)
    text = _PUNCT.sub(" ", text)
    text = _NUMBER.sub("<n>", text)

    # Canonicalise vocabulary so wording differences do not fragment the cache.
    synonyms = synonym_map()
    for phrase in sorted(synonyms, key=len, reverse=True):
        # 3 not 4: "rpm" and "air" are real synonyms, and skipping them meant
        # "average rpm" and "average rotational speed" occupied different cache
        # entries despite being the same question.
        if len(phrase) < 3:
            continue
        _kind, name = synonyms[phrase]
        text = re.sub(rf"\b{re.escape(phrase)}\b", name, text)

    tokens = [t for t in _WHITESPACE.split(text) if t and t not in _FILLER]
    return " ".join(tokens)


@dataclass(slots=True)
class _Entry:
    shape: dict
    hits: int = 0


@dataclass(slots=True)
class PlanCache:
    """Bounded LRU over normalised question shapes.

    Stores plans, never answers. A plan is data-independent, so it cannot go
    stale when the warehouse is rebuilt — that is the answer cache's problem, and
    it is keyed on the data fingerprint instead.
    """

    capacity: int = 512
    _entries: OrderedDict[str, _Entry] = field(default_factory=OrderedDict)
    lookups: int = 0
    hits: int = 0

    def key(self, question: str, *, scope: str = "") -> str:
        shape = normalise(question)
        digest = hashlib.sha256(f"{shape}|{scope}".encode()).hexdigest()[:16]
        return digest

    def get(
        self, question: str, *, scope: str = "", state=None
    ) -> AnalysisPlan | None:
        """Rebind the cached shape onto the question being asked now."""
        from copilot.planner.exemplars import rebind

        self.lookups += 1
        k = self.key(question, scope=scope)
        entry = self._entries.get(k)
        if entry is None:
            return None
        plan = rebind(entry.shape, question, state)
        if plan is None:
            return None
        self._entries.move_to_end(k)
        entry.hits += 1
        self.hits += 1
        return plan

    def put(self, question: str, plan: AnalysisPlan, *, scope: str = "") -> None:
        from copilot.planner.exemplars import plan_shape

        k = self.key(question, scope=scope)
        self._entries[k] = _Entry(shape=plan_shape(plan))
        self._entries.move_to_end(k)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
        self.lookups = 0
        self.hits = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def __len__(self) -> int:
        return len(self._entries)
