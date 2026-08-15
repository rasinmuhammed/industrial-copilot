"""Typed conversational state.

The naive follow-up implementation appends every turn to a growing transcript.
Cost and latency climb turn over turn, and the model starts resolving pronouns
against stale context. Instead the conversation is compressed into a typed
object with a fixed token budget: **turn 40 costs the same as turn 2.**

Two properties matter beyond cost:

  * **Follow-ups resolve structurally.** "What about the H variants?" mutates one
    filter on the previous plan rather than re-planning, which is both faster and
    more correct — the analysis stays identical, so the comparison is valid.
  * **State is visible and correctable.** A bad resolution in turn 3 must not
    propagate silently, so every answer restates its resolved scope.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from copilot.ir import AnalysisPlan, Filter

__all__ = ["Focus", "Turn", "SessionState"]

TURN_MEMORY = 6


class Focus(BaseModel):
    """The entity currently under discussion."""

    model_config = ConfigDict(frozen=True)

    kind: str  # "machine" | "cycle" | "product_type" | "cohort"
    value: str
    label: str = ""

    def as_filter(self) -> Filter | None:
        field = {
            "machine": "machine_id",
            "cycle": "udi",
            "product_type": "product_type",
        }.get(self.kind)
        if field is None:
            return None
        value: Any = int(self.value) if field == "udi" else self.value
        return Filter(field=field, op="=", value=value)

    def describe(self) -> str:
        return self.label or f"{self.kind} {self.value}"


class Turn(BaseModel):
    """One line of history. Deliberately a summary, never a transcript."""

    model_config = ConfigDict(frozen=True)

    question: str
    op: str
    summary: str
    plan_hash: str


class SessionState(BaseModel):
    """Bounded conversational memory."""

    focus: Focus | None = None
    filters: list[Filter] = Field(default_factory=list)
    last_plan: AnalysisPlan | None = None
    last_evidence: str | None = None          # plan hash, for "show me those rows"
    metrics_seen: list[str] = Field(default_factory=list)
    synthetic_used: set[str] = Field(default_factory=set)
    turns: deque[Turn] = Field(default_factory=lambda: deque(maxlen=TURN_MEMORY))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -- mutation ----------------------------------------------------------

    def record(self, question: str, plan: AnalysisPlan, summary: str) -> None:
        self.turns.append(
            Turn(question=question, op=plan.op.value, summary=summary, plan_hash=plan.hash)
        )
        self.last_plan = plan
        self.last_evidence = plan.hash
        for metric in plan.metrics:
            if metric not in self.metrics_seen:
                self.metrics_seen.append(metric)
        self.metrics_seen = self.metrics_seen[-8:]
        self.synthetic_used.update(plan.synthetic_used())
        self._infer_focus(plan)

    def _infer_focus(self, plan: AnalysisPlan) -> None:
        """Adopt an entity focus when the plan pins one down, and DROP it when it
        does not.

        A focus that lingers past the answer it belonged to makes the scope line
        lie — the previous turn's cycle number would be reported as the scope of
        a fleet-wide rate.
        """
        self.focus = None
        for f in plan.filters:
            if f.op.value != "=":
                continue
            if f.field == "machine_id":
                self.focus = Focus(kind="machine", value=str(f.value), label=f"machine {f.value}")
                return
            if f.field == "udi":
                self.focus = Focus(kind="cycle", value=str(f.value), label=f"cycle {f.value}")
                return
            if f.field == "product_type":
                self.focus = Focus(
                    kind="product_type", value=str(f.value), label=f"{f.value} variant"
                )
                return

    def clear(self) -> None:
        self.focus = None
        self.filters = []
        self.last_plan = None
        self.last_evidence = None
        self.metrics_seen = []
        self.synthetic_used = set()
        self.turns.clear()

    # -- inspection --------------------------------------------------------

    def scope_line(self) -> str:
        """One line restating what the answer is about.

        Prepended to every response so a wrong resolution is visible immediately
        rather than three turns later.
        """
        parts: list[str] = []
        if self.focus:
            parts.append(self.focus.describe())
        parts += [f.describe() for f in self.filters]
        if not parts:
            return "all cycles"
        return ", ".join(parts)

    def as_prompt_block(self) -> str:
        """Compact serialisation for the planner prompt.

        This is the ONLY conversational context a model ever sees. It is bounded
        by construction, which is what keeps the token budget flat.
        """
        lines = [f"focus: {self.focus.describe() if self.focus else 'none'}"]
        if self.filters:
            lines.append("active filters: " + "; ".join(f.describe() for f in self.filters))
        if self.metrics_seen:
            lines.append("metrics discussed: " + ", ".join(self.metrics_seen))
        if self.last_plan is not None:
            lines.append(f"previous op: {self.last_plan.op.value}")
        if self.turns:
            lines.append("recent turns:")
            lines += [f"  - {t.question} -> {t.summary}" for t in self.turns]
        return "\n".join(lines)

    def token_estimate(self) -> int:
        """Rough size of the state block. Asserted flat across turns in evals."""
        return len(self.as_prompt_block()) // 4
