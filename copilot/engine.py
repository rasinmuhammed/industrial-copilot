"""The orchestrator: question in, verified answer out.

    question
      -> router      (cache | grammar | model)
      -> plan        validated against the semantic layer
      -> executor    typed op over DuckDB
      -> evidence    slots with units, intervals and provenance
      -> narrator    prose with {{slot}} references, never digits
      -> verifier    fail-closed; a bare numeral is rejected
      -> answer      + evidence + a replay handle

Every stage is independently testable, and the two that involve a language model
are sandwiched between validation and verification, so neither can put a wrong
number in front of an engineer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from copilot.config import settings
from copilot.evidence import EvidenceBundle
from copilot.ingest import connect
from copilot.ir import AnalysisPlan
from copilot.narrate import format_answer, template_narrate
from copilot.ops import ExecutionContext, data_fingerprint, execute, kb_version
from copilot.planner.router import Router, RoutingError
from copilot.session import SessionState
from copilot.planner.unknown import detect_unknown_quantity
from copilot.verify import VerificationResult, verify

__all__ = ["Answer", "Engine"]


@dataclass(slots=True)
class Answer:
    """Everything a caller needs, including how to reproduce it."""

    text: str
    narration: str          # the verified prose only, without scope or provenance
    question: str
    plan: AnalysisPlan | None
    bundle: EvidenceBundle | None
    tier: str
    elapsed_ms: float
    plan_ms: float = 0.0
    exec_ms: float = 0.0
    narrate_ms: float = 0.0
    narrator: str = "template"
    verified: bool = True
    degraded: bool = False
    refused: bool = False
    slots_used: list[str] = field(default_factory=list)

    @property
    def replay_handle(self) -> str:
        return self.bundle.provenance.plan_hash if self.bundle else ""


@dataclass(slots=True)
class Engine:
    router: Router
    ctx: ExecutionContext
    show_evidence: bool = False

    @classmethod
    def build(cls, *, read_only: bool = True, show_evidence: bool = False) -> Engine:
        con = connect(read_only=read_only)
        return cls(
            router=Router.build(exemplar_path=settings().exemplar_path),
            ctx=ExecutionContext(
                con=con, kb_version=kb_version(), data_version=data_fingerprint(con)
            ),
            show_evidence=show_evidence,
        )

    @property
    def provider_name(self) -> str:
        return self.router.provider_name

    def ask(self, question: str, state: SessionState | None = None) -> Answer:
        started = time.perf_counter()
        state = state if state is not None else SessionState()

        # --- refuse before planning ------------------------------------------
        #
        # A question naming a quantity we do not measure must be declined here,
        # not answered with the nearest thing we do have. The grammar tier will
        # happily match an intent verb ("show me", "how much") and fall back to
        # a default metric, which produces a fully verified, entirely correct
        # answer about the wrong sensor — the worst outcome available to an
        # industrial copilot, because every downstream guarantee still holds.
        unknown = detect_unknown_quantity(question)
        if unknown is not None:
            return Answer(
                text=unknown.message,
                narration=unknown.message,
                question=question,
                plan=None,
                bundle=None,
                tier="refused",
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                refused=True,
            )

        # --- plan ----------------------------------------------------------
        try:
            routed = self.router.route(question, state)
        except RoutingError as exc:
            return Answer(
                text=str(exc),
                narration=str(exc),
                question=question,
                plan=None,
                bundle=None,
                tier="refused",
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                refused=True,
            )

        # --- execute ---------------------------------------------------------
        exec_started = time.perf_counter()
        self.ctx.tier = routed.tier
        bundle = execute(routed.plan, self.ctx)
        exec_ms = (time.perf_counter() - exec_started) * 1000.0

        # --- narrate + verify -------------------------------------------------
        narrate_started = time.perf_counter()
        text, narrator, verified, degraded, slots_used = self._narrate(
            question, routed.plan, bundle
        )
        narrate_ms = (time.perf_counter() - narrate_started) * 1000.0

        # Close the learning loop. A plan only becomes an exemplar if its
        # answer survived numeric verification — an unverified plan is not
        # evidence of anything, and storing it would teach the wrong lesson.
        if verified and not degraded:
            self.router.learn(question, routed.plan, routed.tier)

        state.record(question, routed.plan, bundle.summary)

        return Answer(
            text=format_answer(
                text, bundle, scope=state.scope_line(), show_evidence=self.show_evidence
            ),
            narration=text,
            question=question,
            plan=routed.plan,
            bundle=bundle,
            tier=routed.tier,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            plan_ms=routed.elapsed_ms,
            exec_ms=exec_ms,
            narrate_ms=narrate_ms,
            narrator=narrator,
            verified=verified,
            degraded=degraded,
            slots_used=slots_used,
        )

    # -- narration ----------------------------------------------------------

    def _narrate(
        self, question: str, plan: AnalysisPlan, bundle: EvidenceBundle
    ) -> tuple[str, str, bool, bool, list[str]]:
        """Try the model narrator, verify, retry once, then fall back.

        Falling back to the template is a *degradation in readability only*. The
        template is itself verified, so the numbers are identical either way.
        """
        cfg = settings()
        if self.router.llm is not None and cfg.llm_narration:
            result = self._try_llm_narration(question, bundle)
            if result is not None:
                return result[0], "llm", True, False, result[1]

        draft = template_narrate(plan, bundle)
        checked = verify(draft, bundle, question=question)
        if checked.ok:
            return checked.text, "template", True, False, checked.slots_used

        # A template that fails verification is a bug in this repository, not a
        # model failure. Surface it rather than papering over it.
        return (
            _verification_failure_notice(checked),
            "template",
            False,
            True,
            checked.slots_used,
        )

    def _try_llm_narration(
        self, question: str, bundle: EvidenceBundle
    ) -> tuple[str, list[str]] | None:
        assert self.router.llm is not None
        try:
            draft = self.router.llm.narrate(question, bundle)
        except Exception:
            return None  # provider trouble: fall through to the template

        checked = verify(draft, bundle, question=question)
        if checked.ok:
            return checked.text, checked.slots_used

        try:
            retry = self.router.llm.renarrate(question, bundle, checked.reason())
        except Exception:
            return None
        rechecked = verify(retry, bundle, question=question)
        return (rechecked.text, rechecked.slots_used) if rechecked.ok else None


def _verification_failure_notice(result: VerificationResult) -> str:
    return (
        "The analysis completed, but the generated wording failed numeric "
        "verification and was withheld. This is a defect in the copilot, not in "
        f"the data. Details: {result.reason()}"
    )
