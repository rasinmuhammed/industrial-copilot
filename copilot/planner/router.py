"""Tiered planning: escalate only on a miss.

Most plant questions come from a small vocabulary, so most of them should never
reach a model. The router tries the cheapest path first and only escalates when
the cheaper one declines — and declining is explicit, via a confidence score,
rather than a silent guess.

    tier 0  plan cache        ~0 ms   repeat question shapes, shared across tenants
    tier 1  grammar           ~1 ms   the common plant vocabulary
    tier 2  exemplars         ~1 ms   questions whose plans previously VERIFIED
    tier 3  model             ~400 ms genuinely novel questions

Tier 2 is where the system gets better with use. Every verified answer deposits
its plan shape, so a question the model had to solve once is answered from the
cheap tier forever after — and the store doubles as the training corpus for a
distilled planner.

With no provider configured the router still answers everything tiers 0 and 1
cover, and refuses the rest honestly instead of fabricating a plan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from copilot.ir import AnalysisPlan, PlanError
from copilot.planner.cache import PlanCache
from copilot.planner.exemplars import ExemplarStore
from copilot.planner.grammar import plan_from_text
from copilot.planner.llm import LLMPlanner, available_provider
from copilot.session import SessionState

__all__ = ["RoutedPlan", "Router", "RoutingError"]


class RoutingError(RuntimeError):
    """No tier could produce a valid plan. Carries an engineer-readable reason."""


@dataclass(slots=True)
class RoutedPlan:
    plan: AnalysisPlan
    tier: str
    confidence: float
    elapsed_ms: float
    reason: str = ""


@dataclass(slots=True)
class Router:
    cache: PlanCache = field(default_factory=PlanCache)
    exemplars: ExemplarStore = field(default_factory=ExemplarStore)
    llm: LLMPlanner | None = None
    counts: dict[str, int] = field(
        default_factory=lambda: {"cache": 0, "grammar": 0, "exemplar": 0, "llm": 0}
    )

    @classmethod
    def build(cls, *, exemplar_path=None) -> Router:
        provider = available_provider()
        store = ExemplarStore(path=exemplar_path).load()
        return cls(
            exemplars=store,
            llm=LLMPlanner(provider) if provider is not None else None,
        )

    def learn(self, question: str, plan: AnalysisPlan, tier: str) -> bool:
        """Deposit a plan shape that verified. Called only on success.

        Cache hits and exemplar reuses teach nothing new, so they are skipped —
        the store should grow with novelty, not with traffic.
        """
        if tier in {"cache", "exemplar"}:
            return False
        return self.exemplars.record(question, plan, tier=tier)

    @property
    def provider_name(self) -> str:
        return self.llm.provider.name if self.llm is not None else "deterministic"

    def route(self, question: str, state: SessionState | None = None) -> RoutedPlan:
        started = time.perf_counter()
        scope = state.scope_line() if state is not None else ""

        def elapsed() -> float:
            return (time.perf_counter() - started) * 1000.0

        # --- tier 0: cache -------------------------------------------------
        # Skipped for follow-ups, whose meaning depends on state the key erases.
        cacheable = state is None or state.last_plan is None
        if cacheable and (cached := self.cache.get(question, scope=scope, state=state)) is not None:
            self.counts["cache"] += 1
            return RoutedPlan(cached, "cache", 1.0, elapsed(), "cache hit")

        # --- tier 1: grammar -----------------------------------------------
        match = plan_from_text(question, state)
        if match.usable and match.plan is not None:
            self.counts["grammar"] += 1
            if cacheable:
                self.cache.put(question, match.plan, scope=scope)
            return RoutedPlan(match.plan, "grammar", match.confidence, elapsed(), match.reason)

        # --- tier 2: verified exemplars ------------------------------------
        suggestion = self.exemplars.suggest(question, state)
        if suggestion is not None:
            plan, score, reason = suggestion
            self.counts["exemplar"] += 1
            return RoutedPlan(plan, "exemplar", score, elapsed(), reason)

        # --- tier 3: model -------------------------------------------------
        if self.llm is not None:
            try:
                plan = self.llm.plan(question, state)
            except PlanError as exc:
                raise RoutingError(
                    "I could not turn that into a valid analysis. "
                    f"{exc}. Try naming the metric or machine explicitly."
                ) from exc
            except Exception as exc:  # provider/transport failure
                raise RoutingError(
                    f"The planning model was unreachable ({type(exc).__name__}). "
                    "The deterministic planner did not recognise the question either."
                ) from exc
            self.counts["llm"] += 1
            if cacheable:
                self.cache.put(question, plan, scope=scope)
            return RoutedPlan(plan, "llm", 0.8, elapsed(), "model-planned")

        # --- nothing left: refuse, and say what would help -----------------
        raise RoutingError(_refusal(match.reason))

    def tier_distribution(self) -> dict[str, float]:
        total = sum(self.counts.values())
        if not total:
            return {k: 0.0 for k in self.counts}
        return {k: v / total for k, v in self.counts.items()}


def _refusal(reason: str) -> str:
    return (
        "I could not confidently interpret that question, and no planning model is "
        f"configured ({reason}). I can answer questions about operating conditions, "
        "failure rates, root causes, trends, drivers, safe operating windows, "
        "time-to-crossing forecasts, and data quality. Naming a metric — torque, "
        "rotational speed, tool wear, temperature differential, power — or a cycle "
        "number usually resolves it. Set COPILOT_PROVIDER to enable model-backed "
        "planning for open-ended questions."
    )
