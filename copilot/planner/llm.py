"""Tier 3/4: model-backed planning and narration.

One interface, several providers, all optional. With no provider configured the
system still answers every question the grammar tier covers and is fully
evaluable — graceful degradation here means degrading to *less fluent*, never to
*wrong*.

Prompt assembly is ordered so everything cacheable sits above everything
volatile: system prompt, then semantic layer, then operator catalogue, then the
session state and the question. Nothing dynamic appears above a cache breakpoint,
so the ~3,000-token prefix costs a tenth of base input on every call after the
first.

The model is asked for a **plan**, never for a number, and for narration it is
asked for prose containing slot references, never digits. Both outputs are
verified downstream, so a weak model produces worse writing, not worse facts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

from copilot.config import settings
from copilot.evidence import EvidenceBundle
from copilot.ir import AnalysisPlan, OpName, PlanError, parse_plan
from copilot.knowledge import dimension_index, metric_index, semantic_layer
from copilot.session import SessionState

__all__ = ["Provider", "LLMPlanner", "available_provider", "build_plan_schema"]


class Provider(Protocol):
    name: str

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> str: ...
    def complete_text(self, system: str, user: str, max_tokens: int) -> str: ...


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def build_plan_schema() -> dict[str, Any]:
    """JSON schema for constrained decoding.

    Providers that support grammar-constrained generation (XGrammar in vLLM,
    SGLang, TensorRT-LLM; native structured output on Cerebras) use this to make
    structurally invalid output impossible rather than merely unlikely.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["op"],
        "properties": {
            "op": {"type": "string", "enum": [o.value for o in OpName]},
            "metrics": {"type": "array", "items": {"enum": sorted(metric_index())}},
            "dimensions": {"type": "array", "items": {"enum": sorted(dimension_index())}},
            "group_by": {"type": "array", "items": {"enum": sorted(dimension_index())}},
            "filters": {"type": "array", "items": _filter_schema()},
            "cohorts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string"},
                        "filters": {"type": "array", "items": _filter_schema()},
                    },
                },
            },
            "bin": {
                "type": "object",
                "required": ["field"],
                "properties": {
                    "field": {"enum": sorted(set(metric_index()) | set(dimension_index()))},
                    "method": {"enum": ["quantile", "width", "explicit"]},
                    "bins": {"type": "integer", "minimum": 2, "maximum": 50},
                },
            },
            "time_grain": {"enum": ["hour", "shift", "day"]},
            "effect_size": {"enum": ["cohens_d", "rate_ratio", "risk_diff"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 500},
            "params": {"type": "object"},
            "verify_premise": {"type": "boolean"},
        },
    }


def _filter_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["field", "op"],
        "properties": {
            "field": {"enum": sorted(set(metric_index()) | set(dimension_index()))},
            "op": {"enum": ["=", "!=", "<", "<=", ">", ">=", "in", "between", "is_null"]},
            "value": {},
            "unit": {"type": "string"},
        },
    }


def planner_system_prompt() -> str:
    """Byte-stable prefix. Everything here caches."""
    layer = semantic_layer()
    metrics = "\n".join(
        f"  {name} [{spec.get('unit', '')}] — {spec['label']}"
        for name, spec in layer["metrics"].items()
    )
    dimensions = "\n".join(
        f"  {name} — {spec['label']}"
        + (f" (values: {', '.join(map(str, spec['values']))})" if spec.get("values") else "")
        for name, spec in layer["dimensions"].items()
    )
    intents = "\n".join(f"  {name} — {desc}" for name, desc in layer["intents"].items())

    return f"""You translate an engineer's question into an Analysis Plan for an \
Argus platform. You do NOT answer the question and you do NOT compute anything.

Emit a single JSON object and nothing else.

OPERATIONS
{intents}

METRICS (the only metric names that exist)
{metrics}

DIMENSIONS (the only dimension names that exist)
{dimensions}

RULES
- Use only the names listed above. A name that is not listed does not exist; if
  the question needs one, choose the closest listed name or pick an op that does
  not require it.
- Never invent a column, a table, or an aggregate function.
- `compare` requires exactly two cohorts, each with a name and filters.
- `trend` requires an axis: either `bin` on a metric, or `time_grain`.
- `counterfactual` requires params.changes mapping BASE variables to deltas.
  Base variables are air_temp_k, process_temp_k, rotational_speed_rpm,
  torque_nm, tool_wear_min. Derived quantities (power_w, overstrain_min_nm,
  temp_delta_k) cannot be changed directly.
- If a question asserts a comparative claim ("why are there MORE failures
  when..."), keep verify_premise true so the claim is tested before it is
  answered.
- Prefer a narrower op over sql_explore. Use sql_explore only when no structured
  op fits.
"""


def narrator_system_prompt() -> str:
    return """You write a short, precise answer for a maintenance engineer from a \
bundle of computed evidence.

ABSOLUTE RULE: you may not write any digit. Refer to every quantity ONLY by its
slot id in double braces, exactly as given, for example {{failed.torque_nm.mean}}.
The renderer substitutes the value, its unit and its precision. If you write a
number yourself the answer is rejected.

- Do not round, convert units, or compute. If a quantity you want is not in the
  bundle, say it is not available.
- Lead with any critical warning. A refuted premise is the answer, not a footnote.
- State findings plainly. No preamble, no restating the question, no filler.
- Two to five sentences unless the evidence genuinely needs more.
- Do not claim causation from an associational op.
"""


def planner_user_prompt(question: str, state: SessionState | None) -> str:
    parts = []
    if state is not None and (block := state.as_prompt_block()):
        parts.append(f"CONVERSATION STATE\n{block}")
    parts.append(f"QUESTION\n{question}")
    return "\n\n".join(parts)


def narrator_user_prompt(question: str, bundle: EvidenceBundle) -> str:
    slots = "\n".join(
        f"  {{{{{sid}}}}} = {slot.render()}"
        + (f"  [{slot.quality.value}]" if slot.quality.value != "ok" else "")
        for sid, slot in bundle.slots.items()
    )
    warnings = "\n".join(f"  [{w.severity.value}] {w.message}" for w in bundle.warnings)
    return (
        f"QUESTION\n{question}\n\n"
        f"AVAILABLE SLOTS (use these references, never the values)\n{slots}\n\n"
        f"WARNINGS YOU MUST SURFACE\n{warnings or '  (none)'}"
    )


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------


@dataclass(slots=True)
class AnthropicProvider:
    api_key: str
    model: str
    name: str = "anthropic"

    def _client(self):
        import anthropic

        return anthropic.Anthropic(api_key=self.api_key)

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        client = self._client()
        response = client.messages.create(
            model=self.model,
            max_tokens=settings().planner_max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=[
                {
                    "name": "emit_plan",
                    "description": "Emit the Analysis Plan for this question.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": "emit_plan"},
            messages=[{"role": "user", "content": user}],
        )
        for block in response.content:
            if block.type == "tool_use":
                return json.dumps(block.input)
        raise RuntimeError("model did not emit a plan")

    def complete_text(self, system: str, user: str, max_tokens: int) -> str:
        client = self._client()
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in response.content if b.type == "text")


@dataclass(slots=True)
class OpenAICompatibleProvider:
    """Cerebras, Together, Groq, vLLM, Ollama — all speak this dialect.

    Cerebras is the notable free option: 30 RPM, 60k TPM, 1M tokens/day, no card,
    and native JSON-schema structured output, which is exactly what a planner
    needs.
    """

    base_url: str
    api_key: str
    model: str
    name: str = "openai-compatible"

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with httpx.Client(timeout=settings().request_timeout_s) as client:
            response = client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        data = self._post(
            {
                "model": self.model,
                "max_tokens": settings().planner_max_tokens,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "analysis_plan", "strict": True, "schema": schema},
                },
            }
        )
        return data["choices"][0]["message"]["content"]

    def complete_text(self, system: str, user: str, max_tokens: int) -> str:
        data = self._post(
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        )
        return data["choices"][0]["message"]["content"]


def available_provider() -> Provider | None:
    """Resolve a provider from configuration, or None for the offline path."""
    cfg = settings()
    choice = os.environ.get("COPILOT_PROVIDER", "deterministic").lower()

    if choice in {"deterministic", "none", "off"}:
        return None
    if choice == "anthropic" and cfg.anthropic_api_key:
        return AnthropicProvider(api_key=cfg.anthropic_api_key, model=cfg.planner_model)
    if choice == "cerebras" and (key := os.environ.get("CEREBRAS_API_KEY")):
        return OpenAICompatibleProvider(
            base_url=os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"),
            api_key=key,
            model=os.environ.get("COPILOT_SLM_MODEL", "llama-3.3-70b"),
            name="cerebras",
        )
    if choice == "ollama":
        return OpenAICompatibleProvider(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="",
            model=os.environ.get("COPILOT_SLM_MODEL", "qwen2.5:7b-instruct"),
            name="ollama",
        )
    if choice == "slm":
        # Fine-tuned SLM via Ollama with JSON grammar constraints.
        # Activate with:  COPILOT_PROVIDER=slm COPILOT_SLM_MODEL=margin-planner
        from copilot.planner.slm import SLMPlanner
        return SLMPlanner()  # type: ignore[return-value]
    return None



# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------


@dataclass(slots=True)
class LLMPlanner:
    provider: Provider

    def plan(self, question: str, state: SessionState | None = None) -> AnalysisPlan:
        """One repair attempt on validation failure, then give up honestly."""
        system = planner_system_prompt()
        user = planner_user_prompt(question, state)
        schema = build_plan_schema()

        raw = self.provider.complete_json(system, user, schema)
        try:
            return parse_plan(raw)
        except PlanError as first:
            repair = (
                f"{user}\n\nYour previous plan was rejected.\n{first.repair_prompt()}\n"
                "Emit a corrected plan."
            )
            raw = self.provider.complete_json(system, repair, schema)
            return parse_plan(raw)  # a second failure propagates

    def narrate(self, question: str, bundle: EvidenceBundle) -> str:
        return self.provider.complete_text(
            narrator_system_prompt(),
            narrator_user_prompt(question, bundle),
            settings().narrator_max_tokens,
        )

    def renarrate(self, question: str, bundle: EvidenceBundle, reason: str) -> str:
        user = (
            f"{narrator_user_prompt(question, bundle)}\n\n"
            f"YOUR PREVIOUS DRAFT WAS REJECTED\n{reason}\nWrite it again, correctly."
        )
        return self.provider.complete_text(
            narrator_system_prompt(), user, settings().narrator_max_tokens
        )
