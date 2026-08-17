"""SLM planner — Ollama-backed local model with JSON grammar constraints.

This is the inference adapter for an SLM trained with ``scripts/export_sft_data.py``
and fine-tuned via ``notebooks/planner_distillation.ipynb``.

It uses Ollama's ``format`` parameter to enforce the plan schema at the
token level — the model cannot produce structurally invalid JSON.  This
is the same mechanism used by ``ConstrainedPlanner`` (for Ollama) and the
schema passed to Anthropic/Cerebras providers.

Configuration
-------------
    COPILOT_PROVIDER=slm
    COPILOT_SLM_MODEL=margin-planner          # or any Ollama-compatible name
    COPILOT_SLM_ENDPOINT=http://127.0.0.1:11434/api/chat

To pull a fine-tuned model after pushing to Hugging Face Hub:
    ollama pull hf.co/your-name/margin-planner-gguf:latest

If Ollama is not running or the model is not available, the planner raises
``ImportError`` and the router falls back to the LLM tier gracefully.
"""

from __future__ import annotations

import json
import os
from typing import Any

from copilot.planner.constrained import plan_schema, system_prompt


class SLMPlanner:
    """Ollama-hosted SLM with JSON-grammar-constrained plan output.

    Identical interface to ``ConstrainedPlanner`` so the router can swap
    them with a one-line config change.
    """

    name = "slm"

    def __init__(
        self,
        model: str | None = None,
        endpoint: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model    = model    or os.environ.get("COPILOT_SLM_MODEL",    "margin-planner")
        self.endpoint = endpoint or os.environ.get("COPILOT_SLM_ENDPOINT", "http://127.0.0.1:11434/api/chat")
        self.timeout  = timeout

    def propose(self, question: str) -> dict[str, Any]:
        """Return a raw plan dict.  Raises on transport failure, never on shape."""
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt()},
                {"role": "user",   "content": question},
            ],
            "stream": False,
            # JSON grammar constraint — identical to ConstrainedPlanner
            "format": plan_schema(),
            "options": {
                "temperature": 0,
                "top_k":       1,
                "top_p":       1.0,
                "num_predict": 256,
                "seed":        0,
            },
        }).encode()

        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
        except Exception as exc:
            raise RuntimeError(f"SLM endpoint unreachable ({self.endpoint}): {exc}") from exc

        raw = json.loads(payload["message"]["content"])
        return _normalise(raw)

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> str:
        """Provider protocol method — delegates to ``propose``."""
        raw = self.propose(user)
        return json.dumps(raw)

    def complete_text(self, system: str, user: str, max_tokens: int) -> str:
        """Narration path — SLM is a planner only; raise to let LLM narrate."""
        raise NotImplementedError("SLMPlanner handles planning only; narration falls back to templates")


def _normalise(raw: dict[str, Any]) -> dict[str, Any]:
    """A stated refusal reason outranks a non-refuse op (same logic as ConstrainedPlanner)."""
    if raw.get("refuse_reason") and raw.get("op") != "refuse":
        return {"op": "refuse", "refuse_reason": raw["refuse_reason"]}
    return raw


def is_available(model: str | None = None, endpoint: str | None = None) -> bool:
    """Return True if the Ollama endpoint is reachable and the model is loaded."""
    import urllib.request
    ep = endpoint or os.environ.get("COPILOT_SLM_ENDPOINT", "http://127.0.0.1:11434")
    try:
        with urllib.request.urlopen(f"{ep.rstrip('/')}/api/tags", timeout=2.0) as resp:
            tags = json.loads(resp.read())
        mdl = model or os.environ.get("COPILOT_SLM_MODEL", "margin-planner")
        return any(mdl in m.get("name", "") for m in tags.get("models", []))
    except Exception:
        return False
