"""Central configuration. Every tunable lives here; nothing else reads os.environ."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- data ---
    csv_path: Path = field(default_factory=lambda: Path(os.environ.get("COPILOT_CSV", ROOT / "data" / "ai4i2020.csv")))
    db_path: Path = field(default_factory=lambda: Path(os.environ.get("COPILOT_DB", ROOT / "data" / "warehouse.duckdb")))
    # Verified plan shapes, accumulated across sessions. This is the corpus a
    # distilled planner would be trained from (docs/08-DISCOVERY.md §8).
    exemplar_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("COPILOT_EXEMPLARS", ROOT / "data" / "exemplars.jsonl")
        )
    )

    # --- synthetic overlays (see README "Assumptions") ---
    takt_seconds: int = 120
    epoch: str = "2024-01-01T00:00:00"
    virtual_machines_per_type: int = 5

    # --- planner ---
    # Tier 2 only runs when a key is present. Without one the copilot still
    # answers everything the deterministic grammar covers, and says so.
    anthropic_api_key: str | None = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY"))
    planner_model: str = field(default_factory=lambda: os.environ.get("COPILOT_PLANNER_MODEL", "claude-haiku-4-5"))
    narrator_model: str = field(default_factory=lambda: os.environ.get("COPILOT_NARRATOR_MODEL", "claude-haiku-4-5"))
    # Escalation model for questions the fast planner marks low-confidence.
    escalation_model: str = field(default_factory=lambda: os.environ.get("COPILOT_ESCALATION_MODEL", "claude-opus-5"))

    planner_max_tokens: int = 1500
    narrator_max_tokens: int = 1200
    plan_repair_attempts: int = 1
    request_timeout_s: float = 30.0

    # --- behaviour ---
    llm_narration: bool = field(default_factory=lambda: _env_bool("COPILOT_LLM_NARRATION", True))
    plan_cache_size: int = 512
    max_rows_returned: int = 50
    session_turn_memory: int = 6

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def knowledge_dir(self) -> Path:
        return PKG_DIR / "knowledge"


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()
