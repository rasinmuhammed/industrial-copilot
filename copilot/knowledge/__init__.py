"""Loaders for the YAML knowledge base and semantic layer."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from copilot.config import settings


@lru_cache(maxsize=1)
def failure_modes() -> dict[str, Any]:
    with (settings().knowledge_dir / "failure_modes.yaml").open() as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def semantic_layer() -> dict[str, Any]:
    with (settings().knowledge_dir / "semantic_layer.yaml").open() as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def mode_index() -> dict[str, dict[str, Any]]:
    return {m["code"]: m for m in failure_modes()["modes"]}


@lru_cache(maxsize=1)
def metric_index() -> dict[str, dict[str, Any]]:
    return semantic_layer()["metrics"]


@lru_cache(maxsize=1)
def dimension_index() -> dict[str, dict[str, Any]]:
    return semantic_layer()["dimensions"]


@lru_cache(maxsize=1)
def synonym_map() -> dict[str, tuple[str, str]]:
    """Lowercased phrase -> (kind, canonical_name). Longest-match wins downstream."""
    out: dict[str, tuple[str, str]] = {}
    for name, spec in metric_index().items():
        out[name.lower()] = ("metric", name)
        out[spec["label"].lower()] = ("metric", name)
        for syn in spec.get("synonyms", []):
            out[syn.lower()] = ("metric", name)
    for name, spec in dimension_index().items():
        out[name.lower()] = ("dimension", name)
        out[spec["label"].lower()] = ("dimension", name)
        for syn in spec.get("synonyms", []):
            out[syn.lower()] = ("dimension", name)
    return out


def metric_unit(name: str) -> str:
    return metric_index().get(name, {}).get("unit", "")


def metric_label(name: str) -> str:
    return metric_index().get(name, {}).get("label", name)
