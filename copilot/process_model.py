"""Load a process definition from the knowledge base.

WHY THIS EXISTS
---------------
`failure_modes.yaml` declared the failure boundaries. `physics.py` also declared
them, as module-level `Final` literals. Two sources of truth for the same
number, no mechanism to notice them diverging, sitting in the most load-bearing
file in the system - the precise failure this project exists to prevent.

It also made the scale story a claim rather than a capability. The brief asks
how the architecture evolves to 1,000 factories; the honest answer with compiled
constants is that factory #2 requires a code change and a redeploy. A scale
story you cannot execute is not a scale story.

This module makes the YAML authoritative. `physics.py` derives its constants
from a `ProcessModel`, so a different process is a different *file*, not a
different build.

WHAT A PROCESS DEFINITION IS
----------------------------
A set of failure modes, each carrying a predicate over named metrics. Predicates
take three shapes, which is all the AI4I documentation needs and all most
process rules need:

    all_of      every condition must hold      (HDF: hot AND slow)
    any_of      any condition suffices         (PWF: stalled OR overloaded)
    bare        a single condition             (OSF: strain over a limit)

A condition is `{metric, op, value}`. `value` may be a scalar, a two-element
range for `between`, or a `value_by_type` mapping when the limit depends on a
product variant.

Nothing here is AI4I-specific. The AI4I-shaped names in `physics.py` are a
*view* over this structure, kept so the existing call sites and their tests
continue to mean what they meant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import yaml

__all__ = [
    "Condition",
    "FailureMode",
    "ProcessModel",
    "load_process_model",
    "KB_PATH",
]

KB_PATH: Final = Path(__file__).resolve().parent / "knowledge" / "failure_modes.yaml"


@dataclass(frozen=True, slots=True)
class Condition:
    """One comparison against a boundary."""

    metric: str
    op: str
    value: float | tuple[float, float] | None = None
    value_by_type: dict[str, float] | None = None
    unit: str = ""

    def limit_for(self, product_type: str | None = None) -> float | tuple[float, float]:
        """The boundary this condition tests, resolved for a product variant."""
        if self.value_by_type is not None:
            if product_type is None:
                raise ValueError(
                    f"{self.metric} has a per-variant limit; a product type is required"
                )
            try:
                return self.value_by_type[product_type]
            except KeyError:
                raise ValueError(
                    f"no {self.metric} limit declared for product type {product_type!r}"
                ) from None
        if self.value is None:
            raise ValueError(f"{self.metric} condition declares no value")
        return self.value


@dataclass(frozen=True, slots=True)
class FailureMode:
    """A documented way the process fails."""

    code: str
    name: str
    kind: str                      # deterministic | stochastic
    conditions: tuple[Condition, ...]
    combinator: str                # all_of | any_of | single | none
    plain_english: str = ""

    @property
    def deterministic(self) -> bool:
        return self.kind == "deterministic"

    def condition_for(self, metric: str, op: str | None = None) -> Condition:
        for c in self.conditions:
            if c.metric == metric and (op is None or c.op == op):
                return c
        raise KeyError(f"{self.code} declares no condition on {metric!r} ({op})")


@dataclass(frozen=True, slots=True)
class ProcessModel:
    """Everything the physics layer needs, loaded rather than compiled."""

    modes: tuple[FailureMode, ...]
    wear_rate_per_cycle: dict[str, float] = field(default_factory=dict)
    base_variables: frozenset[str] = frozenset()
    version: int = 1

    def mode(self, code: str) -> FailureMode:
        for m in self.modes:
            if m.code == code:
                return m
        raise KeyError(f"no failure mode {code!r} in this process definition")

    def limit(self, code: str, metric: str, op: str | None = None) -> Any:
        """The single number this mode tests `metric` against."""
        return self.mode(code).condition_for(metric, op).limit_for()

    def limits_by_type(self, code: str, metric: str) -> dict[str, float]:
        by_type = self.mode(code).condition_for(metric).value_by_type
        if by_type is None:
            raise KeyError(f"{code}.{metric} is not a per-variant limit")
        return dict(by_type)

    @property
    def deterministic_modes(self) -> tuple[FailureMode, ...]:
        return tuple(m for m in self.modes if m.deterministic)

    # ── Generic evaluation ────────────────────────────────────────────────────
    #
    # `physics.evaluate` is a fast, hand-written path specialised to the AI4I
    # modes. This is the general one, driven entirely by the config, and it is
    # deliberately NOT the same code.
    #
    # That could have re-created the duplication just removed - two engines that
    # can disagree. It does not, because the disagreement is made into a test:
    # tests/test_process_model.py runs both over all 10,000 rows and requires
    # them to agree exactly. The same discipline as evals/reference.py, where an
    # independent implementation exists precisely so that a bug has to occur
    # twice, identically, to escape.
    #
    # The general path is what a new process gets until someone decides its
    # volume justifies a specialised one.

    def fires(self, reading: dict[str, float], product_type: str | None = None) -> list[str]:
        """Which deterministic modes this operating point triggers."""
        return [
            m.code
            for m in self.deterministic_modes
            if self._mode_fires(m, reading, product_type)
        ]

    def margins(
        self, reading: dict[str, float], product_type: str | None = None
    ) -> dict[str, float]:
        """Signed distance to each boundary, in native units.

        Positive is headroom. Negative means the condition is already met. This
        is the quantity the whole system is built on, and it falls out of the
        config with no per-process code.
        """
        out: dict[str, float] = {}
        for mode in self.deterministic_modes:
            for cond in mode.conditions:
                value = reading.get(cond.metric)
                if value is None:
                    continue
                out[f"{mode.code}.{cond.metric}"] = _margin(cond, value, product_type)
        return out

    def _mode_fires(
        self, mode: FailureMode, reading: dict[str, float], product_type: str | None
    ) -> bool:
        results = []
        for cond in mode.conditions:
            value = reading.get(cond.metric)
            if value is None:
                return False
            results.append(_holds(cond, value, product_type))
        if not results:
            return False
        return all(results) if mode.combinator == "all_of" else any(results)


def _holds(cond: Condition, value: float, product_type: str | None) -> bool:
    limit = cond.limit_for(product_type)
    match cond.op:
        case "<":
            return value < limit
        case "<=":
            return value <= limit
        case ">":
            return value > limit
        case ">=":
            return value >= limit
        case "==" | "=":
            return value == limit
        case "between":
            lo, hi = limit  # type: ignore[misc]
            return lo <= value <= hi
        case _:
            raise ValueError(f"unsupported operator {cond.op!r} on {cond.metric}")


def _margin(cond: Condition, value: float, product_type: str | None) -> float:
    """Signed distance to the boundary. Positive is headroom.

    The sign convention is what makes margins compose: a mode fires exactly when
    its margin is negative, whichever direction the comparison ran.
    """
    limit = cond.limit_for(product_type)
    match cond.op:
        case "<" | "<=":
            return value - limit          # firing when value is BELOW the limit
        case ">" | ">=":
            return limit - value          # firing when value is ABOVE the limit
        case "between":
            lo, hi = limit  # type: ignore[misc]
            # Outside the window is safe, so headroom is the distance to whichever
            # edge is nearer; inside, the margin is negative by the same measure.
            return max(lo - value, value - hi)
        case _:
            raise ValueError(f"no margin defined for operator {cond.op!r}")


def _condition(raw: dict[str, Any]) -> Condition:
    value = raw.get("value")
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError(f"range value must have two bounds, got {value!r}")
        value = (float(value[0]), float(value[1]))
    elif value is not None:
        value = float(value)
    by_type = raw.get("value_by_type")
    return Condition(
        metric=raw["metric"],
        op=raw["op"],
        value=value,
        value_by_type={k: float(v) for k, v in by_type.items()} if by_type else None,
        unit=raw.get("unit", ""),
    )


def _mode(raw: dict[str, Any]) -> FailureMode:
    predicate = raw.get("predicate")
    if not predicate:
        combinator, conditions = "none", ()
    elif "all_of" in predicate:
        combinator = "all_of"
        conditions = tuple(_condition(c) for c in predicate["all_of"])
    elif "any_of" in predicate:
        combinator = "any_of"
        conditions = tuple(_condition(c) for c in predicate["any_of"])
    else:
        combinator = "single"
        conditions = (_condition(predicate),)
    return FailureMode(
        code=raw["code"],
        name=raw.get("name", raw["code"]),
        kind=raw.get("kind", "deterministic"),
        conditions=conditions,
        combinator=combinator,
        plain_english=(raw.get("plain_english") or "").strip(),
    )


@lru_cache(maxsize=8)
def load_process_model(path: Path | str | None = None) -> ProcessModel:
    """Read a process definition. Cached, because it is read on every import.

    Passing an explicit path is how a second process is onboarded, and how the
    tests prove that onboarding needs no code change.
    """
    kb_path = Path(path) if path is not None else KB_PATH
    raw = yaml.safe_load(kb_path.read_text())
    process = raw.get("process") or {}
    return ProcessModel(
        modes=tuple(_mode(m) for m in raw.get("modes", [])),
        wear_rate_per_cycle={
            k: float(v) for k, v in (process.get("wear_rate_per_cycle") or {}).items()
        },
        base_variables=frozenset(process.get("base_variables") or ()),
        version=int(raw.get("version", 1)),
    )
