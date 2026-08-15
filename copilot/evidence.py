"""Evidence bundles: the only path by which a number reaches a human.

Every quantity the engineer ever sees is a `Slot`. The narrator may refer to
slots by id but may not write digits; the verifier then rejects any bare numeral
(see copilot/verify.py). This is the Proof-Carrying Numbers protocol
(arXiv:2509.06902) — verification lives in the renderer, not the model.

Two properties are load-bearing:

  * **Slot ids are fully qualified.** `failed.torque_nm.mean` carries its cohort,
    so quoting a real number against the wrong cohort is unrepresentable rather
    than merely unlikely.
  * **Slots can abstain.** A slot whose value cannot be trusted renders as an
    explicit "not determined", never as a plausible number. Bad input produces
    silence plus a warning, never a confident answer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from copilot.units import unit as resolve_unit

__all__ = [
    "Quality",
    "Severity",
    "Interval",
    "Slot",
    "Warning_",
    "Provenance",
    "EvidenceBundle",
    "SLOT_REF",
]

# Slot references in narrator output: {{cohort.metric.stat}}
SLOT_REF = re.compile(r"\{\{([a-zA-Z0-9_.\[\]-]+)\}\}")

# A slot id is dot-separated segments; the first is the cohort/scope.
_SLOT_ID = re.compile(r"^[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_\[\]-]+)*$")


class Quality(StrEnum):
    """Trust level of a computed quantity."""

    OK = "ok"
    LOW_SAMPLE = "low_sample"        # n below the reporting floor
    WIDE_INTERVAL = "wide_interval"  # CI too wide to state a point estimate
    SYNTHETIC = "synthetic"          # depends on a synthetic overlay
    ABSTAIN = "abstain"              # cannot be determined; must not be rendered


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Interval(BaseModel):
    """A closed interval. Used for confidence bounds and for uncertain inputs.

    The three-state decision (`ALERT` / `SAFE` / `ABSTAIN`) is a property of the
    interval's relationship to zero, which is why margins are represented this
    way end to end rather than only at the alerting layer.
    """

    model_config = ConfigDict(frozen=True)

    lo: float
    hi: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.lo > self.hi:
            raise ValueError(f"interval lo {self.lo} exceeds hi {self.hi}")
        return self

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def midpoint(self) -> float:
        return (self.lo + self.hi) / 2.0

    @property
    def is_point(self) -> bool:
        return math.isclose(self.lo, self.hi, rel_tol=1e-12, abs_tol=1e-12)

    def verdict(self) -> Literal["negative", "positive", "straddles"]:
        """Sign of the whole interval. `straddles` is the abstention trigger."""
        if self.hi < 0:
            return "negative"
        if self.lo > 0:
            return "positive"
        return "straddles"

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi

    def __str__(self) -> str:
        return f"[{self.lo:.6g}, {self.hi:.6g}]"


class Slot(BaseModel):
    """One verifiable quantity.

    `id` is fully qualified so that cohort attribution is structural. `render()`
    is the ONLY way a value becomes text — precision and unit come from the slot,
    never from a language model.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    value: float | int | str | None
    unit: str = ""
    n: int | None = None
    ci: Interval | None = None
    quality: Quality = Quality.OK
    sig_figs: int = 4
    note: str = ""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not _SLOT_ID.match(self.id):
            raise ValueError(f"malformed slot id {self.id!r}")
        resolve_unit(self.unit)  # raises UnitError on an unknown unit
        if self.quality is not Quality.ABSTAIN and self.value is None:
            raise ValueError(f"slot {self.id!r} has no value but does not abstain")
        return self

    @property
    def cohort(self) -> str:
        return self.id.split(".", 1)[0]

    def render(self) -> str:
        """Format for insertion into prose. Abstention renders as words."""
        if self.quality is Quality.ABSTAIN or self.value is None:
            return "not determined"
        if isinstance(self.value, str):
            return self.value
        body = _format_number(self.value, self.sig_figs)
        if self.unit and self.unit not in {"", "ratio", "count"}:
            body = f"{body} {self.unit}"
        if self.quality is Quality.LOW_SAMPLE and self.ci is not None:
            body = f"{body} (n={self.n}, 95% CI {_fmt_ci(self.ci, self.sig_figs)})"
        return body

    def render_full(self) -> str:
        """Verbose form for evidence tables and drill-down."""
        parts = [self.render()]
        if self.ci is not None and self.quality is not Quality.LOW_SAMPLE:
            parts.append(f"CI {_fmt_ci(self.ci, self.sig_figs)}")
        if self.n is not None:
            parts.append(f"n={self.n}")
        if self.quality is not Quality.OK:
            parts.append(self.quality.value)
        return "  ".join(parts)


def _format_number(value: float | int, sig_figs: int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    decimals = max(0, sig_figs - 1 - magnitude)
    return f"{value:,.{min(decimals, 6)}f}"


def _fmt_ci(ci: Interval, sig_figs: int) -> str:
    return f"[{_format_number(ci.lo, sig_figs)}, {_format_number(ci.hi, sig_figs)}]"


class Warning_(BaseModel):
    """A caveat the narrator is REQUIRED to surface.

    Named with a trailing underscore to avoid shadowing the builtin.
    """

    model_config = ConfigDict(frozen=True)

    code: Literal[
        "low_sample",
        "wide_interval",
        "collinearity",
        "synthetic_dimension",
        "premise_refuted",
        "data_quality",
        "abstained",
        "exploratory",
        "kb_drift",
        "sensor_suspect",
    ]
    severity: Severity = Severity.WARNING
    message: str
    affects: list[str] = Field(default_factory=list)  # slot ids or metric names


class Provenance(BaseModel):
    """Everything needed to reproduce an answer months later."""

    model_config = ConfigDict(frozen=True)

    plan_hash: str
    kb_version: str
    data_version: str
    op: str
    tier: Literal["cache", "grammar", "exemplar", "slm", "llm"] = "grammar"
    filters: list[str] = Field(default_factory=list)
    row_count: int = 0
    elapsed_ms: float = 0.0
    sql: str | None = None
    synthetic_dimensions: list[str] = Field(default_factory=list)


class EvidenceBundle(BaseModel):
    """The complete, self-describing result of one analysis.

    Serialises to JSON for the API, for replay, and for eval assertions.
    """

    slots: dict[str, Slot] = Field(default_factory=dict)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[Warning_] = Field(default_factory=list)
    provenance: Provenance
    summary: str = ""  # one-line machine-generated gist, for turn_digest

    # -- construction ------------------------------------------------------

    def add(self, slot: Slot) -> Slot:
        if slot.id in self.slots:
            raise ValueError(f"duplicate slot id {slot.id!r}")
        self.slots[slot.id] = slot
        return slot

    def put(
        self,
        slot_id: str,
        value: float | int | str | None,
        *,
        unit: str = "",
        n: int | None = None,
        ci: Interval | None = None,
        quality: Quality = Quality.OK,
        sig_figs: int = 4,
        note: str = "",
    ) -> Slot:
        return self.add(
            Slot(
                id=slot_id,
                value=value,
                unit=unit,
                n=n,
                ci=ci,
                quality=quality,
                sig_figs=sig_figs,
                note=note,
            )
        )

    def warn(
        self,
        code: str,
        message: str,
        *,
        severity: Severity = Severity.WARNING,
        affects: list[str] | None = None,
    ) -> None:
        self.warnings.append(
            Warning_(code=code, severity=severity, message=message, affects=affects or [])
        )

    # -- inspection --------------------------------------------------------

    @property
    def abstained(self) -> list[str]:
        return [s.id for s in self.slots.values() if s.quality is Quality.ABSTAIN]

    def cohorts(self) -> set[str]:
        return {s.cohort for s in self.slots.values()}

    def numeric_values(self) -> set[str]:
        """Rendered forms of every slot — the verifier's whitelist."""
        return {s.render() for s in self.slots.values() if s.value is not None}

    def evidence_table(self) -> str:
        """Human-readable dump, shown under every answer."""
        if not self.slots:
            return "(no evidence)"

        width = max(len(k) for k in self.slots)
        lines = [f"{k:<{width}}  {s.render_full()}" for k, s in self.slots.items()]
        return "\n".join(lines)


def plan_hash(payload: dict[str, Any]) -> str:
    """Stable hash of a canonicalised plan. The replay handle."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]
