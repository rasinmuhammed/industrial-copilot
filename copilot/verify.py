"""Proof-Carrying Numbers: the fail-closed numeric verifier.

Implements the PCN protocol (arXiv:2509.06902). Its central property is that
verification lives in the **renderer, not the model**:

    "only claim-checked numbers are marked as verified, and all others default
     to unverified"

Concretely: the narrator writes prose containing `{{slot.id}}` references and is
forbidden from writing digits at all. This verifier resolves each reference
against the evidence bundle, substitutes the value with the unit and precision
the *slot* declares, and rejects the draft if any bare numeral survives.

Two failure classes this closes that a naive check misses:

  * **Invention.** A number that appears nowhere in the evidence.
  * **Mis-attribution.** A real number quoted against the wrong cohort. Slot ids
    are fully qualified (`failed.torque_nm.mean`), so the cohort is part of the
    reference and cannot be detached from the value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from copilot.evidence import SLOT_REF, EvidenceBundle

__all__ = ["Rejection", "VerificationResult", "verify", "VerifierError"]

# Any run of digits, including decimals, percentages and negatives.
_NUMERAL = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")

# Ordinals and small structural words are allowed as words, never as digits.
_ALLOWED_WORDS = frozenset(
    {"first", "second", "third", "fourth", "fifth", "one", "two", "three", "none"}
)

# Tokens that may legitimately contain digits without being claims.
_SAFE_TOKENS = re.compile(
    r"\b(?:HDF|PWF|OSF|TWF|RNF|ISA-\d+\.\d+|[A-Z]-\d{2})\b"
)


class Rejection(StrEnum):
    UNSOURCED_NUMERAL = "unsourced_numeral"
    UNKNOWN_SLOT = "unknown_slot"
    ABSTAINED_SLOT_RENDERED = "abstained_slot_rendered"
    EMPTY_DRAFT = "empty_draft"


class VerifierError(RuntimeError):
    """Raised only when a caller demands a verified string and none is possible."""


@dataclass(slots=True)
class VerificationResult:
    ok: bool
    text: str = ""
    rejections: list[tuple[Rejection, str]] = field(default_factory=list)
    slots_used: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    def reason(self) -> str:
        """Feedback for the regeneration attempt. Specific, not generic."""
        if not self.rejections:
            return ""
        lines = []
        for kind, detail in self.rejections:
            if kind is Rejection.UNSOURCED_NUMERAL:
                lines.append(
                    f"You wrote the bare numeral '{detail}'. Never write digits. "
                    f"Refer to a quantity only as {{{{slot.id}}}}."
                )
            elif kind is Rejection.UNKNOWN_SLOT:
                lines.append(
                    f"Slot '{detail}' does not exist in the evidence. Use only the "
                    "slot ids you were given."
                )
            elif kind is Rejection.ABSTAINED_SLOT_RENDERED:
                lines.append(
                    f"Slot '{detail}' abstained; do not present it as a value. Say the "
                    "quantity could not be determined."
                )
            else:
                lines.append(str(detail))
        return "\n".join(lines)


def verify(
    draft: str,
    bundle: EvidenceBundle,
    *,
    question: str = "",
) -> VerificationResult:
    """Resolve slot references and reject any unsourced numeral.

    `question` is used only to whitelist numbers the user themselves supplied —
    echoing "you asked about 1380 rpm" is not a claim about the data.
    """
    if not draft or not draft.strip():
        return VerificationResult(
            ok=False, rejections=[(Rejection.EMPTY_DRAFT, "the draft was empty")]
        )

    rejections: list[tuple[Rejection, str]] = []
    used: list[str] = []
    unresolved: list[str] = []

    # --- stage 1: the draft itself must contain no bare digits -------------
    stripped = SLOT_REF.sub("", draft)
    for numeral in _find_numerals(stripped, question):
        rejections.append((Rejection.UNSOURCED_NUMERAL, numeral))

    # --- stage 2: resolve every reference ----------------------------------
    def _substitute(match: re.Match[str]) -> str:
        slot_id = match.group(1)
        slot = bundle.slots.get(slot_id)
        if slot is None:
            rejections.append((Rejection.UNKNOWN_SLOT, slot_id))
            unresolved.append(slot_id)
            return f"[unresolved:{slot_id}]"
        used.append(slot_id)
        return slot.render()

    rendered = SLOT_REF.sub(_substitute, draft)

    if rejections:
        return VerificationResult(
            ok=False, text=rendered, rejections=rejections, slots_used=used,
            unresolved=unresolved,
        )

    # --- stage 3: re-scan the rendered output ------------------------------
    # Every numeral now present must have come from a slot we substituted.
    permitted = {s.render() for s in bundle.slots.values() if s.value is not None}
    residual = rendered
    for value in sorted(permitted, key=len, reverse=True):
        residual = residual.replace(value, " ")
    for numeral in _find_numerals(residual, question):
        rejections.append((Rejection.UNSOURCED_NUMERAL, numeral))

    if rejections:
        return VerificationResult(
            ok=False, text=rendered, rejections=rejections, slots_used=used,
            unresolved=unresolved,
        )

    return VerificationResult(ok=True, text=rendered, slots_used=used)


def _find_numerals(text: str, question: str) -> list[str]:
    """Numerals that are not accounted for by the question or a safe token."""
    cleaned = _SAFE_TOKENS.sub(" ", text)
    from_question = {m.group(0).replace(",", "") for m in _NUMERAL.finditer(question)}
    found = []
    for match in _NUMERAL.finditer(cleaned):
        token = match.group(0)
        if token.replace(",", "").lstrip("+-") in from_question:
            continue
        if token.replace(",", "") in from_question:
            continue
        found.append(token)
    return found


def numerals_in(text: str) -> list[str]:
    """Public helper for evals: every numeral in a finished answer."""
    return [m.group(0) for m in _NUMERAL.finditer(_SAFE_TOKENS.sub(" ", text))]
