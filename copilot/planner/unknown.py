"""Refuse questions about quantities we do not measure.

WHY THIS EXISTS
---------------
Found by the risk-coverage harness, which is the only reason it was found:

    "show me the bearing temperature"     -> 20 arbitrary rows
    "how much does a replacement tool cost" -> air temperature statistics

Neither question is answerable. There is no bearing thermocouple in this
dataset and no cost data at all. But both contain a recognised *intent* verb
("show me", "how much"), and when the grammar matched the verb without
resolving the subject it fell back to `_DEFAULT_DESCRIBE_METRICS` — every
metric we do have — and the narrator picked the first one.

That is the single worst failure mode an industrial copilot has. Not "I don't
know", not a wrong number, but a **confident, correct, verified answer about a
different sensor than the one the engineer asked about.** Every downstream
guarantee holds perfectly: the plan validated, the arithmetic was exact, every
numeral traced to a slot. The answer is still useless and misleading, because
the question was about a bearing and the answer is about ambient air.

The existing refusal test passed only by luck. It used "vibration signature",
which shares no token with any synonym, so nothing matched and the planner
declined for the right reason by accident.

THE RULE
--------
A closed vocabulary tells us what we HAVE. This is the other half: detecting
that a question asks for something we do NOT have, and naming it rather than
substituting the nearest thing we do.

Two checks, both conservative — they only fire on positive evidence that a
foreign quantity was named, never on merely unfamiliar wording:

  1. NAMED QUANTITY. A curated list of quantities common in industrial plants
     and absent from this process. Explicit rather than inferred: a plant
     deploying this writes its own list from the tags it does not have.

  2. QUALIFIED SENSOR. "<qualifier> temperature" where the qualifier is not one
     the semantic layer recognises. We measure air and process temperature;
     "bearing temperature", "coolant temperature" and "oil temperature" are
     different instruments on different parts of the machine, and answering
     about ours when asked about theirs is silent substitution.

Both name the offending term in the refusal, because "I cannot answer that" is
much less useful to an engineer than "there is no bearing temperature sensor in
this dataset".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["UnknownQuantity", "detect_unknown_quantity"]


# Quantities a plant plausibly measures that this process does not expose.
# Deliberately a list rather than a heuristic: a wrong guess here either refuses
# an answerable question or lets a foreign one through, and both failures should
# be traceable to a line somebody wrote on purpose.
FOREIGN_QUANTITIES: dict[str, str] = {
    "vibration": "vibration",
    "acceleration": "accelerometer",
    "accelerometer": "accelerometer",
    "acoustic": "acoustic",
    "noise level": "acoustic",
    "decibel": "acoustic",
    "loud": "acoustic",
    "pressure": "pressure",
    "psi": "pressure",
    "bar pressure": "pressure",
    "flow rate": "flow",
    "flowrate": "flow",
    "voltage": "voltage",
    "amperage": "current",
    "current draw": "current",
    "humidity": "humidity",
    "viscosity": "viscosity",
    "lubricant": "lubrication",
    "lubrication": "lubrication",
    "runout": "spindle runout",
    "alignment": "alignment",
    "backlash": "backlash",
    "coolant": "coolant",
    "chatter": "chatter",
    "surface finish": "surface finish",
    "roughness": "surface roughness",
    "cost": "cost",
    "price": "cost",
    "budget": "cost",
    "operator": "personnel",
    "technician": "personnel",
    "weather": "weather",
    "downtime": "downtime records",
    "work order": "work orders",
    "spare part": "parts inventory",
}

# Qualifiers our temperature channels legitimately answer to. Anything else in
# front of "temperature" is a different thermocouple.
KNOWN_TEMPERATURE_QUALIFIERS = frozenset({
    "air", "ambient", "process", "tool", "working", "room",
    "delta", "differential", "difference",
})

_TEMP_QUALIFIER = re.compile(
    r"\b(?:the\s+)?([a-z]+)\s+(?:temperature|temp)\b", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class UnknownQuantity:
    """A named thing the question wants and this process does not measure."""

    term: str
    quantity: str

    @property
    def message(self) -> str:
        return (
            f"This dataset has no {self.quantity} measurement, so I cannot answer "
            f"a question about {self.term}. The available channels are air "
            f"temperature, process temperature, rotational speed, torque and tool "
            f"wear. Answering with one of those instead would be a different "
            f"question from the one you asked."
        )


def detect_unknown_quantity(question: str) -> UnknownQuantity | None:
    """Name what the question wants and we do not have, or None."""
    text = question.lower()

    for phrase, quantity in FOREIGN_QUANTITIES.items():
        if re.search(rf"\b{re.escape(phrase)}", text):
            return UnknownQuantity(term=phrase, quantity=quantity)

    for match in _TEMP_QUALIFIER.finditer(text):
        qualifier = match.group(1).lower()
        if qualifier in KNOWN_TEMPERATURE_QUALIFIERS:
            continue
        # Articles and quantifiers are not qualifiers; "the temperature" and
        # "average temperature" are questions about our own channels.
        if qualifier in {
            "the", "a", "an", "average", "mean", "median", "typical", "normal",
            "high", "low", "max", "maximum", "min", "minimum", "current", "this",
            "what", "is", "of", "in", "at", "and", "or", "machine", "cycle",
        }:
            continue
        return UnknownQuantity(
            term=f"{qualifier} temperature",
            quantity=f"{qualifier} temperature",
        )
    return None
