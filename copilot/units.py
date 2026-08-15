"""Dimensional type system.

Unit mismatch is a documented, deployment-killing class of bug in industrial
systems: one site reports °C, another K; one reports N·m, another lbf·ft. The
arithmetic succeeds and the answer is silently wrong.

This module makes that a *validation error* rather than a runtime surprise. It
is the same machinery that powers rule discovery (see discovery/dimensional.py):
knowing units both prevents bugs and collapses the hypothesis space.

Two subtleties that matter in practice and are handled explicitly:

  * **Absolute vs difference.** 300 K and a 10 K temperature *rise* have the same
    dimension but are not interchangeable. Converting 10 °C-of-rise to K is +0,
    not +273.15. Comparing an absolute to a delta is a bug and is rejected.
  * **Angle is dimensionless, deliberately.** A radian is m/m. Tracking angle as
    a base dimension is tempting — it looks like it would catch rpm/rad-per-second
    confusion — but it breaks the identity `torque × ω = power`, which holds
    *precisely because* radians are dimensionless. rpm and rad/s therefore share
    dimension T⁻¹ and differ by a scale factor of 2π/60, which `to_si` applies.
    That is what actually prevents the bug: any derived quantity is computed in
    SI, so "torque × rpm-as-a-raw-number" is unrepresentable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

__all__ = [
    "Dimension",
    "Unit",
    "UnitError",
    "DIMENSIONLESS",
    "UNITS",
    "unit",
    "convert",
    "compatible",
    "assert_compatible",
]


class UnitError(ValueError):
    """Raised when an operation is not dimensionally coherent."""


@dataclass(frozen=True, slots=True)
class Dimension:
    """Exponents over the SI base dimensions this domain needs."""

    mass: int = 0
    length: int = 0
    time: int = 0
    temperature: int = 0

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(
            self.mass + other.mass,
            self.length + other.length,
            self.time + other.time,
            self.temperature + other.temperature,
        )

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(
            self.mass - other.mass,
            self.length - other.length,
            self.time - other.time,
            self.temperature - other.temperature,
        )

    def __str__(self) -> str:
        parts = [
            f"{sym}^{exp}" if exp != 1 else sym
            for sym, exp in (
                ("M", self.mass),
                ("L", self.length),
                ("T", self.time),
                ("Θ", self.temperature),
            )
            if exp
        ]
        return "·".join(parts) or "1"


DIMENSIONLESS: Final = Dimension()
_TEMPERATURE: Final = Dimension(temperature=1)
_TIME: Final = Dimension(time=1)
_ANGULAR_VELOCITY: Final = Dimension(time=-1)
_TORQUE: Final = Dimension(mass=1, length=2, time=-2)
_POWER: Final = Dimension(mass=1, length=2, time=-3)
_STRAIN: Final = Dimension(mass=1, length=2, time=-1)  # torque × time


@dataclass(frozen=True, slots=True)
class Unit:
    """A named unit: a dimension, a scale to SI, and an optional zero offset.

    `to_si(x) = x * scale + offset` for absolute quantities; the offset is
    skipped for differences.
    """

    symbol: str
    dimension: Dimension
    scale: float = 1.0
    offset: float = 0.0
    is_delta: bool = False
    label: str = ""

    def to_si(self, value: float) -> float:
        return value * self.scale + (0.0 if self.is_delta else self.offset)

    def from_si(self, value: float) -> float:
        return (value - (0.0 if self.is_delta else self.offset)) / self.scale

    def as_delta(self) -> Unit:
        """The difference-flavoured counterpart of this unit."""
        if self.is_delta:
            return self
        return Unit(
            symbol=f"Δ{self.symbol}",
            dimension=self.dimension,
            scale=self.scale,
            offset=self.offset,
            is_delta=True,
            label=f"{self.label} difference" if self.label else "",
        )

    def __str__(self) -> str:
        return self.symbol


# --- Registry -------------------------------------------------------------
# Only units this project actually uses. A closed registry is deliberate: an
# unknown unit string should fail loudly, not be silently coerced.

UNITS: Final[dict[str, Unit]] = {
    u.symbol: u
    for u in (
        Unit("", DIMENSIONLESS, label="dimensionless"),
        Unit("ratio", DIMENSIONLESS, label="ratio"),
        Unit("rate", DIMENSIONLESS, label="rate"),
        Unit("count", DIMENSIONLESS, label="count"),
        # A percentage is a ratio scaled by 1/100, so convert('%', 'ratio') works.
        Unit("%", DIMENSIONLESS, scale=0.01, label="percent"),
        Unit("Δ%", DIMENSIONLESS, scale=0.01, is_delta=True, label="percentage point difference"),
        # temperature — absolute
        Unit("K", _TEMPERATURE, label="kelvin"),
        Unit("degC", _TEMPERATURE, offset=273.15, label="degrees celsius"),
        # temperature — difference (ΔK and Δ°C are numerically identical)
        Unit("ΔK", _TEMPERATURE, is_delta=True, label="kelvin difference"),
        Unit("ΔdegC", _TEMPERATURE, is_delta=True, label="celsius difference"),
        # Difference flavours for every quantity a margin is expressed in.
        # Numerically identical to their absolute counterparts (offset 0), but
        # declaring them separately means "power margin > 3500 W" — comparing a
        # headroom to an absolute threshold — is rejected rather than computed.
        Unit("Δrpm", _ANGULAR_VELOCITY, scale=2 * math.pi / 60, is_delta=True,
             label="rpm difference"),
        Unit("ΔN·m", _TORQUE, is_delta=True, label="newton metre difference"),
        Unit("ΔW", _POWER, is_delta=True, label="watt difference"),
        Unit("Δmin", _TIME, scale=60.0, is_delta=True, label="minute difference"),
        Unit("Δmin·N·m", _STRAIN, scale=60.0, is_delta=True,
             label="minute newton metre difference"),
        # rotation
        Unit("rpm", _ANGULAR_VELOCITY, scale=2 * math.pi / 60, label="revolutions per minute"),
        Unit("rad/s", _ANGULAR_VELOCITY, label="radians per second"),
        # mechanical
        Unit("N·m", _TORQUE, label="newton metre"),
        Unit("lbf·ft", _TORQUE, scale=1.3558179483314004, label="pound-force foot"),
        Unit("W", _POWER, label="watt"),
        Unit("kW", _POWER, scale=1000.0, label="kilowatt"),
        # time
        Unit("s", _TIME, label="second"),
        Unit("min", _TIME, scale=60.0, label="minute"),
        Unit("h", _TIME, scale=3600.0, label="hour"),
        # composite
        Unit("min·N·m", _STRAIN, scale=60.0, label="minute newton metre"),
    )
}

# Tolerated spellings from CSV headers, OPC-UA tags, and human input.
_ALIASES: Final[dict[str, str]] = {
    "k": "K",
    "kelvin": "K",
    "c": "degC",
    "°c": "degC",
    "celsius": "degC",
    "deg_c": "degC",
    "nm": "N·m",
    "n*m": "N·m",
    "n.m": "N·m",
    "newton_metre": "N·m",
    "newton_meter": "N·m",
    "minnm": "min·N·m",
    "min*nm": "min·N·m",
    "min_nm": "min·N·m",
    "watt": "W",
    "watts": "W",
    "rev/min": "rpm",
    "r/min": "rpm",
    "rads": "rad/s",
    "rad_s": "rad/s",
    "minute": "min",
    "minutes": "min",
    "sec": "s",
    "seconds": "s",
    "-": "",
    "none": "",
    "unitless": "",
    "percent": "%",
    "pct": "%",
    "pp": "Δ%",
}


def unit(symbol: str | Unit | None) -> Unit:
    """Resolve a unit symbol, tolerating common spellings.

    Raises UnitError on anything unrecognised — silence here is how bad data
    becomes a wrong answer.
    """
    if isinstance(symbol, Unit):
        return symbol
    if symbol is None:
        return UNITS[""]
    raw = symbol.strip()
    if raw in UNITS:
        return UNITS[raw]
    resolved = _ALIASES.get(raw.lower())
    if resolved is not None:
        return UNITS[resolved]
    raise UnitError(
        f"Unknown unit {symbol!r}. Known units: {', '.join(s for s in UNITS if s)}"
    )


def compatible(a: str | Unit, b: str | Unit) -> bool:
    """True when two units measure the same kind of thing.

    Absolute and difference flavours of the same dimension are deliberately
    *incompatible*: comparing a process temperature to a temperature rise is a
    bug, not a conversion.
    """
    ua, ub = unit(a), unit(b)
    return ua.dimension == ub.dimension and ua.is_delta == ub.is_delta


def assert_compatible(a: str | Unit, b: str | Unit, *, context: str = "") -> None:
    ua, ub = unit(a), unit(b)
    if compatible(ua, ub):
        return
    where = f" in {context}" if context else ""
    if ua.dimension == ub.dimension:
        raise UnitError(
            f"Cannot mix an absolute quantity with a difference{where}: "
            f"{ua.symbol} vs {ub.symbol}. A temperature and a temperature rise "
            f"are not the same kind of quantity."
        )
    raise UnitError(
        f"Incompatible units{where}: {ua.symbol} [{ua.dimension}] vs "
        f"{ub.symbol} [{ub.dimension}]"
    )


def convert(value: float, source: str | Unit, target: str | Unit) -> float:
    """Convert between compatible units. Raises UnitError otherwise."""
    us, ut = unit(source), unit(target)
    assert_compatible(us, ut, context="conversion")
    return ut.from_si(us.to_si(value))


def derived(*factors: tuple[str | Unit, int]) -> Dimension:
    """Dimension of a product of powers, e.g. derived(('N·m', 1), ('rad/s', 1))."""
    result = DIMENSIONLESS
    for sym, power in factors:
        d = unit(sym).dimension
        for _ in range(abs(power)):
            result = result * d if power > 0 else result / d
    return result
