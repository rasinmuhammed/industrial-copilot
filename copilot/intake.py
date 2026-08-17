"""The transport boundary: what arrives is not what was measured.

WHY THIS EXISTS
---------------
The observer decides whether a *reading* deserves belief. This decides whether a
*message* deserves to become a reading at all - a different question, and the
one every streaming deployment actually trips over.

A CSV replay hides all of it: rows arrive once, in order, with a clean index. A
plant does not. Between the sensor and this process sit a PLC scan cycle, an
OPC-UA subscription, a broker with at-least-once delivery, a network that
buffers and bursts, and at least two clocks that disagree. The failures that
follow are not exotic; they are the normal weather of industrial messaging:

  * **Duplicates.** MQTT QoS 1 and Kafka both guarantee at-least-once, which
    means "more than once" is the contract, not a fault. A duplicate feeds the
    same sample into the observer twice, halving its effective noise estimate
    and biasing every statistic built on it.
  * **Out-of-order arrival.** Two gateways, two paths, different latency. A
    late sample applied as if current drags the state estimate backwards.
  * **Gaps.** A dropped connection leaves a hole. Computing a slope across it as
    though the samples were adjacent invents a trend that never happened - and
    the forecast built on that slope is confidently wrong.
  * **Clock skew.** NTP steps, DST, a gateway with a dead RTC reporting 1970.
    An event timestamped in the future poisons any watermark that trusts it.
  * **Staleness.** The last message arrived four hours ago. The margin computed
    from it is not current, and presenting it as current is the quietest lie
    this system could tell.

TWO CLOCKS, ALWAYS
------------------
Every message carries an **event time** (when the measurement happened) and an
**ingestion time** (when we saw it). Conflating them is the root of most of the
above. Ordering, staleness and gaps are all event-time questions; latency and
liveness are ingestion-time questions. Keeping both lets each be answered with
the right one.

WHAT THIS GUARANTEES
--------------------
Nothing reaches the observer that is a duplicate, out of order beyond the
declared lateness bound, or timestamped implausibly. Everything that is dropped
is counted and attributable, because a silent drop is indistinguishable from a
sensor that stopped - and those demand opposite responses.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterator

__all__ = [
    "Verdict",
    "Envelope",
    "IntakeStats",
    "Intake",
    "IntakeConfig",
]

# ── Policy. Stated as bounds, so behaviour is a consequence of a declared
# tolerance rather than of whatever the code happened to do. ──────────────────
DEFAULT_LATENESS_S = 30.0        # how late a sample may arrive and still be used
DEFAULT_STALE_S = 300.0          # beyond this, a reading is not "current"
DEFAULT_FUTURE_SKEW_S = 5.0      # event times further ahead than this are wrong
DEFAULT_GAP_FACTOR = 3.0         # gap > this x the observed period is a hole
DEDUPE_MEMORY = 4096             # recent keys retained per machine


class Verdict(StrEnum):
    """What happened to a message. Every one is counted."""

    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"          # already seen; at-least-once delivery
    TOO_LATE = "too_late"            # older than the lateness bound
    FUTURE = "future"                # event time ahead of now; clock is wrong
    MALFORMED = "malformed"          # no usable timestamp or identity
    GAP_BEFORE = "gap_before"        # accepted, but a hole precedes it


@dataclass(frozen=True, slots=True)
class Envelope:
    """A message that has passed the boundary, with what we learned about it."""

    machine_id: str
    event_time: float
    ingest_time: float
    payload: dict[str, Any]
    verdict: Verdict
    gap_s: float = 0.0
    lateness_s: float = 0.0
    stale: bool = False
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.verdict in (Verdict.ACCEPTED, Verdict.GAP_BEFORE)

    @property
    def transport_latency_s(self) -> float:
        return max(self.ingest_time - self.event_time, 0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "machine": self.machine_id,
            "event_time": self.event_time,
            "verdict": self.verdict.value,
            "latency_s": round(self.transport_latency_s, 3),
            "gap_s": round(self.gap_s, 3),
            "stale": self.stale,
            "reason": self.reason,
        }


@dataclass(slots=True)
class IntakeStats:
    """Counters, because a silent drop is indistinguishable from a dead sensor."""

    seen: int = 0
    accepted: int = 0
    duplicate: int = 0
    too_late: int = 0
    future: int = 0
    malformed: int = 0
    gaps: int = 0
    stale: int = 0
    max_latency_s: float = 0.0

    def record(self, env: Envelope) -> None:
        self.seen += 1
        match env.verdict:
            case Verdict.ACCEPTED | Verdict.GAP_BEFORE:
                self.accepted += 1
                if env.verdict is Verdict.GAP_BEFORE:
                    self.gaps += 1
            case Verdict.DUPLICATE:
                self.duplicate += 1
            case Verdict.TOO_LATE:
                self.too_late += 1
            case Verdict.FUTURE:
                self.future += 1
            case Verdict.MALFORMED:
                self.malformed += 1
        if env.stale:
            self.stale += 1
        self.max_latency_s = max(self.max_latency_s, env.transport_latency_s)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "accepted": self.accepted,
            "dropped": self.seen - self.accepted,
            "duplicate": self.duplicate,
            "too_late": self.too_late,
            "future": self.future,
            "malformed": self.malformed,
            "gaps": self.gaps,
            "stale": self.stale,
            "max_latency_s": round(self.max_latency_s, 3),
            "accept_rate": round(self.accepted / self.seen, 4) if self.seen else 0.0,
        }


@dataclass(frozen=True, slots=True)
class IntakeConfig:
    lateness_s: float = DEFAULT_LATENESS_S
    stale_s: float = DEFAULT_STALE_S
    future_skew_s: float = DEFAULT_FUTURE_SKEW_S
    gap_factor: float = DEFAULT_GAP_FACTOR
    event_time_field: str = "ts"
    machine_field: str = "machine_id"
    sequence_field: str = "seq"


@dataclass(slots=True)
class _MachineIntake:
    """Per-machine ordering state. Machines are independent; a late message on
    one must not suppress a timely message on another."""

    last_event_time: float | None = None
    high_watermark: float = float("-inf")
    seen_keys: OrderedDict[Any, None] = field(default_factory=OrderedDict)
    periods: deque[float] = field(default_factory=lambda: deque(maxlen=64))

    def median_period(self) -> float | None:
        if len(self.periods) < 8:
            return None
        ordered = sorted(self.periods)
        return ordered[len(ordered) // 2]

    def remember(self, key: Any) -> bool:
        """True if this key is new. Bounded memory, oldest evicted first."""
        if key in self.seen_keys:
            return False
        self.seen_keys[key] = None
        if len(self.seen_keys) > DEDUPE_MEMORY:
            self.seen_keys.popitem(last=False)
        return True


@dataclass(slots=True)
class Intake:
    """The boundary. Feed it raw messages, take usable readings out."""

    config: IntakeConfig = field(default_factory=IntakeConfig)
    stats: IntakeStats = field(default_factory=IntakeStats)
    _machines: dict[str, _MachineIntake] = field(default_factory=dict)

    def offer(self, message: dict[str, Any], *, now: float | None = None) -> Envelope:
        """Admit or reject one message. Never raises on bad input.

        A malformed message from a field gateway is data, not an exception. It
        gets counted and dropped; it does not take the pipeline down with it.
        """
        now = time.time() if now is None else now
        cfg = self.config

        machine = message.get(cfg.machine_field)
        if machine is None:
            return self._reject(message, now, Verdict.MALFORMED,
                                f"no {cfg.machine_field!r} field")
        machine = str(machine)

        event_time = _as_epoch(message.get(cfg.event_time_field))
        if event_time is None:
            # No event time is survivable: treat arrival as the event and say
            # so. Refusing the message entirely would discard a real reading
            # over a metadata problem.
            event_time = now

        # ── Clock sanity. An event from the future is always wrong, and a
        # watermark that accepts one will reject every honest message after it.
        if event_time > now + cfg.future_skew_s:
            return self._reject(
                message, now, Verdict.FUTURE, machine=machine, event_time=event_time,
                reason=(f"event time is {event_time - now:.1f}s ahead of now; "
                        f"the source clock is wrong"),
            )

        state = self._machines.setdefault(machine, _MachineIntake())

        # ── Duplicates. At-least-once delivery means repeats are the contract.
        # Prefer an explicit sequence number; fall back to the event time, which
        # is the natural key for a periodic sampler.
        key = message.get(cfg.sequence_field, event_time)
        if not state.remember(key):
            return self._reject(
                message, now, Verdict.DUPLICATE, machine=machine, event_time=event_time,
                reason=f"already processed {cfg.sequence_field}={key!r}",
            )

        # ── Ordering. Anything older than the watermark minus the declared
        # lateness is too late to apply: the estimate has already moved past it,
        # and folding it in now would drag the state backwards.
        lateness = 0.0
        if state.high_watermark > float("-inf"):
            lateness = state.high_watermark - event_time
            if lateness > cfg.lateness_s:
                return self._reject(
                    message, now, Verdict.TOO_LATE, machine=machine,
                    event_time=event_time,
                    reason=(f"{lateness:.1f}s behind the watermark, past the "
                            f"{cfg.lateness_s:.0f}s lateness bound"),
                )

        # ── Gaps. A hole must be visible, because a slope computed across one
        # is a trend that never happened.
        gap = 0.0
        verdict = Verdict.ACCEPTED
        reason = ""
        if state.last_event_time is not None:
            delta = event_time - state.last_event_time
            if delta > 0:
                state.periods.append(delta)
            period = state.median_period()
            if period and delta > cfg.gap_factor * period:
                gap = delta
                verdict = Verdict.GAP_BEFORE
                reason = (
                    f"{delta:.1f}s since the previous sample against a typical "
                    f"{period:.1f}s; samples are missing and any rate of change "
                    f"across this hole is not observed"
                )

        stale = (now - event_time) > cfg.stale_s
        if stale and not reason:
            reason = f"measured {now - event_time:.0f}s ago; not a current reading"

        state.last_event_time = event_time
        state.high_watermark = max(state.high_watermark, event_time)

        env = Envelope(
            machine_id=machine, event_time=event_time, ingest_time=now,
            payload=message, verdict=verdict, gap_s=gap, lateness_s=max(lateness, 0.0),
            stale=stale, reason=reason,
        )
        self.stats.record(env)
        return env

    def accept(self, messages, *, now=None) -> Iterator[Envelope]:
        """Filter an iterable down to the messages worth acting on."""
        for message in messages:
            env = self.offer(message, now=now)
            if env.usable:
                yield env

    def _reject(
        self, message: dict[str, Any], now: float, verdict: Verdict,
        reason: str = "", machine: str = "unknown", event_time: float | None = None,
    ) -> Envelope:
        env = Envelope(
            machine_id=machine, event_time=now if event_time is None else event_time,
            ingest_time=now, payload=message, verdict=verdict, reason=reason,
        )
        self.stats.record(env)
        return env


def _as_epoch(value: Any) -> float | None:
    """Parse a timestamp without trusting its shape.

    Field gateways emit seconds, milliseconds, ISO-8601 with and without a zone,
    and occasionally an empty string. Each is handled; anything else is treated
    as absent rather than guessed at.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        # Milliseconds are the common alternative to seconds, and the two are
        # distinguishable by magnitude: 1e11 seconds is the year 5138.
        return v / 1000.0 if v > 1e11 else v
    if isinstance(value, str):
        text = value.strip()
        try:
            v = float(text)
            return v / 1000.0 if v > 1e11 else v
        except ValueError:
            pass
        from datetime import datetime, timezone

        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            # A naive timestamp is ambiguous. UTC is the least-surprising
            # reading for machine data and, unlike local time, does not shift
            # twice a year underneath a running deployment.
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return None
