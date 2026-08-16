"""Where plant data actually comes from.

WHY THIS EXISTS
---------------
Everything upstream of here read a CSV. That is fine for a dataset and useless
for a plant, and the gap is not a detail — connector coverage is the moat the
incumbents actually own. This does not close that moat. What it does is make the
boundary explicit, so the rest of the system is written against a *source of
readings* rather than against a file, and adding a protocol is a class rather
than a refactor.

THE SHAPE
---------
A `Source` yields raw messages. It does no validation, no ordering, no
deduplication — those belong to `intake.Intake`, which every source feeds,
because at-least-once delivery and clock skew are properties of the transport
and not of any one protocol. A source that tried to be clean would duplicate
that logic once per protocol and get it subtly different each time.

    source ──▶ Intake ──▶ FleetObserver ──▶ margins ──▶ answer
    (raw)      (ordering,   (is the         (physics)
               dedupe,       signal
               clocks)       real?)

WHAT IS AND IS NOT HERE
-----------------------
`CsvSource` and `JsonlSource` are complete and tested: replay is what the demo
and the evals run on, and file-based history is what onboarding a new plant
starts from.

`MqttSource` and `OpcUaSource` are written against the real client libraries but
are **not** integration-tested, because doing that honestly needs a broker and a
PLC rather than a mock that would only prove the mock works. They are marked
`verified: False` and say so at runtime. Claiming otherwise would be the same
category of error as a constant described as measured when it was chosen.

Tag mapping is deliberately external. A plant's tag names are
`SITE1.LINE3.SPINDLE.TORQUE`, not `torque_nm`, and inventing a heuristic to
guess that correspondence is how a system silently reads the wrong sensor.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

__all__ = [
    "Source",
    "SourceInfo",
    "CsvSource",
    "JsonlSource",
    "MqttSource",
    "OpcUaSource",
    "TagMap",
    "build_source",
]


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """What a source is, and whether we have actually proven it works."""

    kind: str
    detail: str
    verified: bool
    requires: tuple[str, ...] = ()


@runtime_checkable
class Source(Protocol):
    """Yields raw messages. Ordering and validity are Intake's problem."""

    def info(self) -> SourceInfo: ...

    def read(self) -> Iterator[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class TagMap:
    """Plant tag name -> canonical channel name.

    Kept explicit and external on purpose. Guessing that
    `SITE1.LINE3.SPINDLE.TQ_FB` means `torque_nm` is how a system ends up
    confidently reading the wrong sensor, and the failure is invisible because
    every number downstream still looks plausible.
    """

    tags: dict[str, str] = field(default_factory=dict)
    machine_from: str = ""      # tag whose value names the machine
    time_from: str = ""         # tag carrying the event timestamp

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TagMap":
        import yaml

        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            tags=dict(raw.get("tags") or {}),
            machine_from=raw.get("machine_from", ""),
            time_from=raw.get("time_from", ""),
        )

    def apply(self, reading: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for tag, value in reading.items():
            canonical = self.tags.get(tag)
            if canonical:
                out[canonical] = value
        if self.machine_from and self.machine_from in reading:
            out["machine_id"] = str(reading[self.machine_from])
        if self.time_from and self.time_from in reading:
            out["ts"] = reading[self.time_from]
        return out


@dataclass(slots=True)
class CsvSource:
    """Replay a CSV as a stream. What the demo, evals and onboarding run on."""

    path: Path
    tag_map: TagMap | None = None
    machine_field: str = "machine_id"
    time_field: str = "ts"
    period_s: float = 0.0        # 0 = as fast as possible
    epoch: float | None = None   # synthesise timestamps when the file has none

    def info(self) -> SourceInfo:
        return SourceInfo("csv", str(self.path), verified=True)

    def read(self) -> Iterator[dict[str, Any]]:
        start = self.epoch if self.epoch is not None else time.time()
        with self.path.open(newline="") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                message = self.tag_map.apply(row) if self.tag_map else dict(row)
                message.setdefault(self.machine_field, "unknown")
                if self.time_field not in message:
                    # A file of process cycles has an index, not a clock. State
                    # the assumption in the data rather than leaving downstream
                    # code to invent one silently.
                    message[self.time_field] = start + i * max(self.period_s, 1.0)
                if self.period_s:
                    time.sleep(self.period_s)
                yield message


@dataclass(slots=True)
class JsonlSource:
    """One JSON object per line — the common historian export format."""

    path: Path
    tag_map: TagMap | None = None

    def info(self) -> SourceInfo:
        return SourceInfo("jsonl", str(self.path), verified=True)

    def read(self) -> Iterator[dict[str, Any]]:
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A corrupt line is data, not an exception. Intake counts
                    # the malformed message; the stream keeps running.
                    yield {"_malformed": line[:200]}
                    continue
                yield self.tag_map.apply(record) if self.tag_map else record


@dataclass(slots=True)
class MqttSource:
    """MQTT subscription. Written against paho-mqtt, NOT integration-tested.

    The topic-per-tag convention assumed here (`.../<machine>/<tag>`) is the
    common plain-MQTT layout. Sparkplug B instead carries a typed payload with
    its own birth/death semantics, and supporting it properly means decoding
    protobuf and tracking sequence numbers — worth doing against a real broker,
    not guessed at here.
    """

    host: str
    topic: str = "plant/+/+"
    port: int = 1883
    tag_map: TagMap | None = None
    qos: int = 1
    timeout_s: float = 60.0

    def info(self) -> SourceInfo:
        return SourceInfo(
            "mqtt", f"{self.host}:{self.port} {self.topic}",
            verified=False, requires=("paho-mqtt",),
        )

    def read(self) -> Iterator[dict[str, Any]]:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as e:
            raise RuntimeError(
                "MqttSource needs paho-mqtt:  pip install 'industrial-copilot[mqtt]'"
            ) from e

        import queue

        inbox: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=10_000)

        def on_message(_client, _userdata, msg) -> None:
            parts = msg.topic.split("/")
            machine = parts[-2] if len(parts) >= 2 else "unknown"
            tag = parts[-1]
            raw = msg.payload.decode("utf-8", "replace")
            try:
                value: Any = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            record = value if isinstance(value, dict) else {tag: value}
            record.setdefault("machine_id", machine)
            # Broker receipt time is NOT measurement time; leaving ts absent
            # lets Intake record that honestly rather than backfilling a clock.
            try:
                inbox.put_nowait(record)
            except queue.Full:
                # Backpressure. Dropping here is visible as a gap downstream,
                # which is the correct signal — unlike blocking the network
                # loop, which would stall every other machine's data too.
                pass

        client = mqtt.Client()
        client.on_message = on_message
        client.connect(self.host, self.port)
        client.subscribe(self.topic, qos=self.qos)
        client.loop_start()
        try:
            while True:
                try:
                    record = inbox.get(timeout=self.timeout_s)
                except queue.Empty:
                    return
                yield self.tag_map.apply(record) if self.tag_map else record
        finally:
            client.loop_stop()
            client.disconnect()


@dataclass(slots=True)
class OpcUaSource:
    """OPC-UA polling client. Written against asyncua, NOT integration-tested.

    Polling rather than subscribing is the deliberate choice for a first
    adapter: subscriptions are more efficient but add callback lifetime and
    reconnection semantics that are only worth getting right against a real
    server.

    OPC-UA is also where a real deployment would obtain ENGINEERING UNITS, from
    the EUInformation attribute on each node. That matters more than it sounds:
    the dimensional layer and the discovery step both take units as input, and
    a plant that publishes them gets the physics for free.
    """

    endpoint: str
    node_ids: dict[str, str] = field(default_factory=dict)   # channel -> NodeId
    machine_id: str = "unknown"
    period_s: float = 1.0

    def info(self) -> SourceInfo:
        return SourceInfo(
            "opcua", self.endpoint, verified=False, requires=("asyncua",),
        )

    def read(self) -> Iterator[dict[str, Any]]:
        try:
            from asyncua.sync import Client
        except ImportError as e:
            raise RuntimeError(
                "OpcUaSource needs asyncua:  pip install 'industrial-copilot[opcua]'"
            ) from e

        client = Client(url=self.endpoint)
        client.connect()
        try:
            nodes = {name: client.get_node(nid) for name, nid in self.node_ids.items()}
            while True:
                record: dict[str, Any] = {
                    "machine_id": self.machine_id,
                    "ts": time.time(),
                }
                for name, node in nodes.items():
                    try:
                        record[name] = node.read_value()
                    except Exception:
                        # A single bad tag must not kill the scan. Absent is a
                        # value the observer already knows how to handle: it
                        # predicts without updating and widens the interval.
                        record[name] = None
                yield record
                time.sleep(self.period_s)
        finally:
            client.disconnect()


def build_source(spec: dict[str, Any]) -> Source:
    """Construct a source from configuration.

        {"kind": "csv",  "path": "data/ai4i2020.csv"}
        {"kind": "mqtt", "host": "broker.plant.local", "topic": "plant/+/+"}
        {"kind": "opcua","endpoint": "opc.tcp://plc:4840", "node_ids": {...}}
    """
    kind = spec.get("kind", "csv")
    tag_map = TagMap.from_yaml(spec["tag_map"]) if spec.get("tag_map") else None
    match kind:
        case "csv":
            return CsvSource(path=Path(spec["path"]), tag_map=tag_map,
                             period_s=float(spec.get("period_s", 0.0)))
        case "jsonl":
            return JsonlSource(path=Path(spec["path"]), tag_map=tag_map)
        case "mqtt":
            return MqttSource(host=spec["host"], port=int(spec.get("port", 1883)),
                              topic=spec.get("topic", "plant/+/+"), tag_map=tag_map)
        case "opcua":
            return OpcUaSource(endpoint=spec["endpoint"],
                               node_ids=dict(spec.get("node_ids") or {}),
                               machine_id=spec.get("machine_id", "unknown"),
                               period_s=float(spec.get("period_s", 1.0)))
        case _:
            raise ValueError(
                f"unknown source kind {kind!r}; expected csv, jsonl, mqtt or opcua"
            )
