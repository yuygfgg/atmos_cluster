"""Build per-source metadata timelines from DAMF events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from atmos_cluster.config import BED_POSITIONS, DEFAULT_HOP
from atmos_cluster.damf.manifest import parse_bed_layout


@dataclass
class Timeline:
    obj_id: int
    channel: int
    fixed_pos: np.ndarray | None
    times: np.ndarray
    pos: np.ndarray
    active: np.ndarray
    size: np.ndarray
    importance: np.ndarray
    dialog: np.ndarray
    music: np.ndarray
    zones: str
    ramp: int

    @property
    def is_fixed(self) -> bool:
        return self.fixed_pos is not None


@dataclass(frozen=True)
class SourceSet:
    sources: list[Timeline]
    lfe_channel: int | None
    bed_count: int
    channels: list[int]

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    @property
    def fixed_indices(self) -> list[int]:
        return [index for index, source in enumerate(self.sources) if source.is_fixed]


def build_timeline(
    obj_id: int,
    channel: int,
    events: list[dict[str, Any]],
    fixed_pos: np.ndarray | None = None,
) -> Timeline:
    if fixed_pos is not None:
        position = np.asarray(fixed_pos, dtype=np.float64)
        return Timeline(
            obj_id=obj_id,
            channel=channel,
            fixed_pos=position,
            times=np.array([0], dtype=np.int64),
            pos=position.reshape(1, 3),
            active=np.array([True]),
            size=np.array([0.0]),
            importance=np.array([1.0]),
            dialog=np.array([-1.0]),
            music=np.array([-1.0]),
            zones="all",
            ramp=DEFAULT_HOP,
        )

    state: dict[str, Any] = {
        "pos": [0.0, 0.0, 0.0],
        "active": True,
        "size": 0.0,
        "importance": 1.0,
        "dialog": -1.0,
        "music": -1.0,
        "zones": "all",
        "ramp": DEFAULT_HOP,
    }
    rows: list[tuple[int, list[float], bool, float, float, float, float, str, int]] = []
    for event in events:
        if event.get("ID") != obj_id:
            continue
        sample_pos = int(event.get("samplePos", 0) or 0)
        if event.get("pos") is not None:
            position = list(event["pos"])
            if len(position) >= 3:
                state["pos"] = [
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                ]
        if event.get("active") is not None:
            state["active"] = bool(event["active"])
        if event.get("size") is not None:
            state["size"] = float(event["size"])
        if event.get("importance") is not None:
            state["importance"] = float(event["importance"])
        if event.get("dialog") is not None:
            state["dialog"] = float(event["dialog"])
        if event.get("music") is not None:
            state["music"] = float(event["music"])
        if event.get("zones") is not None:
            state["zones"] = str(event["zones"])
        if event.get("rampLength") is not None:
            state["ramp"] = int(event["rampLength"])
        row = (
            sample_pos,
            list(state["pos"]),
            bool(state["active"]),
            float(state["size"]),
            float(state["importance"]),
            float(state["dialog"]),
            float(state["music"]),
            str(state["zones"]),
            int(state["ramp"]),
        )
        if rows and rows[-1][0] == sample_pos:
            rows[-1] = row
        else:
            rows.append(row)

    if not rows:
        position = np.array([[0.0, 0.0, 0.0]])
        return Timeline(
            obj_id=obj_id,
            channel=channel,
            fixed_pos=None,
            times=np.array([0], dtype=np.int64),
            pos=position,
            active=np.array([True]),
            size=np.array([0.0]),
            importance=np.array([1.0]),
            dialog=np.array([-1.0]),
            music=np.array([-1.0]),
            zones="all",
            ramp=DEFAULT_HOP,
        )

    if rows[0][0] > 0:
        rows.insert(0, (0, *rows[0][1:]))

    return Timeline(
        obj_id=obj_id,
        channel=channel,
        fixed_pos=None,
        times=np.array([row[0] for row in rows], dtype=np.int64),
        pos=np.array([row[1] for row in rows], dtype=np.float64),
        active=np.array([row[2] for row in rows], dtype=bool),
        size=np.array([row[3] for row in rows], dtype=np.float64),
        importance=np.array([row[4] for row in rows], dtype=np.float64),
        dialog=np.array([row[5] for row in rows], dtype=np.float64),
        music=np.array([row[6] for row in rows], dtype=np.float64),
        zones=str(rows[0][7]),
        ramp=int(rows[0][8]),
    )


def step_at(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    index = np.searchsorted(times, query, side="right") - 1
    index = np.clip(index, 0, len(times) - 1)
    return values[index]


def pos_at(times: np.ndarray, positions: np.ndarray, query: np.ndarray) -> np.ndarray:
    if len(times) == 1:
        return np.repeat(positions[:1], len(query), axis=0)
    index = np.searchsorted(times, query, side="right") - 1
    index = np.clip(index, 0, len(times) - 2)
    t0 = times[index].astype(np.float64)
    t1 = times[index + 1].astype(np.float64)
    fraction = np.clip(
        (query.astype(np.float64) - t0) / np.maximum(t1 - t0, 1.0), 0.0, 1.0
    )
    return (
        positions[index] * (1.0 - fraction)[:, None]
        + positions[index + 1] * fraction[:, None]
    )


def build_sources(
    presentation: dict[str, Any],
    events: list[dict[str, Any]],
) -> SourceSet:
    beds, bed_count = parse_bed_layout(presentation)
    lfe_channel: int | None = None
    sources: list[Timeline] = []
    channels: list[int] = []

    for obj_id, name, channel in beds:
        if name.upper() == "LFE":
            if lfe_channel is None:
                lfe_channel = channel
            continue
        fixed = BED_POSITIONS.get(name.upper())
        if fixed is None:
            print(
                f"warning: unknown bed channel {name!r}; treating it as a fixed center object"
            )
            fixed = (0.0, 0.0, 0.0)
        sources.append(
            build_timeline(
                obj_id, channel, events, fixed_pos=np.asarray(fixed, dtype=np.float64)
            )
        )
        channels.append(channel)

    for index, obj in enumerate(presentation.get("objects") or []):
        obj_id = int(obj.get("ID"))
        channel = bed_count + index
        sources.append(build_timeline(obj_id, channel, events))
        channels.append(channel)

    return SourceSet(
        sources=sources,
        lfe_channel=lfe_channel,
        bed_count=bed_count,
        channels=channels,
    )
