"""Read source metadata events and write clustered DAMF metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from atmos_cluster.config import DEFAULT_FS, LFE_ID, OUT_ID_BASE


@dataclass
class ClusterTracks:
    """Per-frame metadata of the reduced object set."""

    positions: np.ndarray
    sizes: np.ndarray
    importance: np.ndarray
    dialog: np.ndarray
    music: np.ndarray
    active: np.ndarray
    zones: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_clusters(self) -> int:
        return int(self.positions.shape[1])

    @classmethod
    def empty(cls, n_frames: int, n_clusters: int) -> ClusterTracks:
        return cls(
            positions=np.zeros((n_frames, n_clusters, 3), dtype=np.float64),
            sizes=np.zeros((n_frames, n_clusters), dtype=np.float64),
            importance=np.zeros((n_frames, n_clusters), dtype=np.float64),
            dialog=np.full((n_frames, n_clusters), -1.0, dtype=np.float64),
            music=np.full((n_frames, n_clusters), -1.0, dtype=np.float64),
            active=np.zeros((n_frames, n_clusters), dtype=bool),
            zones=["all"] * n_clusters,
        )


def load_events(path: Path) -> tuple[int, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    rate = int(meta.get("sampleRate") or DEFAULT_FS)
    raw_events = meta.get("events") or []
    events: list[dict[str, Any]] = []
    current_sample = 0
    for event in raw_events:
        sample = event.get("samplePos")
        if sample is not None:
            current_sample = int(sample)
            events.append(event)
            continue
        copy = dict(event)
        copy["samplePos"] = current_sample
        events.append(copy)
    return rate, events


def write_metadata(
    path: Path,
    fs: int,
    tracks: ClusterTracks,
    hop: int,
    has_lfe: bool,
    interval_frames: int = 1,
) -> None:
    last: list[dict[str, Any] | None] = [None] * tracks.n_clusters
    last_time = [-1] * tracks.n_clusters
    interval = max(1, int(interval_frames))
    position_threshold = 1e-5 if interval <= 1 else 0.02
    importance_threshold = 1e-4 if interval <= 1 else 0.05
    max_interval = interval * 2 if interval > 1 else 0
    with path.open("w", encoding="utf-8") as f:
        f.write(f"sampleRate: {fs}\n")
        f.write("events:\n")
        if has_lfe:
            f.write(f"  - ID: {LFE_ID}\n")
            f.write("    samplePos: 0\n")
            f.write("    active: true\n")
            f.write("    importance: 1.0\n")
            f.write("    gain: 0\n")
            f.write("    rampLength: 1536\n")
            f.write("    trimBypass: false\n")
            f.write("    headTrackMode: undefined\n")
            f.write("    binauralRenderMode: off\n")

        for frame in range(tracks.n_frames):
            sample_pos = frame * hop
            for cluster in range(tracks.n_clusters):
                state = {
                    "pos": tracks.positions[frame, cluster],
                    "active": bool(tracks.active[frame, cluster]),
                    "size": float(tracks.sizes[frame, cluster]),
                    "importance": float(tracks.importance[frame, cluster]),
                    "dialog": float(tracks.dialog[frame, cluster]),
                    "music": float(tracks.music[frame, cluster]),
                    "zones": tracks.zones[cluster],
                }
                previous = last[cluster]
                elapsed = sample_pos - last_time[cluster]
                if previous is None:
                    changed = True
                else:
                    active_changed = bool(state["active"]) != previous["active"]
                    significant = (
                        active_changed
                        or abs(state["size"] - previous["size"]) > 1e-5
                        or abs(state["importance"] - previous["importance"])
                        > importance_threshold
                        or abs(state["dialog"] - previous["dialog"]) > 1e-4
                        or abs(state["music"] - previous["music"]) > 1e-4
                        or state["zones"] != previous["zones"]
                        or float(np.linalg.norm(state["pos"] - previous["pos"]))
                        > position_threshold
                    )
                    due = elapsed >= interval * hop
                    forced = max_interval > 0 and elapsed >= max_interval * hop
                    changed = (
                        (significant and due)
                        or forced
                        or (active_changed and elapsed >= hop)
                    )
                if not changed:
                    continue
                position = state["pos"]
                x = float(np.clip(position[0], -1.0, 1.0))
                y = float(np.clip(position[1], -1.0, 1.0))
                z = float(np.clip(position[2], 0.0, 1.0))
                f.write(f"  - ID: {OUT_ID_BASE + cluster}\n")
                f.write(f"    samplePos: {sample_pos}\n")
                f.write(f"    active: {'true' if state['active'] else 'false'}\n")
                f.write(f"    pos: [{x:.6f}, {y:.6f}, {z:.6f}]\n")
                f.write("    snap: false\n")
                f.write("    elevation: true\n")
                f.write(f"    zones: {state['zones']}\n")
                f.write(f"    size: {state['size']:.6f}\n")
                f.write(f"    importance: {state['importance']:.6f}\n")
                f.write("    gain: 0\n")
                f.write("    rampLength: 1536\n")
                f.write("    trimBypass: false\n")
                f.write(f"    dialog: {state['dialog']:.6f}\n")
                f.write(f"    music: {state['music']:.6f}\n")
                f.write("    screenFactor: 0.0\n")
                f.write("    depthFactor: 0.25\n")
                f.write("    headTrackMode: undefined\n")
                f.write("    binauralRenderMode: undefined\n")
                last[cluster] = state
                last_time[cluster] = sample_pos
