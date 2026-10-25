"""Sample source timelines onto the clustering analysis grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atmos_cluster.damf.timeline import Timeline, pos_at, step_at


@dataclass(frozen=True)
class FrameTracks:
    frame_times: np.ndarray
    positions: np.ndarray
    active: np.ndarray
    sizes: np.ndarray
    importance: np.ndarray
    dialog: np.ndarray
    music: np.ndarray
    zones: list[str]

    @property
    def n_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_sources(self) -> int:
        return int(self.positions.shape[1])


def sample_timelines(
    sources: list[Timeline],
    n_frames: int,
    hop: int,
    total_frames: int,
) -> FrameTracks:
    frame_times = np.arange(n_frames, dtype=np.float64) * hop
    frame_times = np.minimum(frame_times, max(total_frames - 1, 0))
    return FrameTracks(
        frame_times=frame_times,
        positions=np.stack(
            [pos_at(source.times, source.pos, frame_times) for source in sources], axis=1
        ).astype(np.float64),
        active=np.stack(
            [step_at(source.times, source.active, frame_times) for source in sources], axis=1
        ).astype(bool),
        sizes=np.stack(
            [step_at(source.times, source.size, frame_times) for source in sources], axis=1
        ).astype(np.float64),
        importance=np.stack(
            [step_at(source.times, source.importance, frame_times) for source in sources],
            axis=1,
        ).astype(np.float64),
        dialog=np.stack(
            [step_at(source.times, source.dialog, frame_times) for source in sources], axis=1
        ).astype(np.float64),
        music=np.stack(
            [step_at(source.times, source.music, frame_times) for source in sources], axis=1
        ).astype(np.float64),
        zones=[source.zones for source in sources],
    )


def content_prior(tracks: FrameTracks, dialog_boost_db: float) -> np.ndarray:
    prior = np.maximum(tracks.importance, 0.0)
    if dialog_boost_db:
        boost = 10.0 ** (dialog_boost_db / 20.0)
        prior = prior * np.where(tracks.dialog >= 0.5, boost, 1.0)
    return prior
