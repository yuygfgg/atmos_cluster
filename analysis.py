"""Per-frame spectral analysis feeding the clustering decisions."""

from __future__ import annotations

import numpy as np

from atmos_cluster.loudness import Loudness


def frame_excitation(
    loudness: Loudness,
    source: np.ndarray,
    channels: list[int],
    start: int,
    length: int,
    total_frames: int,
) -> np.ndarray:
    """Return the excitation pattern of one centred analysis window."""
    segment = np.zeros((loudness.n_fft, len(channels)), dtype=np.float32)
    center = start + length // 2
    window_start = center - loudness.n_fft // 2
    window_end = window_start + loudness.n_fft
    source_start = max(0, window_start)
    source_end = min(total_frames, window_end)
    if source_end > source_start:
        segment[source_start - window_start : source_end - window_start, :] = source[
            source_start:source_end, channels
        ]
    return loudness.excitation(loudness.band_levels(segment))
