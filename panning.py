"""Layer-aware panning used by the audio-reconstruction objective."""

from __future__ import annotations

import numpy as np

FLOOR_SPEAKERS = np.array(
    [
        [-1.0, 1.0, 0.0],  # L
        [1.0, 1.0, 0.0],  # R
        [0.0, 1.0, 0.0],  # C
        [-1.0, 0.0, 0.0],  # Ls
        [1.0, 0.0, 0.0],  # Rs
        [-1.0, -1.0, 0.0],  # Lrs
        [1.0, -1.0, 0.0],  # Rrs
    ],
    dtype=np.float64,
)

TOP_SPEAKERS = np.array(
    [
        [-1.0, 1.0, 1.0],  # Ltf
        [1.0, 1.0, 1.0],  # Rtf
        [-1.0, -1.0, 1.0],  # Ltr
        [1.0, -1.0, 1.0],  # Rtr
    ],
    dtype=np.float64,
)

VIRTUAL_SPEAKERS = np.vstack([FLOOR_SPEAKERS, TOP_SPEAKERS])

TOP_BLEND_START = 0.15
TOP_BLEND_END = 0.65


def _inverse_distance_gains(
    point: np.ndarray,
    speakers: np.ndarray,
    nearest: int,
) -> np.ndarray:
    distance = np.linalg.norm(speakers - point[None, :], axis=1)
    count = min(nearest, speakers.shape[0])
    order = np.argsort(distance)[:count]
    weights = 1.0 / np.square(distance[order] + 0.05)
    norm = float(np.linalg.norm(weights))
    gains = np.zeros(speakers.shape[0], dtype=np.float64)
    gains[order] = weights / max(norm, 1e-30)
    return gains


def panning_vector(position: np.ndarray) -> np.ndarray:
    """Power-normalized 11-channel gain vector for one object position."""
    point = np.asarray(position, dtype=np.float64)
    blend = (point[2] - TOP_BLEND_START) / (TOP_BLEND_END - TOP_BLEND_START)
    elevation = float(np.clip(blend, 0.0, 1.0))

    gains = np.zeros(VIRTUAL_SPEAKERS.shape[0], dtype=np.float64)
    if elevation < 1.0:
        floor = _inverse_distance_gains(point, FLOOR_SPEAKERS, 4)
        gains[: FLOOR_SPEAKERS.shape[0]] = np.sqrt(1.0 - elevation) * floor
    if elevation > 0.0:
        top = _inverse_distance_gains(point, TOP_SPEAKERS, 4)
        gains[FLOOR_SPEAKERS.shape[0] :] = np.sqrt(elevation) * top

    norm = float(np.linalg.norm(gains))
    if norm <= 1e-30:
        return panning_vector(np.array([point[0], point[1], 0.0]))
    return gains / norm


def panning_matrix(positions: np.ndarray) -> np.ndarray:
    """Layer-aware panning gains, shape ``(speakers, positions)``."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n, 3)")
    if positions.shape[0] == 0:
        return np.zeros((VIRTUAL_SPEAKERS.shape[0], 0), dtype=np.float64)
    return np.stack([panning_vector(position) for position in positions], axis=1)
