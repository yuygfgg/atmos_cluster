"""Spatial unmasking matrices used by the clustering criterion."""

from __future__ import annotations

import numpy as np


def width_distance_scale(sizes: np.ndarray, width_weight: float) -> np.ndarray:
    return 1.0 - float(width_weight) * np.clip(sizes, 0.0, 1.0)


def unmasking_matrix(positions: np.ndarray, fmax: float, d0: float) -> np.ndarray:
    distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    unmasking = fmax * (1.0 - np.exp(-distance / max(d0, 1e-9)))
    np.fill_diagonal(unmasking, 0.0)
    return unmasking


def clustering_unmasking(
    positions: np.ndarray,
    sizes: np.ndarray,
    width_weight: float,
    fmax: float,
    d0: float,
) -> np.ndarray:
    distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    distance = distance * width_distance_scale(sizes, width_weight)[:, None]
    unmasking = fmax * (1.0 - np.exp(-distance / max(d0, 1e-9)))
    np.fill_diagonal(unmasking, 0.0)
    return unmasking
