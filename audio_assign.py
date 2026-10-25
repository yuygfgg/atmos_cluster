"""Audio-reconstruction-driven source-to-cluster assignment.

The positional assignment modes in :mod:`atmos_cluster.clustering` choose gains
from geometry alone.  This mode instead minimizes the error between the
source scene and the clustered scene rendered through a small virtual speaker
basis.  All inputs are treated equally; there is no bed/object special case.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from atmos_cluster.config import SILENCE_POWER
from atmos_cluster.panning import VIRTUAL_SPEAKERS, panning_matrix
from atmos_cluster.reconstruction import solve_gains, source_covariance

__all__ = ["VIRTUAL_SPEAKERS", "audio_reconstruction_gains", "panning_matrix"]


def audio_reconstruction_gains(
    block: np.ndarray,
    positions: np.ndarray,
    sizes: np.ndarray,
    active: np.ndarray,
    selected: Sequence[int],
    centroids: np.ndarray,
    *,
    width_weight: float,
    spatial_weight: float = 0.02,
    iterations: int = 8,
    anchor_indices: Sequence[int] | None = None,
    source_cov: np.ndarray | None = None,
    source_weighting: str = "none",
    importance: np.ndarray | None = None,
) -> np.ndarray:
    """Choose gains that minimise rendered source-vs-cluster reconstruction error."""
    block = np.asarray(block, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    power = np.mean(np.square(block), axis=0)
    audible = np.asarray(active, dtype=bool) & (power > SILENCE_POWER)
    if source_cov is None:
        source_cov = source_covariance(block, audible, importance, source_weighting)
    if not np.any(audible):
        return np.zeros((positions.shape[0], len(selected)), dtype=np.float64)
    source_pan = panning_matrix(positions)
    cluster_pan = panning_matrix(centroids)
    return solve_gains(
        source_cov,
        source_pan,
        cluster_pan,
        positions,
        centroids,
        audible,
        selected,
        fixed_indices=() if anchor_indices is None else anchor_indices,
        sizes=sizes,
        width_weight=width_weight,
        spatial_weight=spatial_weight,
        iterations=iterations,
    )
