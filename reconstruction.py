"""Audio-reconstruction-driven centroid selection and gain solving.

The objective is the rendered-speaker error between the original scene and the
clustered scene.  Centroids are chosen by their marginal residual reduction
(orthogonal matching pursuit) instead of by loudness alone, so a loud object
that is already covered by a nearby fixed bed does not consume a cluster.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from atmos_cluster.clustering import assign_gains
from atmos_cluster.panning import panning_matrix


def source_covariance(
    block: np.ndarray,
    active: np.ndarray,
    importance: np.ndarray | None = None,
    weighting: str = "none",
) -> np.ndarray:
    """Weighted source covariance ``X^T diag(w) X``, trace-normalized."""
    block = np.asarray(block, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    n_sources = block.shape[1]
    power = np.mean(np.square(block), axis=0)

    weights = np.ones(n_sources, dtype=np.float64)
    if weighting == "power":
        weights = active.astype(np.float64) / (power + 1e-8)
    elif weighting == "importance":
        if importance is not None:
            importance = np.maximum(np.asarray(importance, dtype=np.float64), 0.0)
            scale = float(importance.max())
            normalised = importance / scale if scale > 0.0 else np.zeros_like(importance)
        else:
            normalised = np.zeros(n_sources, dtype=np.float64)
        weights = active.astype(np.float64) * (0.05 + 0.95 * normalised) / (power + 1e-8)
    elif weighting != "none":
        raise ValueError(f"unknown source weighting: {weighting!r}")

    if np.any(active):
        mean_weight = float(weights[active].mean())
        if mean_weight > 0.0:
            weights /= mean_weight
    weights = np.clip(weights, 0.0, 1e6)

    scaled = block * np.sqrt(weights)[None, :]
    covariance = scaled.T @ scaled
    trace = float(np.trace(covariance))
    if trace > 0.0:
        covariance /= trace
    covariance += np.eye(n_sources) * 1e-9
    return covariance


def _project(gains: np.ndarray, free: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    gains = np.maximum(gains, 0.0)
    gains[~free] = anchors[~free]
    free_gains = gains[free]
    if free_gains.size:
        norms = np.sqrt(np.sum(np.square(free_gains), axis=1, keepdims=True))
        gains[free] = free_gains / np.maximum(norms, 1e-30)
    return gains


def solve_gains(
    covariance: np.ndarray,
    source_pan: np.ndarray,
    cluster_pan: np.ndarray,
    positions: np.ndarray,
    centroids: np.ndarray,
    active: np.ndarray,
    selected: Sequence[int],
    *,
    fixed_indices: Sequence[int] = (),
    sizes: np.ndarray | None = None,
    width_weight: float = 0.0,
    spatial_weight: float = 0.02,
    iterations: int = 24,
) -> np.ndarray:
    """Minimise the rendered reconstruction error over non-negative gains."""
    n_sources = positions.shape[0]
    n_clusters = len(selected)
    gains = assign_gains(
        positions,
        np.zeros(n_sources) if sizes is None else sizes,
        active,
        selected,
        "auto",
        width_weight,
    )
    if n_clusters == 0:
        return np.zeros((n_sources, 0), dtype=np.float64)

    selected_index = {index: cluster for cluster, index in enumerate(selected)}
    active = np.asarray(active, dtype=bool)
    fixed_set = {int(index) for index in fixed_indices}
    anchors = np.zeros((n_sources, n_clusters), dtype=np.float64)
    for index, cluster in selected_index.items():
        if index in fixed_set:
            anchors[index, cluster] = 1.0

    gains[~active] = 0.0
    for index, cluster in selected_index.items():
        if active[index]:
            gains[index] = 0.0
            gains[index, cluster] = 1.0

    anchored_rows = [index for index in fixed_indices if index in selected_index]
    if anchored_rows:
        gains[anchored_rows] = anchors[anchored_rows]
    free = active.copy()
    if anchored_rows:
        free[anchored_rows] = False
    if not np.any(free):
        return gains

    source_to_cluster = covariance @ (source_pan.T @ cluster_pan)
    cluster_cov = cluster_pan.T @ cluster_pan
    distance = np.sum(
        np.square(positions[:, None, :] - centroids[None, :, :]),
        axis=2,
    )
    scale = float(distance[active].mean()) if np.any(active) else 1.0
    distance /= max(scale, 1e-6)

    def objective(candidate: np.ndarray) -> float:
        covered = covariance @ candidate @ cluster_cov
        return (
            -2.0 * float(np.sum(source_to_cluster * candidate))
            + float(np.sum(covered * candidate))
            + spatial_weight * float(np.sum(np.square(distance * candidate)))
        )

    gains = _project(gains, free, anchors)
    current = objective(gains)
    for _ in range(max(0, iterations)):
        gradient = 2.0 * (
            covariance @ gains @ cluster_cov
            - source_to_cluster
            + spatial_weight * np.square(distance) * gains
        )
        gradient[~free] = 0.0
        step = 1.0
        accepted = False
        for _ in range(12):
            trial = _project(gains - step * gradient, free, anchors)
            value = objective(trial)
            if value < current - 1e-12:
                gains = trial
                current = value
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    gains[anchored_rows] = anchors[anchored_rows]
    return gains


def select_centroids_reconstruction(
    positions: np.ndarray,
    sizes: np.ndarray,
    active: np.ndarray,
    block: np.ndarray,
    importance: np.ndarray,
    content_prior: np.ndarray,
    target: int,
    fixed_indices: Sequence[int],
    *,
    width_weight: float = 0.0,
    spatial_weight: float = 0.02,
    iterations: int = 16,
    source_weighting: str = "importance",
    prev_selected: Sequence[int] = (),
    stability: float = 0.0,
    min_centroid_distance: float = 0.0,
) -> list[int]:
    """Greedy centroid selection by marginal rendered-error reduction."""
    positions = np.asarray(positions, dtype=np.float64)
    n_sources = positions.shape[0]
    if target >= n_sources:
        return list(range(n_sources))

    selected: list[int] = []
    for index in fixed_indices:
        if len(selected) >= target:
            break
        selected.append(int(index))

    fixed_set = {int(index) for index in fixed_indices}
    if min_centroid_distance > 0.0 and fixed_indices:
        fixed_positions = positions[list(fixed_indices)]
        distance_to_fixed = np.linalg.norm(
            positions[:, None, :] - fixed_positions[None, :, :],
            axis=2,
        ).min(axis=1)
        near_fixed = distance_to_fixed < min_centroid_distance
    else:
        near_fixed = np.zeros(n_sources, dtype=bool)

    for index in prev_selected:
        index = int(index)
        if index in fixed_set or near_fixed[index] or index in selected:
            continue
        if len(selected) >= target:
            break
        selected.append(index)

    active = np.asarray(active, dtype=bool)
    if not np.any(active):
        active = np.ones(n_sources, dtype=bool)

    combined_importance = np.maximum(importance, 0.0) * np.maximum(content_prior, 1e-9)
    covariance = source_covariance(block, active, combined_importance, source_weighting)
    source_pan = panning_matrix(positions)
    diagonal = np.diag(covariance)
    prev_mask = np.zeros(n_sources, dtype=bool)
    prev_mask[list(prev_selected)] = True
    prev_mask &= ~near_fixed
    candidate_mask = ((active | prev_mask) & ~near_fixed).copy()
    candidate_mask[selected] = False
    candidate_mask &= diagonal > 1e-12

    while len(selected) < target:
        if min_centroid_distance > 0.0 and selected:
            selected_positions = positions[selected]
            separation = np.linalg.norm(
                positions[:, None, :] - selected_positions[None, :, :],
                axis=2,
            ).min(axis=1)
            candidate_mask &= separation >= min_centroid_distance
        if selected:
            cluster_pan = source_pan[:, selected]
            gains = solve_gains(
                covariance,
                source_pan,
                cluster_pan,
                positions,
                positions[selected],
                active,
                selected,
                fixed_indices=fixed_indices,
                sizes=sizes,
                width_weight=width_weight,
                spatial_weight=spatial_weight,
                iterations=iterations,
            )
            residual = source_pan @ covariance - cluster_pan @ (gains.T @ covariance)
        else:
            residual = source_pan @ covariance

        delta = 2.0 * np.sum(residual * source_pan, axis=0) - diagonal
        if stability > 0.0 and np.any(prev_mask & candidate_mask):
            active_diagonal = diagonal[candidate_mask]
            floor = 0.05 * float(active_diagonal.mean()) if active_diagonal.size else 1.0
            bonus = stability * np.maximum(diagonal, floor)
            delta = np.where(prev_mask & candidate_mask, delta + bonus, delta)
        delta[~candidate_mask] = -np.inf
        best: int | None = None
        if np.any(np.isfinite(delta)):
            candidate = int(np.argmax(delta))
            if float(delta[candidate]) > 1e-12:
                best = candidate
            else:
                residual_energy = np.sum(np.square(residual), axis=0)
                residual_energy[~candidate_mask] = -np.inf
                if np.any(np.isfinite(residual_energy)):
                    best = int(np.argmax(residual_energy))
        if best is None:
            pool = [
                index
                for index in range(n_sources)
                if index not in selected and not near_fixed[index]
            ]
            if not pool:
                pool = [index for index in range(n_sources) if index not in selected]
            if not pool:
                break
            best = max(pool, key=lambda index: float(combined_importance[index]))
        selected.append(best)
        candidate_mask[best] = False

    return selected
