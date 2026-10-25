"""Centroid selection, gain assignment and cluster metadata aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from atmos_cluster.loudness import LoudnessWeights
from atmos_cluster.spatial import width_distance_scale


@dataclass
class ClusterFrameMetadata:
    positions: np.ndarray
    sizes: np.ndarray
    importance: np.ndarray
    dialog: np.ndarray
    music: np.ndarray
    zones: list[str]
    active: np.ndarray


def cluster_scores(
    loud: LoudnessWeights,
    excitation: np.ndarray,
    f_cluster: np.ndarray,
) -> np.ndarray:
    total = excitation.sum(axis=0)
    if float(total.max()) <= 1e-15:
        return np.zeros(excitation.shape[0], dtype=np.float64)
    far = np.einsum("mb,mc->cb", excitation, f_cluster)
    total_term = (loud.g * total + loud.a) ** loud.alpha
    scores = loud.c * np.sum(
        total_term[None, :] - (loud.g * far + loud.a) ** loud.alpha,
        axis=1,
    )
    return np.maximum(scores, 0.0)


def select_centroids(
    loud: LoudnessWeights,
    excitation: np.ndarray,
    f_cluster: np.ndarray,
    active: np.ndarray,
    content_prior: np.ndarray,
    n_smooth: np.ndarray,
    target: int,
    fixed_indices: Sequence[int],
) -> list[int]:
    """Select centroids with the FIG. 12 loudness-accounted-for loop."""
    if target >= excitation.shape[0]:
        return list(range(excitation.shape[0]))
    selected: list[int] = []
    remaining = excitation.copy()

    for index in fixed_indices:
        if len(selected) >= target:
            break
        selected.append(index)
        remaining *= f_cluster[:, index][:, None]

    active = np.asarray(active, dtype=bool)
    while len(selected) < target:
        scores = cluster_scores(loud, remaining, f_cluster)
        scores *= np.maximum(content_prior, 1e-9)
        scores[selected] = -np.inf
        scores[~active] = -np.inf
        best = int(np.argmax(scores))

        if not np.isfinite(scores[best]) or scores[best] <= 1e-15:
            candidates = [index for index in range(excitation.shape[0]) if index not in selected]
            active_candidates = [index for index in candidates if active[index]]
            pool = active_candidates or candidates
            if not pool:
                break
            best = max(pool, key=lambda index: float(n_smooth[index] * content_prior[index]))

        selected.append(best)
        remaining *= f_cluster[:, best][:, None]

    return selected


def _simplex_weights(
    centroids: np.ndarray,
    point: np.ndarray,
    max_residual: float | None = 0.15,
) -> np.ndarray | None:
    """Non-negative power weights that place ``point`` in the centroid simplex."""
    count = centroids.shape[0]
    if count == 1:
        return np.ones(1, dtype=np.float64)
    if count == 2:
        direction = centroids[1] - centroids[0]
        denominator = float(np.dot(direction, direction))
        if denominator <= 1e-18:
            return None
        fraction = float(np.dot(point - centroids[0], direction) / denominator)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        residual = float(np.linalg.norm(centroids[0] + fraction * direction - point))
        if max_residual is not None and residual > max_residual:
            return None
        return np.array([1.0 - fraction, fraction], dtype=np.float64)
    matrix = np.vstack([centroids.T, np.ones((1, count))])
    target = np.concatenate([point, [1.0]])
    weights, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    if float(weights.min()) < -1e-6:
        return None
    if abs(float(weights.sum()) - 1.0) > 1e-4:
        return None
    residual = float(np.linalg.norm(matrix @ weights - target))
    if max_residual is not None and residual > max_residual:
        return None
    weights = np.clip(weights, 0.0, None)
    return weights / max(float(weights.sum()), 1e-12)


def _panning_gains(
    point: np.ndarray,
    centroids: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray | None:
    order = np.argsort(distance)
    for count in (2, 3, 4):
        if centroids.shape[0] < count:
            continue
        selected = order[:count]
        weights = _simplex_weights(centroids[selected], point)
        if weights is None:
            continue
        gains = np.zeros(centroids.shape[0], dtype=np.float64)
        gains[selected] = np.sqrt(weights)
        return gains
    return None


def assign_gains(
    positions: np.ndarray,
    sizes: np.ndarray,
    active: np.ndarray,
    selected: Sequence[int],
    mode: str,
    width_weight: float,
) -> np.ndarray:
    n_obj = positions.shape[0]
    n_cluster = len(selected)
    gains = np.zeros((n_obj, n_cluster), dtype=np.float64)
    if n_cluster == 0:
        return gains
    centroids = positions[list(selected)]
    distance = np.linalg.norm(positions[:, None, :] - centroids[None, :, :], axis=2)
    distance *= width_distance_scale(sizes, width_weight)[:, None]
    selected_index = {index: cluster for cluster, index in enumerate(selected)}

    for index in range(n_obj):
        if not active[index]:
            continue
        if index in selected_index:
            gains[index, selected_index[index]] = 1.0
            continue
        if mode == "nearest":
            gains[index, int(np.argmin(distance[index]))] = 1.0
            continue

        panned = None
        if mode == "pan2":
            panned = _simplex_weights(
                centroids[np.argsort(distance[index])[:2]],
                positions[index],
                max_residual=None,
            )
            if panned is not None:
                gains[index, np.argsort(distance[index])[:2]] = np.sqrt(panned)
        elif mode == "pan3":
            panned = _simplex_weights(
                centroids[np.argsort(distance[index])[:3]],
                positions[index],
            )
            if panned is not None:
                gains[index, np.argsort(distance[index])[:3]] = np.sqrt(panned)
        elif mode == "auto":
            panned = _panning_gains(
                positions[index],
                centroids,
                distance[index],
            )
            if panned is not None:
                gains[index] = panned
        else:
            raise ValueError(f"unknown assignment mode: {mode!r}")
        if not np.any(gains[index]):
            gains[index, int(np.argmin(distance[index]))] = 1.0
    return gains


def cluster_metadata(
    gains: np.ndarray,
    positions: np.ndarray,
    sizes: np.ndarray,
    n_smooth: np.ndarray,
    dialog: np.ndarray,
    music: np.ndarray,
    zones: Sequence[str],
    centroids: np.ndarray,
    source_power: np.ndarray | None = None,
    audible: np.ndarray | None = None,
) -> ClusterFrameMetadata:
    n_obj, n_cluster = gains.shape
    importance_weights = (gains * gains) * np.maximum(n_smooth, 1e-12)[:, None]
    if source_power is None:
        position_weights = importance_weights
    else:
        position_weights = (gains * gains) * np.maximum(source_power, 0.0)[:, None]
    out_pos = np.zeros((n_cluster, 3), dtype=np.float64)
    out_size = np.zeros(n_cluster, dtype=np.float64)
    out_imp = np.zeros(n_cluster, dtype=np.float64)
    out_dialog = np.full(n_cluster, -1.0)
    out_music = np.full(n_cluster, -1.0)
    out_zones: list[str] = []
    out_active = np.zeros(n_cluster, dtype=bool)

    for cluster in range(n_cluster):
        importance_weight = importance_weights[:, cluster]
        position_weight = position_weights[:, cluster]
        total = float(position_weight.sum())
        on = gains[:, cluster] > 0.01
        rendered = on & audible if audible is not None else on
        out_active[cluster] = bool(np.any(rendered))
        if total > 1e-15:
            out_pos[cluster] = (position_weight[:, None] * positions).sum(axis=0) / total
            out_size[cluster] = float((position_weight * sizes).sum() / total)
        else:
            out_pos[cluster] = centroids[cluster]
            out_size[cluster] = 0.0
        out_imp[cluster] = float((gains[:, cluster] ** 2 * n_smooth).sum())

        for source, destination in ((dialog, out_dialog), (music, out_music)):
            valid = rendered & (source >= 0.0)
            if np.any(valid):
                selected_weights = importance_weight[valid]
                destination[cluster] = float(
                    (selected_weights * source[valid]).sum()
                    / max(float(selected_weights.sum()), 1e-15)
                )
        cluster_zones = sorted({zones[index] for index in range(n_obj) if rendered[index]})
        out_zones.append(cluster_zones[0] if len(cluster_zones) == 1 else "all")

    return ClusterFrameMetadata(
        positions=out_pos,
        sizes=out_size,
        importance=out_imp,
        dialog=out_dialog,
        music=out_music,
        zones=out_zones,
        active=out_active,
    )
