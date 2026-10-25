"""Stateful streaming renderer for the clustering decisions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from atmos_cluster.audio_assign import audio_reconstruction_gains
from atmos_cluster.clustering import (
    ClusterFrameMetadata,
    assign_gains,
    cluster_metadata,
    select_centroids,
)
from atmos_cluster.config import SILENCE_POWER, ClusterSettings
from atmos_cluster.loudness import LoudnessProvider
from atmos_cluster.reconstruction import select_centroids_reconstruction
from atmos_cluster.spatial import clustering_unmasking, unmasking_matrix


@dataclass(frozen=True)
class FrameFeatures:
    positions: np.ndarray
    active: np.ndarray
    sizes: np.ndarray
    content_prior: np.ndarray
    dialog: np.ndarray
    music: np.ndarray
    content_gain: np.ndarray | None = None
    content_importance: np.ndarray | None = None
    dialog_probability: np.ndarray | None = None


@dataclass(frozen=True)
class ClusterFrame:
    audio: np.ndarray
    metadata: ClusterFrameMetadata
    centroids: np.ndarray
    selected: list[int]

    @property
    def n_clusters(self) -> int:
        return int(len(self.selected))


@dataclass(frozen=True)
class RenderStats:
    err_num: float
    err_weight: float
    err_max: float
    effective_err_num: float
    effective_err_weight: float
    effective_err_max: float
    occupancy: np.ndarray
    occ_frames: int

    @property
    def error_mean(self) -> float:
        return self.err_num / self.err_weight if self.err_weight > 0.0 else 0.0

    @property
    def effective_error_mean(self) -> float:
        if self.effective_err_weight <= 0.0:
            return 0.0
        return self.effective_err_num / self.effective_err_weight

    @property
    def occupancy_mean(self) -> np.ndarray:
        if self.occ_frames <= 0:
            return np.zeros_like(self.occupancy)
        return self.occupancy / self.occ_frames


class ClusterRenderer:
    """Maintains smoothing, cluster identity and gain ramps across frames."""

    def __init__(
        self,
        loudness: LoudnessProvider,
        settings: ClusterSettings,
        zones: Sequence[str],
        fixed_mask: np.ndarray,
    ) -> None:
        self.loudness = loudness
        self.settings = settings
        self.zones = list(zones)
        self.fixed_mask = np.asarray(fixed_mask, dtype=bool)
        self.fixed_indices = np.flatnonzero(self.fixed_mask).tolist()
        self.n_sources = int(self.fixed_mask.shape[0])
        if len(self.zones) != self.n_sources:
            raise ValueError("zones and fixed_mask must have the same length")
        self.target = min(settings.max_objects, self.n_sources)

        self.n_smooth = np.zeros(self.n_sources, dtype=np.float64)
        self.base_smooth_tau = max(settings.importance_smooth_s, 1e-6)
        self.gain_alpha = math.exp(
            -settings.hop / (loudness.fs * max(settings.gain_smooth_ms, 1e-6) / 1000.0)
        )

        self.prev_gains: np.ndarray | None = None
        self.prev_centroids: np.ndarray | None = None
        self.prev_energy: np.ndarray | None = None
        self.prev_selected: list[int] = []
        self.hold_frames = max(
            1,
            int(round(settings.centroid_hold_ms / 1000.0 * loudness.fs / settings.hop)),
        )
        self.selected_hold = np.zeros(self.n_sources, dtype=np.int64)
        self.cluster_zones: list[str] = []
        self.started = False

        self._err_num = 0.0
        self._err_weight = 0.0
        self._err_max = 0.0
        self._effective_err_num = 0.0
        self._effective_err_weight = 0.0
        self._effective_err_max = 0.0
        self._occupancy = np.zeros(self.target, dtype=np.float64)
        self._occ_frames = 0

    @property
    def stats(self) -> RenderStats:
        return RenderStats(
            err_num=self._err_num,
            err_weight=self._err_weight,
            err_max=self._err_max,
            effective_err_num=self._effective_err_num,
            effective_err_weight=self._effective_err_weight,
            effective_err_max=self._effective_err_max,
            occupancy=self._occupancy.copy(),
            occ_frames=self._occ_frames,
        )

    def render(
        self,
        excitation: np.ndarray,
        features: FrameFeatures,
        block: np.ndarray,
        lfe: np.ndarray | None,
    ) -> ClusterFrame:
        block_power = np.mean(np.square(block, dtype=np.float64), axis=0)
        audible = features.active & (block_power > SILENCE_POWER)
        self.selected_hold = np.maximum(self.selected_hold - 1, 0)
        if not np.any(audible):
            excitation[:] = 0.0
        else:
            excitation[~audible] = 0.0
        if features.content_gain is not None:
            excitation *= features.content_gain[:, None]

        f_pair = unmasking_matrix(
            features.positions,
            self.settings.unmask_fmax,
            self.settings.unmask_d0,
        )
        n_partial = self.loudness.partial_loudness_objects(excitation, f_pair)
        n_partial[~audible] = 0.0
        energy = excitation.sum(axis=1)
        if not self.started:
            self.n_smooth = n_partial.copy()
        else:
            importance = (
                features.content_importance
                if features.content_importance is not None
                else np.zeros(self.n_sources, dtype=np.float64)
            )
            tau = self.base_smooth_tau + self.settings.adaptive_smooth_s * np.clip(
                importance, 0.0, 1.0
            )
            alpha = 1.0 - np.exp(-self.settings.hop / (self.loudness.fs * np.maximum(tau, 1e-6)))
            if self.settings.transient_db > 0.0 and self.prev_energy is not None:
                rise_db = 10.0 * np.log10((energy + 1e-20) / (self.prev_energy + 1e-20))
                transient = (rise_db > self.settings.transient_db) & (energy > 1e-12)
                alpha = np.where(transient, 1.0, alpha)
            self.n_smooth = (1.0 - alpha) * self.n_smooth + alpha * n_partial
        self.n_smooth[~audible] = 0.0
        self.prev_energy = np.asarray(energy, dtype=np.float64).copy()

        f_cluster = clustering_unmasking(
            features.positions,
            features.sizes,
            self.settings.width_weight,
            self.settings.unmask_fmax,
            self.settings.unmask_d0,
        )
        if self.settings.centroid_score == "reconstruction":
            candidate_active = audible | (self.selected_hold > 0)
            prev_hold = [
                index
                for index in self.prev_selected
                if index < self.n_sources and candidate_active[index]
            ]
            selected = select_centroids_reconstruction(
                features.positions,
                features.sizes,
                candidate_active,
                block,
                self.n_smooth,
                features.content_prior,
                self.target,
                self.fixed_indices,
                width_weight=self.settings.width_weight,
                spatial_weight=self.settings.audio_spatial_weight,
                iterations=self.settings.audio_iterations,
                source_weighting=self.settings.audio_source_weighting,
                prev_selected=prev_hold,
                stability=self.settings.centroid_stability,
                min_centroid_distance=self.settings.centroid_min_distance,
            )
        else:
            selected = select_centroids(
                self.loudness,
                excitation,
                f_cluster,
                audible,
                features.content_prior,
                self.n_smooth,
                self.target,
                self.fixed_indices,
            )
        if not selected:
            selected = [int(np.argmax(self.n_smooth))]
        n_clusters = len(selected)
        if n_clusters != self.target:
            raise RuntimeError(f"selected {n_clusters} clusters, expected {self.target}")

        centroids = features.positions[selected]
        if self.prev_centroids is not None and len(self.prev_centroids) == n_clusters:
            cost = np.linalg.norm(centroids[:, None, :] - self.prev_centroids[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            ordered_selected: list[int | None] = [None] * n_clusters
            ordered_centroids = np.zeros_like(centroids)
            for row, col in zip(rows, cols, strict=False):
                ordered_selected[int(col)] = selected[int(row)]
                ordered_centroids[int(col)] = centroids[int(row)]
            if all(index is not None for index in ordered_selected):
                selected = [int(index) for index in ordered_selected if index is not None]
                centroids = ordered_centroids
        self.prev_centroids = centroids.copy()

        if self.settings.assign == "audio":
            gains_target = audio_reconstruction_gains(
                block,
                features.positions,
                features.sizes,
                audible,
                selected,
                centroids,
                width_weight=self.settings.width_weight,
                spatial_weight=self.settings.audio_spatial_weight,
                iterations=self.settings.audio_iterations,
                anchor_indices=(
                    self.fixed_indices
                    if self.settings.centroid_score == "reconstruction"
                    else None
                ),
                source_weighting=self.settings.audio_source_weighting,
                importance=self.n_smooth
                * np.maximum(features.content_prior, 1e-9),
            )
        else:
            gains_target = assign_gains(
                features.positions,
                features.sizes,
                features.active,
                selected,
                self.settings.assign,
                self.settings.width_weight,
            )
        if self.prev_gains is None or self.prev_gains.shape[1] != n_clusters:
            gains_smooth = gains_target.copy()
            gains_start = gains_smooth
        else:
            gains_smooth = (
                self.gain_alpha * self.prev_gains + (1.0 - self.gain_alpha) * gains_target
            )
            gains_start = self.prev_gains

        length = block.shape[0]
        ramp = (np.arange(length, dtype=np.float64) / max(length, 1))[None, None, :]
        gains_ramp = gains_start[:, :, None] + (gains_smooth - gains_start)[:, :, None] * ramp
        out_clusters = np.zeros((length, n_clusters), dtype=np.float64)
        for index in range(self.n_sources):
            out_clusters += block[:, index : index + 1] * gains_ramp[index].T

        lfe_block = lfe if lfe is not None else np.zeros(length, dtype=np.float64)
        audio = np.empty((length, 1 + n_clusters), dtype=np.float32)
        audio[:, 0] = lfe_block.astype(np.float32)
        audio[:, 1:] = out_clusters.astype(np.float32)
        self.prev_gains = gains_smooth

        dialog = (
            features.dialog_probability
            if features.dialog_probability is not None
            else features.dialog
        )
        metadata = cluster_metadata(
            gains_smooth,
            features.positions,
            features.sizes,
            self.n_smooth,
            dialog,
            features.music,
            self.zones,
            centroids,
            source_power=block_power,
            audible=audible,
        )
        for cluster, source_index in enumerate(selected):
            if self.fixed_mask[source_index]:
                metadata.positions[cluster] = features.positions[source_index]
                metadata.sizes[cluster] = 0.0
        for index in selected:
            if audible[index]:
                self.selected_hold[index] = self.hold_frames
        self.prev_selected = list(selected)
        if not self.cluster_zones:
            self.cluster_zones = list(metadata.zones)

        distance = np.linalg.norm(features.positions[:, None, :] - centroids[None, :, :], axis=2)
        dmin = distance.min(axis=1)
        weight = np.maximum(self.n_smooth, 0.0)
        self._err_num += float(np.sum(weight * dmin))
        self._err_weight += float(np.sum(weight))
        if np.any(weight > 0.0):
            self._err_max = max(self._err_max, float(np.max(dmin[weight > 0.0])))

        power = gains_smooth * gains_smooth
        power_sum = power.sum(axis=1)
        rendered = np.zeros_like(features.positions)
        valid = power_sum > 1e-15
        rendered[valid] = (power[valid] @ centroids) / power_sum[valid, None]
        effective_distance = np.linalg.norm(features.positions - rendered, axis=1)
        effective_weight = np.where(valid, np.maximum(self.n_smooth, 0.0), 0.0)
        self._effective_err_num += float(np.sum(effective_weight * effective_distance))
        self._effective_err_weight += float(np.sum(effective_weight))
        if np.any(effective_weight > 0.0):
            self._effective_err_max = max(
                self._effective_err_max,
                float(np.max(effective_distance[effective_weight > 0.0])),
            )
        self._occupancy += (gains_smooth > 0.01).sum(axis=0)
        self._occ_frames += 1
        self.started = True

        return ClusterFrame(
            audio=audio,
            metadata=metadata,
            centroids=centroids,
            selected=selected,
        )
