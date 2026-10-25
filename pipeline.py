"""End-to-end orchestration for DAMF clustering."""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from atmos_cluster.analysis import frame_excitation
from atmos_cluster.audio import decode_caf_to_f32, raw_f32_to_caf
from atmos_cluster.config import ClusterSettings
from atmos_cluster.damf.caf import read_caf_info
from atmos_cluster.damf.frames import content_prior, sample_timelines
from atmos_cluster.damf.manifest import first_presentation, load_manifest, write_manifest
from atmos_cluster.damf.metadata import ClusterTracks, load_events, write_metadata
from atmos_cluster.damf.timeline import build_sources
from atmos_cluster.loudness import Loudness
from atmos_cluster.renderer import ClusterRenderer, FrameFeatures, RenderStats
from atmos_cluster.vad import (
    content_gain,
    content_importance,
    load_silero_vad,
    vad_probability_tracks,
)


@dataclass(frozen=True)
class ClusterResult:
    source: Path
    manifest: Path
    metadata: Path
    audio: Path
    output_dir: Path
    stem: str
    clusters: int
    sample_rate: int
    total_frames: int
    stats: RenderStats


def run_cluster(
    source: Path,
    out: Path,
    settings: ClusterSettings,
    *,
    stem: str | None = None,
    workdir: Path | None = None,
    keep_workdir: bool = False,
    progress_every: int = 500,
) -> ClusterResult:
    source = source.expanduser().resolve()
    manifest = load_manifest(source)
    presentation = first_presentation(manifest)
    source_dir = source.parent
    metadata_path = source_dir / presentation["metadata"]
    audio_path = source_dir / presentation["audio"]

    fs_meta, events = load_events(metadata_path)
    caf = read_caf_info(audio_path)
    fs = int(round(caf.sample_rate)) if caf.sample_rate else fs_meta
    if fs != fs_meta:
        print(f"warning: CAF rate {fs} != metadata sampleRate {fs_meta}; using CAF rate")

    source_set = build_sources(presentation, events)
    if not source_set.sources:
        raise ValueError("no object sources")
    n_sources = source_set.n_sources
    n_clusters = min(settings.max_objects, n_sources)

    requested_frames = caf.frames
    if settings.max_seconds:
        requested_frames = min(requested_frames, int(round(settings.max_seconds * fs)))

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="cluster_atmos_",
            dir=str(workdir) if workdir is not None else None,
        )
    )
    raw_source = temp_dir / "source.f32"
    raw_output = temp_dir / "clusters.f32"
    completed = False
    source_audio: np.memmap | None = None

    try:
        print(f"decoding {audio_path} -> {raw_source}")
        decode_caf_to_f32(audio_path, raw_source)
        actual_frames = raw_source.stat().st_size // (4 * caf.channels)
        total_frames = min(requested_frames, actual_frames)
        if total_frames <= 0:
            raise ValueError(f"no audio frames decoded from {audio_path}")

        n_frames = int(math.ceil(total_frames / settings.hop))
        tracks = sample_timelines(source_set.sources, n_frames, settings.hop, total_frames)
        priorities = content_prior(tracks, settings.dialog_boost_db)

        source_audio = np.memmap(
            raw_source,
            dtype="<f4",
            mode="r",
            shape=(actual_frames, caf.channels),
        )
        print(
            f"master: {caf.channels} ch, {n_sources} sources "
            f"({sum(source.is_fixed for source in source_set.sources)} fixed-position, "
            f"beds_as_objects={settings.beds_as_objects}), "
            f"{total_frames} samples ({total_frames / fs:.2f}s), target {settings.max_objects}"
        )

        loudness = Loudness(
            settings.model_dir,
            fs,
            settings.window_ms,
            settings.db_max,
        )

        content_gains = np.ones((n_frames, n_sources), dtype=np.float64)
        content_importance_tracks = np.zeros((n_frames, n_sources), dtype=np.float64)
        vad_probabilities: np.ndarray | None = None
        if n_clusters < n_sources and settings.use_vad:
            vad = load_silero_vad()
            vad_probabilities = vad_probability_tracks(
                vad,
                source_audio,
                source_set.channels,
                total_frames,
                n_frames,
                settings.hop,
                fs,
                settings.vad_smooth_ms,
            )
            content_importance_tracks = content_importance(
                vad_probabilities,
                threshold=settings.vad_threshold,
            )
            content_gains = content_gain(
                content_importance_tracks,
                settings.vad_gain_db,
            )
            max_gain_db = 20.0 * math.log10(float(content_gains.max()))
            print(
                f"vad: {vad.name} on {n_sources} sources, "
                f"speech p mean {float(vad_probabilities.mean()):.4f} "
                f"max {float(vad_probabilities.max()):.4f}, "
                f"max excitation gain {max_gain_db:+.2f} dB"
            )

        fixed_mask = np.array(
            [
                source.is_fixed and not settings.beds_as_objects
                for source in source_set.sources
            ],
            dtype=bool,
        )
        renderer = ClusterRenderer(
            loudness,
            settings,
            tracks.zones,
            fixed_mask,
        )
        cluster_tracks = ClusterTracks.empty(n_frames, n_clusters)

        started = time.time()
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        with raw_output.open("wb") as out_file:
            for frame in range(n_frames):
                start = frame * settings.hop
                length = min(settings.hop, total_frames - start)
                if length <= 0:
                    break
                excitation = frame_excitation(
                    loudness,
                    source_audio,
                    source_set.channels,
                    start,
                    length,
                    total_frames,
                )
                block = np.asarray(
                    source_audio[start : start + length, source_set.channels],
                    dtype=np.float64,
                )
                lfe = (
                    np.asarray(
                        source_audio[start : start + length, source_set.lfe_channel],
                        dtype=np.float64,
                    )
                    if source_set.lfe_channel is not None
                    else None
                )
                output = renderer.render(
                    excitation,
                    FrameFeatures(
                        positions=tracks.positions[frame],
                        active=tracks.active[frame],
                        sizes=tracks.sizes[frame],
                        content_prior=priorities[frame],
                        dialog=tracks.dialog[frame],
                        music=tracks.music[frame],
                        content_gain=content_gains[frame],
                        content_importance=content_importance_tracks[frame],
                        dialog_probability=(
                            vad_probabilities[frame] if vad_probabilities is not None else None
                        ),
                    ),
                    block,
                    lfe,
                )
                out_file.write(output.audio.tobytes())

                cluster_tracks.positions[frame] = output.metadata.positions
                cluster_tracks.sizes[frame] = output.metadata.sizes
                cluster_tracks.importance[frame] = output.metadata.importance
                cluster_tracks.dialog[frame] = output.metadata.dialog
                cluster_tracks.music[frame] = output.metadata.music
                cluster_tracks.active[frame] = output.metadata.active

                if progress_every and (frame % progress_every == 0 or frame + 1 == n_frames):
                    print(
                        f"  frame {frame + 1}/{n_frames}  k={output.n_clusters}/"
                        f"{settings.max_objects}  elapsed {time.time() - started:.1f}s"
                    )

        if renderer.cluster_zones:
            cluster_tracks.zones = list(renderer.cluster_zones)

        peak = 0.0
        raw_view = np.memmap(
            raw_output,
            dtype="<f4",
            mode="r+",
            shape=(total_frames, 1 + n_clusters),
        )
        for index in range(0, total_frames, 65536):
            peak = max(peak, float(np.max(np.abs(raw_view[index : index + 65536]))))
        scale = 1.0
        if settings.limit == "peak" and peak > 0.99:
            scale = 0.99 / peak
            for index in range(0, total_frames, 65536):
                raw_view[index : index + 65536] *= scale
        del raw_view
        if scale != 1.0:
            print(f"peak {peak:.3f} -> scaled by {20.0 * math.log10(scale):+.2f} dB")
        elif peak > 1.0:
            print(f"warning: peak {peak:.3f} > 1.0; use --limit peak or check source levels")

        out_base = out.expanduser().resolve()
        if out_base.suffix == ".atmos":
            out_manifest = out_base
            out_stem = out_base.name[: -len(".atmos")]
            out_dir = out_base.parent
        else:
            out_dir = out_base
            out_stem = stem or source.stem
            out_manifest = out_dir / f"{out_stem}.atmos"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_metadata = out_dir / f"{out_stem}.atmos.metadata"
        out_audio = out_dir / f"{out_stem}.atmos.audio"

        write_metadata(
            out_metadata,
            fs,
            cluster_tracks,
            settings.hop,
            source_set.lfe_channel is not None,
            interval_frames=settings.metadata_interval_frames,
        )
        write_manifest(
            out_manifest,
            out_stem,
            float(presentation.get("fps", 24)),
            n_clusters,
            source_set.lfe_channel is not None,
        )
        print(f"converting {raw_output} -> {out_audio}")
        raw_f32_to_caf(raw_output, out_audio, 1 + n_clusters, fs)

        print(
            f"wrote {out_manifest}  ({n_clusters} dynamic clusters + "
            f"{'LFE' if source_set.lfe_channel is not None else 'no LFE'}, "
            f"{total_frames / fs:.2f}s)"
        )
        stats = renderer.stats
        if stats.err_weight > 0.0:
            print(
                "spatial: importance-weighted nearest-centroid error "
                f"mean {stats.error_mean:.4f} room units, max {stats.err_max:.4f}"
            )
        if stats.effective_err_weight > 0.0:
            print(
                "spatial: importance-weighted gain-rendered error "
                f"mean {stats.effective_error_mean:.4f} room units, "
                f"max {stats.effective_err_max:.4f}"
            )
        if stats.occ_frames > 0:
            occupancy = stats.occupancy_mean
            print(
                "clusters: mean source occupancy "
                + " ".join(f"{value:.2f}" for value in occupancy)
                + f"  (total {float(occupancy.sum()):.2f})"
            )
        print(
            "note: 5.1 downmix correlation is insensitive to height; "
            "use object metadata/3D renders for quality"
        )

        completed = True
        return ClusterResult(
            source=source,
            manifest=out_manifest,
            metadata=out_metadata,
            audio=out_audio,
            output_dir=out_dir,
            stem=out_stem,
            clusters=n_clusters,
            sample_rate=fs,
            total_frames=total_frames,
            stats=stats,
        )
    finally:
        source_audio = None
        if keep_workdir:
            print(f"workdir kept: {temp_dir}")
        elif completed:
            shutil.rmtree(temp_dir, ignore_errors=True)
