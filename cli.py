"""Command-line entry point for DAMF object clustering."""

from __future__ import annotations

import argparse
from pathlib import Path

from atmos_cluster.config import (
    DEFAULT_DB_MAX,
    DEFAULT_HOP,
    DEFAULT_MODEL_DIR,
    DEFAULT_WINDOW_MS,
    VERSION,
    ClusterSettings,
)
from atmos_cluster.pipeline import run_cluster


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atmos-cluster",
        description=(
            "Perceptual object clustering for DAMF masters (US 9,805,725 B2). "
            "The output is a standard DAMF master with fewer dynamic objects."
        ),
        epilog=(
            "Example: python3 -m atmos_cluster 16ch.atmos "
            "--out /tmp/16ch_c8.atmos --max-objects 8"
        ),
    )
    parser.add_argument("atmos", type=Path, help="input .atmos manifest")
    parser.add_argument("--out", type=Path, required=True, help="output .atmos or directory")
    parser.add_argument("--stem", help="output stem when --out is a directory")
    parser.add_argument(
        "--max-objects",
        type=int,
        default=15,
        help="target dynamic clusters (default 15)",
    )
    parser.add_argument(
        "--analysis-hop",
        type=int,
        default=DEFAULT_HOP,
        help="samples between centroid updates",
    )
    parser.add_argument("--window-ms", type=float, default=DEFAULT_WINDOW_MS)
    parser.add_argument(
        "--db-max",
        type=float,
        default=DEFAULT_DB_MAX,
        help="SPL of a full-scale sinusoid",
    )
    parser.add_argument("--importance-smooth-s", type=float, default=0.5)
    parser.add_argument(
        "--adaptive-smooth-s",
        type=float,
        default=1.0,
        help="additional loudness smoothing for speech-like content",
    )
    parser.add_argument("--gain-smooth-ms", type=float, default=20.0)
    parser.add_argument(
        "--transient-db",
        type=float,
        default=6.0,
        help="energy rise that bypasses smoothing for one frame (0 disables)",
    )
    parser.add_argument("--unmask-fmax", type=float, default=0.995)
    parser.add_argument("--unmask-d0", type=float, default=0.5)
    parser.add_argument(
        "--width-weight",
        type=float,
        default=0.5,
        help="0 disables width-relaxed clustering",
    )
    parser.add_argument(
        "--dialog-boost-db",
        type=float,
        default=0.0,
        help="legacy DAMF dialog-metadata boost; VAD is separate",
    )
    parser.add_argument(
        "--no-vad",
        action="store_false",
        dest="use_vad",
        help="disable the Silero VAD content prior",
    )
    parser.add_argument("--vad-gain-db", type=float, default=6.0)
    parser.add_argument("--vad-threshold", type=float, default=0.1)
    parser.add_argument("--vad-smooth-ms", type=float, default=100.0)
    parser.add_argument(
        "--centroid-score",
        choices=("loudness", "reconstruction"),
        default="loudness",
        help=(
            "centroid ranking: FIG. 12 loudness or marginal rendered-audio "
            "reconstruction gain (requires --assign audio)"
        ),
    )
    parser.add_argument(
        "--assign",
        choices=("nearest", "pan2", "pan3", "auto", "audio"),
        default="auto",
        help=(
            "object-to-cluster assignment; auto uses simplex panning, "
            "audio minimizes rendered audio reconstruction error"
        ),
    )
    parser.add_argument(
        "--beds-as-objects",
        action="store_true",
        help="treat non-LFE bed channels as fixed-position inputs instead of forced centroids",
    )
    parser.add_argument(
        "--audio-spatial-weight",
        type=float,
        default=0.02,
        help="spatial regularizer for --assign audio",
    )
    parser.add_argument(
        "--audio-iterations",
        type=int,
        default=12,
        help="projected-gradient iterations for --assign audio",
    )
    parser.add_argument(
        "--audio-source-weighting",
        choices=("none", "power", "importance"),
        default="power",
        help=(
            "per-source weighting for the audio objective: none, inverse power, "
            "or importance/inverse power"
        ),
    )
    parser.add_argument(
        "--metadata-interval-frames",
        type=int,
        default=2,
        help="minimum frames between metadata keyframes (1 = every frame)",
    )
    parser.add_argument(
        "--centroid-hold-ms",
        type=float,
        default=500.0,
        help="keep a selected centroid through short source gaps",
    )
    parser.add_argument(
        "--centroid-stability",
        type=float,
        default=0.5,
        help="bonus for keeping the previous centroid set",
    )
    parser.add_argument(
        "--centroid-min-distance",
        type=float,
        default=0.1,
        help="minimum room-space separation for a new centroid",
    )
    parser.add_argument("--limit", choices=("peak", "none"), default="peak")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--progress", type=int, default=500)
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = ClusterSettings(
        max_objects=args.max_objects,
        hop=args.analysis_hop,
        window_ms=args.window_ms,
        db_max=args.db_max,
        importance_smooth_s=args.importance_smooth_s,
        adaptive_smooth_s=args.adaptive_smooth_s,
        gain_smooth_ms=args.gain_smooth_ms,
        transient_db=args.transient_db,
        unmask_fmax=args.unmask_fmax,
        unmask_d0=args.unmask_d0,
        width_weight=args.width_weight,
        dialog_boost_db=args.dialog_boost_db,
        use_vad=args.use_vad,
        vad_gain_db=args.vad_gain_db,
        vad_threshold=args.vad_threshold,
        vad_smooth_ms=args.vad_smooth_ms,
        centroid_score=args.centroid_score,
        assign=args.assign,
        beds_as_objects=args.beds_as_objects,
        audio_spatial_weight=args.audio_spatial_weight,
        audio_iterations=args.audio_iterations,
        audio_source_weighting=args.audio_source_weighting,
        metadata_interval_frames=args.metadata_interval_frames,
        centroid_hold_ms=args.centroid_hold_ms,
        centroid_stability=args.centroid_stability,
        centroid_min_distance=args.centroid_min_distance,
        limit=args.limit,
        model_dir=args.model_dir,
        max_seconds=max(0.0, args.max_seconds),
    )
    try:
        run_cluster(
            args.atmos,
            args.out,
            settings,
            stem=args.stem,
            workdir=args.workdir,
            keep_workdir=args.keep_workdir,
            progress_every=args.progress,
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        parser.error(str(exc))
    return 0
