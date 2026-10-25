"""Constants and validated settings for the clustering pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

VERSION = "0.1.0"

LFE_ID = 3
OUT_ID_BASE = 10
DEFAULT_FS = 48000
DEFAULT_HOP = 1536
DEFAULT_WINDOW_MS = 64.0
DEFAULT_DB_MAX = 100.0
SILENCE_POWER = 1e-10

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PACKAGE_DIR / "vendor" / "loudness_model"
AssignMode = Literal["nearest", "pan2", "pan3", "auto", "audio"]
CentroidScoreMode = Literal["loudness", "reconstruction"]
AudioSourceWeighting = Literal["none", "power", "importance"]
LimitMode = Literal["peak", "none"]

BED_POSITIONS: dict[str, tuple[float, float, float]] = {
    "L": (-1.0, 1.0, 0.0),
    "R": (1.0, 1.0, 0.0),
    "C": (0.0, 1.0, 0.0),
    "LS": (-1.0, -1.0, 0.0),
    "RS": (1.0, -1.0, 0.0),
    "LB": (-1.0, -1.0, 0.0),
    "RB": (1.0, -1.0, 0.0),
    "LSS": (-1.0, 0.0, 0.0),
    "RSS": (1.0, 0.0, 0.0),
    "LRS": (-1.0, -1.0, 0.0),
    "RRS": (1.0, -1.0, 0.0),
    "LTS": (-1.0, 0.0, 1.0),
    "RTS": (1.0, 0.0, 1.0),
    "TFL": (-1.0, 1.0, 1.0),
    "TFR": (1.0, 1.0, 1.0),
    "TRL": (-1.0, -1.0, 1.0),
    "TRR": (1.0, -1.0, 1.0),
    "TSL": (-1.0, -1.0, 1.0),
    "TSR": (1.0, -1.0, 1.0),
}


@dataclass(frozen=True)
class ClusterSettings:
    """Algorithm parameters for one clustering run."""

    max_objects: int = 15
    hop: int = DEFAULT_HOP
    window_ms: float = DEFAULT_WINDOW_MS
    db_max: float = DEFAULT_DB_MAX
    importance_smooth_s: float = 0.5
    adaptive_smooth_s: float = 1.0
    gain_smooth_ms: float = 20.0
    transient_db: float = 6.0
    unmask_fmax: float = 0.995
    unmask_d0: float = 0.5
    width_weight: float = 0.5
    dialog_boost_db: float = 0.0
    use_vad: bool = True
    vad_gain_db: float = 6.0
    vad_threshold: float = 0.1
    vad_smooth_ms: float = 100.0
    centroid_score: CentroidScoreMode = "loudness"
    assign: AssignMode = "auto"
    beds_as_objects: bool = False
    audio_spatial_weight: float = 0.02
    audio_iterations: int = 12
    audio_source_weighting: AudioSourceWeighting = "power"
    metadata_interval_frames: int = 2
    centroid_hold_ms: float = 500.0
    centroid_stability: float = 0.5
    centroid_min_distance: float = 0.1
    limit: LimitMode = "peak"
    model_dir: Path = DEFAULT_MODEL_DIR
    max_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.max_objects <= 0:
            raise ValueError("max_objects must be positive")
        if self.hop <= 0:
            raise ValueError("hop must be positive")
        if self.window_ms <= 0:
            raise ValueError("window_ms must be positive")
        if self.importance_smooth_s <= 0:
            raise ValueError("importance_smooth_s must be positive")
        if self.adaptive_smooth_s < 0:
            raise ValueError("adaptive_smooth_s must not be negative")
        if self.gain_smooth_ms <= 0:
            raise ValueError("gain_smooth_ms must be positive")
        if self.transient_db < 0:
            raise ValueError("transient_db must not be negative")
        if self.assign not in ("nearest", "pan2", "pan3", "auto", "audio"):
            raise ValueError(f"unknown assignment mode: {self.assign!r}")
        if self.audio_spatial_weight < 0.0:
            raise ValueError("audio_spatial_weight must not be negative")
        if self.audio_iterations < 0:
            raise ValueError("audio_iterations must not be negative")
        if self.centroid_score not in ("loudness", "reconstruction"):
            raise ValueError(f"unknown centroid score mode: {self.centroid_score!r}")
        if self.centroid_score == "reconstruction" and self.assign != "audio":
            raise ValueError("--centroid-score reconstruction requires --assign audio")
        if self.audio_source_weighting not in ("none", "power", "importance"):
            raise ValueError(
                f"unknown audio source weighting: {self.audio_source_weighting!r}"
            )
        if self.metadata_interval_frames < 1:
            raise ValueError("metadata_interval_frames must be positive")
        if self.centroid_hold_ms < 0.0:
            raise ValueError("centroid_hold_ms must not be negative")
        if self.centroid_stability < 0.0:
            raise ValueError("centroid_stability must not be negative")
        if self.centroid_min_distance < 0.0:
            raise ValueError("centroid_min_distance must not be negative")
        if self.vad_gain_db < 0:
            raise ValueError("vad_gain_db must not be negative")
        if not 0.0 <= self.vad_threshold <= 1.0:
            raise ValueError("vad_threshold must be in [0, 1]")
        if self.vad_smooth_ms < 0:
            raise ValueError("vad_smooth_ms must not be negative")
        if self.limit not in ("peak", "none"):
            raise ValueError(f"unknown limiter mode: {self.limit!r}")
        if self.max_seconds < 0:
            raise ValueError("max_seconds must not be negative")
        object.__setattr__(self, "model_dir", Path(self.model_dir))
