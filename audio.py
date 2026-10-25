"""FFmpeg-based CAF/raw conversion used by the pipeline."""

from __future__ import annotations

import subprocess
from pathlib import Path


def decode_caf_to_f32(caf: Path, raw: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(caf),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            str(raw),
        ],
        check=True,
    )


def raw_f32_to_caf(raw: Path, caf: Path, channels: int, rate: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(rate),
            "-ac",
            str(channels),
            "-i",
            str(raw),
            "-c:a",
            "pcm_s24be",
            "-f",
            "caf",
            str(caf),
        ],
        check=True,
    )
