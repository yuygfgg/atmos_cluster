"""Read and write the DAMF ``.atmos`` manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from atmos_cluster.config import LFE_ID, OUT_ID_BASE


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid DAMF manifest: {path}")
    return data


def first_presentation(manifest: dict[str, Any]) -> dict[str, Any]:
    presentations = manifest.get("presentations") or []
    if not presentations:
        raise ValueError("manifest has no presentations")
    return presentations[0]


def parse_bed_layout(presentation: dict[str, Any]) -> tuple[list[tuple[int, str, int]], int]:
    """Return ``(bed channels, channel count)`` in manifest order."""
    beds: list[tuple[int, str, int]] = []
    channel = 0
    for bed in presentation.get("bedInstances") or []:
        for entry in bed.get("channels") or []:
            name = str(entry.get("channel", "")).strip()
            obj_id = int(entry.get("ID", 0))
            beds.append((obj_id, name, channel))
            channel += 1
    return beds, channel


def write_manifest(
    path: Path,
    stem: str,
    fps: float,
    n_clusters: int,
    has_lfe: bool,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("version: 0.5.1\n")
        f.write("presentations:\n")
        f.write("  - type: home\n")
        f.write("    simplified: false\n")
        f.write(f"    metadata: {stem}.atmos.metadata\n")
        f.write(f"    audio: {stem}.atmos.audio\n")
        f.write("    offset: 0.0\n")
        f.write(f"    fps: {fps}\n")
        f.write(f"    scBedConfiguration: {[3] if has_lfe else []}\n")
        f.write("    creationTool: cluster_atmos.py\n")
        f.write("    creationToolVersion: 0.1.0\n")
        f.write("    bedInstances:\n")
        if has_lfe:
            f.write("      - channels:\n")
            f.write("          - channel: LFE\n")
            f.write(f"            ID: {LFE_ID}\n")
        else:
            f.write("      - channels: []\n")
        f.write("    objects:\n")
        for index in range(n_clusters):
            f.write(f"      - ID: {OUT_ID_BASE + index}\n")
