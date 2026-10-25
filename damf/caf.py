"""CAF header parser for the DAMF audio essence."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CafInfo:
    sample_rate: float
    channels: int
    bits: int
    bytes_per_packet: int
    frames_per_packet: int
    data_offset: int
    data_bytes: int

    @property
    def frames(self) -> int:
        return self.data_bytes // max(self.bytes_per_packet, 1)


def read_caf_info(path: Path) -> CafInfo:
    """Read the ``desc`` and ``data`` chunk metadata from a CAF file."""
    with path.open("rb") as f:
        if f.read(4) != b"caff":
            raise ValueError(f"not a CAF file: {path}")
        f.seek(4, os.SEEK_CUR)
        desc: dict[str, Any] | None = None
        while True:
            ctype = f.read(4)
            if len(ctype) != 4:
                break
            size = int.from_bytes(f.read(8), "big", signed=True)
            if ctype == b"desc":
                sample_rate = float(np.frombuffer(f.read(8), dtype=">f8")[0])
                f.read(4)
                flags = int.from_bytes(f.read(4), "big")
                bytes_per_packet = int.from_bytes(f.read(4), "big")
                frames_per_packet = int.from_bytes(f.read(4), "big")
                channels = int.from_bytes(f.read(4), "big")
                bits = int.from_bytes(f.read(4), "big")
                if size > 32:
                    f.seek(size - 32, os.SEEK_CUR)
                desc = {
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "bits": bits,
                    "bytes_per_packet": bytes_per_packet,
                    "frames_per_packet": frames_per_packet,
                    "flags": flags,
                }
            elif ctype == b"data":
                f.read(4)
                pos = f.tell()
                data_bytes = (path.stat().st_size - pos) if size < 0 else max(0, size - 4)
                if desc is None:
                    raise ValueError("CAF data chunk before desc")
                return CafInfo(
                    sample_rate=desc["sample_rate"],
                    channels=desc["channels"],
                    bits=desc["bits"],
                    bytes_per_packet=desc["bytes_per_packet"],
                    frames_per_packet=desc["frames_per_packet"],
                    data_offset=pos,
                    data_bytes=data_bytes,
                )
            else:
                if size < 0:
                    raise ValueError(f"unknown negative-size CAF chunk {ctype!r}")
                f.seek(size, os.SEEK_CUR)
    raise ValueError(f"CAF missing desc/data: {path}")
