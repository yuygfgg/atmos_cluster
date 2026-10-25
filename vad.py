"""Silero VAD content-semantics prior."""

from __future__ import annotations

import importlib.metadata
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.signal import resample_poly

SILERO_RATE = 16000
SILERO_CHUNK = 512
SILERO_STATE_SHAPE = (2, 1, 128)


class VadBackend(Protocol):
    name: str
    hop_seconds: float

    def probabilities(self, signal: np.ndarray, sample_rate: int) -> np.ndarray: ...


def silero_model_path() -> Path:
    """Return the ONNX model shipped by the ``silero-vad`` dependency."""
    try:
        distribution = importlib.metadata.distribution("silero-vad")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("silero-vad is required: pip install silero-vad") from exc
    path = Path(str(distribution.locate_file("silero_vad/data/silero_vad_16k_op15.onnx")))
    if not path.is_file():
        raise FileNotFoundError(
            "silero-vad is installed but its ONNX model is missing; reinstall silero-vad"
        )
    return path


@dataclass
class SileroOnnxVad:
    model_path: Path
    name: str = "silero"
    hop_seconds: float = SILERO_CHUNK / SILERO_RATE
    session: Any = field(init=False, repr=False)
    input_name: str = field(init=False)
    state_name: str = field(init=False)
    rate_name: str = field(init=False)
    output_name: str = field(init=False)
    next_state_name: str = field(init=False)

    def __post_init__(self) -> None:
        import onnxruntime as ort

        self.model_path = Path(self.model_path)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_name = inputs[0].name
        self.state_name = inputs[1].name
        self.rate_name = inputs[2].name
        self.output_name = outputs[0].name
        self.next_state_name = outputs[1].name

    def probabilities(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != SILERO_RATE:
            divisor = math.gcd(int(sample_rate), SILERO_RATE)
            signal = resample_poly(
                np.asarray(signal, dtype=np.float32),
                SILERO_RATE // divisor,
                int(sample_rate) // divisor,
            )
        signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        if signal.size == 0:
            return np.zeros(0, dtype=np.float32)
        chunks = int(math.ceil(signal.size / SILERO_CHUNK))
        padded = np.zeros(chunks * SILERO_CHUNK, dtype=np.float32)
        padded[: signal.size] = signal
        state = np.zeros(SILERO_STATE_SHAPE, dtype=np.float32)
        context = np.zeros(64, dtype=np.float32)
        probabilities = np.zeros(chunks, dtype=np.float32)
        sample_rate_input = np.array(SILERO_RATE, dtype=np.int64)
        for index in range(chunks):
            chunk = padded[index * SILERO_CHUNK : (index + 1) * SILERO_CHUNK]
            model_input = np.concatenate([context, chunk]).reshape(1, -1)
            output, state = self.session.run(
                [self.output_name, self.next_state_name],
                {
                    self.input_name: model_input,
                    self.state_name: state,
                    self.rate_name: sample_rate_input,
                },
            )
            context = model_input[0, -64:].copy()
            probabilities[index] = float(output[0, 0])
        return np.clip(probabilities, 0.0, 1.0)


def load_silero_vad() -> SileroOnnxVad:
    return SileroOnnxVad(silero_model_path())


def vad_probability_tracks(
    backend: VadBackend,
    source: np.ndarray,
    channels: list[int],
    total_frames: int,
    n_frames: int,
    hop: int,
    sample_rate: int,
    smooth_ms: float,
) -> np.ndarray:
    """Return speech probabilities on the clustering frame grid."""
    probabilities = np.zeros((n_frames, len(channels)), dtype=np.float64)
    if n_frames <= 0 or not channels:
        return probabilities
    query = (np.arange(n_frames, dtype=np.float64) * hop + hop / 2.0) / sample_rate
    for index, channel in enumerate(channels):
        signal = np.asarray(source[:total_frames, channel], dtype=np.float32)
        chunk_probabilities = backend.probabilities(signal, sample_rate)
        if chunk_probabilities.size == 0:
            continue
        chunk_times = (
            np.arange(chunk_probabilities.size, dtype=np.float64) + 0.5
        ) * backend.hop_seconds
        probabilities[:, index] = np.interp(
            query,
            chunk_times,
            chunk_probabilities,
            left=float(chunk_probabilities[0]),
            right=float(chunk_probabilities[-1]),
        )
    if smooth_ms > 0:
        alpha = 1.0 - math.exp(-hop / (sample_rate * max(smooth_ms, 1e-6) / 1000.0))
        for frame in range(1, n_frames):
            probabilities[frame] = (1.0 - alpha) * probabilities[frame - 1] + alpha * probabilities[
                frame
            ]
    return np.clip(probabilities, 0.0, 1.0)


def content_importance(
    probability: np.ndarray,
    threshold: float = 0.1,
    slope: float = 8.0,
    bias: float = 2.0,
) -> np.ndarray:
    """Map speech probability to normalized content importance in [0, 1]."""
    raw = 1.0 / (1.0 + np.exp(-slope * np.maximum(probability - threshold, 0.0) + bias))
    floor = 1.0 / (1.0 + math.exp(bias))
    ceiling = 1.0 / (1.0 + math.exp(-slope * max(1.0 - threshold, 0.0) + bias))
    normalized = (raw - floor) / max(ceiling - floor, 1e-12)
    return np.clip(normalized, 0.0, 1.0)


def content_gain(importance: np.ndarray, gain_db: float) -> np.ndarray:
    """Linear excitation gain for normalized content importance."""
    return np.power(10.0, float(gain_db) * np.asarray(importance) / 20.0)
