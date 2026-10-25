"""Partial loudness from the vendored TVL 2018 model."""

from __future__ import annotations

import importlib.util
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Protocol

import numpy as np

from atmos_cluster.config import DEFAULT_MODEL_DIR


class LoudnessWeights(Protocol):
    g: np.ndarray
    a: np.ndarray
    alpha: np.ndarray
    c: float


class LoudnessProvider(LoudnessWeights, Protocol):
    fs: int

    def partial_loudness_objects(
        self, excitation: np.ndarray, f_pair: np.ndarray
    ) -> np.ndarray: ...


@cache
def load_model_module(model_dir: Path) -> ModuleType:
    """Load ``tvl2018`` from the vendored directory or a user-supplied tree."""
    model_dir = Path(model_dir)
    if model_dir.resolve() == DEFAULT_MODEL_DIR.resolve():
        from atmos_cluster.vendor.loudness_model import tvl2018

        return tvl2018

    module_path = model_dir / "tvl2018.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"loudness model not found: {module_path}")
    module_name = f"_atmos_cluster_tvl2018_{abs(hash(module_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load loudness model: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Loudness:
    """One-third-octave excitation and partial loudness evaluator."""

    def __init__(self, model_dir: Path, fs: int, window_ms: float, db_max: float) -> None:
        tvl = load_model_module(Path(model_dir))
        self.fs = fs
        self.db_max = float(db_max)
        self.n_fft = int(round(fs * window_ms / 1000.0))
        if self.n_fft < 256:
            raise ValueError("window too short")
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.n_bins = self.n_fft // 2 + 1
        self.freqs = np.fft.rfftfreq(self.n_fft, 1.0 / fs)

        centers = 25.0 * 2.0 ** (np.arange(0, 28) / 3.0)
        centers = centers[centers < fs / 2.0]
        low = centers / 2.0 ** (1.0 / 6.0)
        high = centers * 2.0 ** (1.0 / 6.0)
        band = np.zeros((len(centers), self.n_bins), dtype=np.float32)
        for index, (lo, hi) in enumerate(zip(low, high, strict=False)):
            bins = np.where((self.freqs >= lo) & (self.freqs < hi))[0]
            if bins.size == 0:
                bins = np.array([int(np.argmin(np.abs(self.freqs - centers[index])))])
            band[index, bins] = 1.0
        valid = band.sum(axis=1) > 0
        band = band[valid]
        self.band_centers = centers[valid]
        self.band_matrix = band
        self.d_hann_correction = 10.0 ** (3.32 / 10.0)

        self.fc = tvl.erb_number_to_frequency(np.arange(1.75, 39.25, 0.25))
        self.alpha = np.asarray(tvl.get_alpha(self.fc), dtype=np.float64)
        self.a = np.asarray(tvl.get_a(self.fc), dtype=np.float64)
        self.g = np.asarray(tvl.get_g_tvl(self.fc), dtype=np.float64)
        self.c = 0.0631
        self.pu = np.asarray(tvl.get_p(self.fc), dtype=np.float64)
        self.p1000 = float(tvl.get_p(1000.0))

    def band_levels(self, block: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(block * self.window[:, None], axis=0)
        magnitude = np.abs(spectrum) / float(self.n_fft)
        if magnitude.shape[0] > 1:
            magnitude[1:, :] *= 2.0
        intensity = (magnitude * magnitude) * self.d_hann_correction
        band_power = self.band_matrix @ intensity
        levels = 10.0 * np.log10(band_power + 1e-30) + self.db_max
        return levels.T.astype(np.float64)

    def excitation(self, levels_db: np.ndarray) -> np.ndarray:
        g = (self.band_centers[None, None, :] - self.fc[None, :, None]) / self.fc[None, :, None]
        pu = self.pu[None, :, None]
        pl = pu - 0.35 * (pu / self.p1000) * (levels_db[:, None, :] - 51.0)
        p = np.where(g >= 0.0, pu, pl)
        ag = np.abs(g)
        weight = (1.0 + p * ag) * np.exp(-p * ag) * (g <= 4.0)
        excitation = np.sum(weight * (10.0 ** (levels_db[:, None, :] / 10.0)), axis=2)
        return np.maximum(excitation, 1e-12)

    def partial_loudness_objects(self, excitation: np.ndarray, f_pair: np.ndarray) -> np.ndarray:
        """Partial loudness of every object with spatial release from masking.

        The masking term is reduced by ``f_pair`` (unmasking), but only the
        object's own excitation is added back.  The total-scene form credited
        each object with the unmasking of every other object, which made
        silent objects appear important whenever they were far from the scene.
        """
        loudness = np.zeros(excitation.shape[0], dtype=np.float64)
        for index in range(excitation.shape[0]):
            masker = np.sum(excitation * (1.0 - f_pair[index])[:, None], axis=0)
            masker = np.maximum(masker - excitation[index], 0.0)
            own = excitation[index]
            loudness[index] = self.c * np.sum(
                (self.g * (own + masker) + self.a) ** self.alpha
                - (self.g * masker + self.a) ** self.alpha
            )
        return np.maximum(loudness, 0.0)
