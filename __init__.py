"""Perceptual object clustering for Dolby Atmos DAMF masters.

The package implements the method described in US 9,805,725 B2 ("Object
Clustering for Rendering Object-Based Audio Content Based on Perceptual
Criteria") for the workspace's DAMF-to-E-AC-3 pipeline.  The result is a
normal DAMF master with fewer dynamic objects, so it can be fed to the
existing Rust encoder without changes.

The core library is deliberately independent of the external tools used for
validation.  Encoder/decoder-driven measurements live in :mod:`atmos_quality`.
Content importance uses Silero VAD through its packaged ONNX model.
Centroid selection supports the original FIG. 12 loudness loop and marginal
rendered-audio reconstruction gain (``--centroid-score reconstruction``); object
assignment supports nearest-cluster, simplex panning, and audio reconstruction.

Typical use::

    python3 -m atmos_cluster 16ch.atmos --out /tmp/16ch_c8.atmos --max-objects 8

Licensing: the vendored loudness model is BSD-2-Clause (see
``atmos_cluster/vendor/loudness_model/LICENSE``).  The rest follows the
workspace's research-only terms.
"""

from __future__ import annotations

from .config import VERSION as __version__
from .config import ClusterSettings
from .pipeline import ClusterResult, run_cluster

__all__ = ["ClusterResult", "ClusterSettings", "__version__", "run_cluster"]
