from __future__ import annotations

import numpy as np

from heatflux.math_core.geometry import SourceGeometry


def map_to_mrad_batch(
    centroids: np.ndarray, source: np.ndarray, geometry: SourceGeometry
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroids = np.asarray(centroids, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError("centroids must be shaped (N, 3)")
    if source.shape != (3,):
        raise ValueError("source must be shape (3,)")

    v = centroids - source
    r = np.linalg.norm(v, axis=1)
    if np.any(r == 0.0):
        raise ValueError("centroid matches source; cannot map to angular coordinates")
    x_proj = np.clip((v @ geometry.e_x) / r, -1.0, 1.0)
    y_proj = np.clip((v @ geometry.e_y) / r, -1.0, 1.0)
    xmrad = np.arcsin(x_proj) * 1000.0
    ymrad = np.arcsin(y_proj) * 1000.0
    return xmrad, ymrad, r

