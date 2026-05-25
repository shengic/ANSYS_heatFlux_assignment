from __future__ import annotations

import numpy as np


def compute_grazing_angle_rad(corner_xyz: np.ndarray, source_xyz: np.ndarray, centroid_xyz: np.ndarray) -> float:
    corner_xyz = np.asarray(corner_xyz, dtype=np.float64)
    source_xyz = np.asarray(source_xyz, dtype=np.float64)
    centroid_xyz = np.asarray(centroid_xyz, dtype=np.float64)

    if corner_xyz.shape != (4, 3):
        raise ValueError("corner_xyz must be shape (4, 3)")

    v1 = corner_xyz[0] - corner_xyz[1]
    v2 = corner_xyz[2] - corner_xyz[1]
    normal = np.cross(v1, v2)
    n_norm = np.linalg.norm(normal)
    if n_norm == 0.0:
        raise ValueError("degenerate surface normal")
    n_hat = normal / n_norm

    beam_vec = source_xyz - centroid_xyz
    b_norm = np.linalg.norm(beam_vec)
    if b_norm == 0.0:
        raise ValueError("source equals centroid")
    b_hat = beam_vec / b_norm

    cosine = np.clip(np.abs(np.dot(n_hat, b_hat)), 0.0, 1.0)
    return (np.pi / 2.0) - float(np.arccos(cosine))

