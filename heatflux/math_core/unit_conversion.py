from __future__ import annotations

import numpy as np


def kw_mrad2_to_w_mm2(normal_power_density_kw_mrad2: np.ndarray | float, distance_mm: np.ndarray | float) -> np.ndarray:
    normal_power_density_kw_mrad2 = np.asarray(normal_power_density_kw_mrad2, dtype=np.float64)
    distance_mm = np.asarray(distance_mm, dtype=np.float64)
    if np.any(distance_mm == 0.0):
        raise ValueError("distance_mm must be non-zero")
    return normal_power_density_kw_mrad2 * (1.0e9 / (distance_mm**2))
