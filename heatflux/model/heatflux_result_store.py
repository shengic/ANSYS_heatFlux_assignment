from __future__ import annotations

import numpy as np


RESULT_DTYPE = np.dtype(
    [
        ("element_id", np.int32),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("xmrad", np.float64),
        ("ymrad", np.float64),
        ("distance_mm", np.float64),
        ("grazing_rad", np.float64),
        ("normal_pd", np.float64),
        ("projected_pd", np.float64),
        ("total_power_w", np.float64),
    ]
)


class HeatFluxResultStore:
    def __init__(self, n: int):
        self.data = np.zeros(n, dtype=RESULT_DTYPE)

    def to_output_array(self, total_power_ratio: float) -> np.ndarray:
        out = np.zeros((len(self.data), 5), dtype=np.float64)
        out[:, 0] = self.data["x"]
        out[:, 1] = self.data["y"]
        out[:, 2] = self.data["z"]
        out[:, 3] = self.data["projected_pd"] * total_power_ratio
        out[:, 4] = self.data["projected_pd"]
        return out
