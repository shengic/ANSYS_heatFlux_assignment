from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SpectraGrid:
    x_boundaries: np.ndarray
    y_boundaries: np.ndarray
    n_col: int
    n_row: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def find_element(self, xmrad: float, ymrad: float) -> int | None:
        if xmrad < self.x_min or xmrad > self.x_max:
            return None
        if ymrad < self.y_min or ymrad > self.y_max:
            return None
        xi = min(np.searchsorted(self.x_boundaries, xmrad, side="right") - 1, self.n_col - 2)
        yi = min(np.searchsorted(self.y_boundaries, ymrad, side="right") - 1, self.n_row - 2)
        return int(yi * (self.n_col - 1) + xi + 1)

    def find_elements_batch(self, xmrad: np.ndarray, ymrad: np.ndarray) -> np.ndarray:
        xi = np.searchsorted(self.x_boundaries, xmrad, side="right") - 1
        yi = np.searchsorted(self.y_boundaries, ymrad, side="right") - 1
        xi = np.clip(xi, 0, self.n_col - 2)
        yi = np.clip(yi, 0, self.n_row - 2)
        out = (xmrad < self.x_min) | (xmrad > self.x_max) | (ymrad < self.y_min) | (ymrad > self.y_max)
        indices = yi * (self.n_col - 1) + xi + 1
        indices = indices.astype(np.int64, copy=False)
        indices[out] = -1
        return indices

