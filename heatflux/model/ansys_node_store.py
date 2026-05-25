from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class AnsysNodeStore:
    node_ids: np.ndarray
    xyz: np.ndarray
    _id_to_idx: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.node_ids.ndim != 1:
            raise ValueError("node_ids must be 1D")
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError("xyz must be shaped (N, 3)")
        if len(self.node_ids) != len(self.xyz):
            raise ValueError("node_ids length must match xyz rows")
        self._id_to_idx: dict[int, int] = {int(node_id): i for i, node_id in enumerate(self.node_ids)}

    def get_xyz(self, node_id: int) -> np.ndarray:
        return self.xyz[self._id_to_idx[node_id]]

    def get_xyz_batch(self, node_ids: list[int]) -> np.ndarray:
        indices = [self._id_to_idx[node_id] for node_id in node_ids]
        return self.xyz[indices]
