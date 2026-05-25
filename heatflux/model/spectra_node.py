from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SpectraNode:
    node_id: int
    xmrad: float
    ymrad: float
    power_density: float

