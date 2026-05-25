from __future__ import annotations

from dataclasses import dataclass

from heatflux.model.spectra_node import SpectraNode


@dataclass(slots=True)
class SpectraElement:
    nodes: list[SpectraNode]
    area_mrad2: float
    a0: float
    a1: float
    a2: float
    a3: float

    def interpolate(self, xmrad: float, ymrad: float) -> float:
        return self.a0 + (self.a1 * xmrad) + (self.a2 * ymrad) + (self.a3 * xmrad * ymrad)

