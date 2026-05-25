from __future__ import annotations

from dataclasses import dataclass, field

from heatflux.model.ansys_node import AnsysNode


@dataclass(slots=True)
class AnsysHeatFluxElement:
    element_id: int
    corner_nodes: list[AnsysNode]
    midside_nodes: list[AnsysNode]
    surface_area_mm2: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    xmrad: float = 0.0
    ymrad: float = 0.0
    distance_from_source_mm: float = 0.0
    grazing_angle_rad: float = 0.0
    normal_power_density_kw_mrad2: float = 0.0
    projected_power_density_w_mm2: float = 0.0
    total_power_w: float = 0.0
    metadata: dict[str, float] = field(default_factory=dict)

