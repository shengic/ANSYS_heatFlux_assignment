from __future__ import annotations

from pathlib import Path

from heatflux.io.output_writer import write_output_from_elements
from heatflux.model.ansys_heatflux_element import AnsysHeatFluxElement
from heatflux.model.ansys_node import AnsysNode


def _elem() -> AnsysHeatFluxElement:
    c = [
        AnsysNode(1, 0.0, 0.0, 0.0),
        AnsysNode(2, 1.0, 0.0, 0.0),
        AnsysNode(3, 1.0, 1.0, 0.0),
        AnsysNode(4, 0.0, 1.0, 0.0),
    ]
    m = [AnsysNode(5, 0.5, 0.0, 0.0), AnsysNode(6, 1.0, 0.5, 0.0), AnsysNode(7, 0.5, 1.0, 0.0), AnsysNode(8, 0.0, 0.5, 0.0)]
    e = AnsysHeatFluxElement(element_id=1, corner_nodes=c, midside_nodes=m, surface_area_mm2=1.0)
    e.x, e.y, e.z = 1.0, 2.0, 3.0
    e.projected_power_density_w_mm2 = 4.0
    return e


def test_output_has_exactly_five_columns(tmp_path: Path) -> None:
    out = tmp_path / "out.inp"
    write_output_from_elements(out, [_elem()], total_power_ratio=1.0)
    line = out.read_text(encoding="utf-8").strip()
    assert len(line.split(",")) == 5


def test_ratio_scales_only_column4(tmp_path: Path) -> None:
    out = tmp_path / "out.inp"
    write_output_from_elements(out, [_elem()], total_power_ratio=2.5)
    c = out.read_text(encoding="utf-8").strip().split(",")
    col4 = float(c[3])
    col5 = float(c[4])
    assert col4 == 10.0
    assert col5 == 4.0
