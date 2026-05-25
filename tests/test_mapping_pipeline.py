from __future__ import annotations

from pathlib import Path

import numpy as np

from heatflux.io.spectra_reader import read_spectra_file
from heatflux.math_core.geometry import build_source_geometry
from heatflux.model.ansys_heatflux_element import AnsysHeatFluxElement
from heatflux.model.ansys_node import AnsysNode
from heatflux.pipeline.mapping_pipeline import run_mapping


def _write_spectra_fixture(path: Path) -> None:
    path.write_text(
        "# Power density distribution\n"
        "# Units: kW/mrad^2\n"
        " -1.0  -1.0   5.00000E+00\n"
        "  0.0  -1.0   8.00000E+00\n"
        "  1.0  -1.0   5.00000E+00\n"
        " -1.0   0.0   8.00000E+00\n"
        "  0.0   0.0  1.20000E+01\n"
        "  1.0   0.0   8.00000E+00\n"
        " -1.0   1.0   5.00000E+00\n"
        "  0.0   1.0   8.00000E+00\n"
        "  1.0   1.0   5.00000E+00\n",
        encoding="utf-8",
    )


def _make_hf_element(element_id: int, cx: float, cy: float, cz: float, half: float = 0.2) -> AnsysHeatFluxElement:
    corners = [
        AnsysNode(1000 + element_id * 10 + 1, cx - half, cy - half, cz),
        AnsysNode(1000 + element_id * 10 + 2, cx + half, cy - half, cz),
        AnsysNode(1000 + element_id * 10 + 3, cx + half, cy + half, cz),
        AnsysNode(1000 + element_id * 10 + 4, cx - half, cy + half, cz),
    ]
    midside = [
        AnsysNode(1000 + element_id * 10 + 5, cx, cy - half, cz),
        AnsysNode(1000 + element_id * 10 + 6, cx + half, cy, cz),
        AnsysNode(1000 + element_id * 10 + 7, cx, cy + half, cz),
        AnsysNode(1000 + element_id * 10 + 8, cx - half, cy, cz),
    ]
    return AnsysHeatFluxElement(
        element_id=element_id,
        corner_nodes=corners,
        midside_nodes=midside,
        surface_area_mm2=(2.0 * half) * (2.0 * half),
    )


def test_vectorized_matches_sequential(tmp_path: Path) -> None:
    f = tmp_path / "power.dta"
    _write_spectra_fixture(f)
    spectra = read_spectra_file(f)

    source = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    target = np.array([0.0, 0.0, 1000.0], dtype=np.float64)
    horizontal = np.array([100.0, 0.0, 0.0], dtype=np.float64)
    geometry = build_source_geometry(source=source, target=target, horizontal_point=horizontal)

    hf1 = [_make_hf_element(1, 0.2, 0.2, 1000.0), _make_hf_element(2, -0.3, 0.1, 1000.0)]
    hf2 = [_make_hf_element(1, 0.2, 0.2, 1000.0), _make_hf_element(2, -0.3, 0.1, 1000.0)]

    out_v = run_mapping(hf1, spectra.elements, spectra.grid, geometry, source, vectorized=True)
    out_s = run_mapping(hf2, spectra.elements, spectra.grid, geometry, source, vectorized=False)

    for ev, es in zip(out_v, out_s):
        assert np.isclose(ev.xmrad, es.xmrad, atol=1e-12)
        assert np.isclose(ev.ymrad, es.ymrad, atol=1e-12)
        assert np.isclose(ev.projected_power_density_w_mm2, es.projected_power_density_w_mm2, atol=1e-12)
        assert np.isclose(ev.total_power_w, es.total_power_w, atol=1e-12)


def test_progress_callback_reaches_total(tmp_path: Path) -> None:
    f = tmp_path / "power.dta"
    _write_spectra_fixture(f)
    spectra = read_spectra_file(f)

    source = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    target = np.array([0.0, 0.0, 1000.0], dtype=np.float64)
    horizontal = np.array([100.0, 0.0, 0.0], dtype=np.float64)
    geometry = build_source_geometry(source=source, target=target, horizontal_point=horizontal)
    hf = [_make_hf_element(1, 0.2, 0.2, 1000.0), _make_hf_element(2, 10.0, 10.0, 1000.0)]

    events: list[tuple[int, int, str]] = []
    _ = run_mapping(
        hf_elements=hf,
        spectra_elements=spectra.elements,
        grid=spectra.grid,
        geometry=geometry,
        source=source,
        progress_cb=lambda c, t, s: events.append((c, t, s)),
        vectorized=True,
    )
    assert events
    assert events[-1][0] == events[-1][1]
