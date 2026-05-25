from __future__ import annotations

from pathlib import Path

from heatflux.io.spectra_reader import read_spectra_file


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


def test_grid_dimensions_and_elements(tmp_path: Path) -> None:
    f = tmp_path / "power.dta"
    _write_spectra_fixture(f)
    result = read_spectra_file(f)

    assert result.n_row == 3
    assert result.n_col == 3
    assert len(result.nodes) == 9
    assert len(result.elements) == 4
    e0 = result.elements[0]
    # node1 top-left, node2 bottom-left, node3 top-right, node4 bottom-right
    assert (e0.nodes[0].xmrad, e0.nodes[0].ymrad) == (-1.0, 0.0)
    assert (e0.nodes[1].xmrad, e0.nodes[1].ymrad) == (-1.0, -1.0)
    assert (e0.nodes[2].xmrad, e0.nodes[2].ymrad) == (0.0, 0.0)
    assert (e0.nodes[3].xmrad, e0.nodes[3].ymrad) == (0.0, -1.0)
    assert result.peak_power_density_kw_mrad2 == 12.0


def test_progress_callback_reaches_total(tmp_path: Path) -> None:
    f = tmp_path / "power.dta"
    _write_spectra_fixture(f)
    events: list[tuple[int, int, str]] = []
    _ = read_spectra_file(f, progress_cb=lambda c, t, s: events.append((c, t, s)))
    assert events
    assert events[-1][0] == events[-1][1]
