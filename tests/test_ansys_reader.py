from __future__ import annotations

from pathlib import Path

from heatflux.io.ansys_reader import read_ansys_file


def _write_ansys_fixture(path: Path) -> None:
    path.write_text(
        "/com,*** Nodes for the whole assembly\n"
        "NBLOCK,6,SOLID,,\n"
        "(3i9,6e21.13e3)\n"
        "        1        0        0  1.00000000000E+002  0.00000000000E+000  5.00000000000E+001\n"
        "        2        0        0  2.00000000000E+002  0.00000000000E+000  5.00000000000E+001\n"
        "        3        0        0  1.50000000000E+002  0.00000000000E+000  6.00000000000E+001\n"
        "        4        0        0  1.50000000000E+002  0.00000000000E+000  4.00000000000E+001\n"
        "        5        0        0  1.00000000000E+002  0.00000000000E+000  5.00000000000E+001\n"
        "        6        0        0  2.00000000000E+002  0.00000000000E+000  5.00000000000E+001\n"
        "        7        0        0  1.50000000000E+002  0.00000000000E+000  6.00000000000E+001\n"
        "        8        0        0  1.50000000000E+002  0.00000000000E+000  4.00000000000E+001\n"
        "-1\n"
        "/com,*** Elements for the solver\n"
        "EBLOCK,19,SOLID,,1\n"
        "(19i9)\n"
        "        1        1        1        1        0        0        0        0        8        0        1        2        3        4        5        6        7        8\n"
        "-1\n"
        "/com,*** Create \"Heat Flux\" on surface\n"
        "CMBLOCK,HeatFlux,ELEM,1\n"
        "(8i10)\n"
        "         1         0         0         0         0        1        2        3        4        5        6        7        8\n"
        "-1\n",
        encoding="utf-8",
    )


def test_full_file_minimal_fixture(tmp_path: Path) -> None:
    f = tmp_path / "model.dat"
    _write_ansys_fixture(f)
    result = read_ansys_file(f)

    assert len(result.node_store.node_ids) == 8
    assert result.total_elements == 1
    assert len(result.heatflux_elements) == 1
    elem = result.heatflux_elements[0]
    assert [n.node_id for n in elem.corner_nodes] == [1, 2, 3, 4]
    assert [n.node_id for n in elem.midside_nodes] == [5, 6, 7, 8]
    assert elem.surface_area_mm2 > 0.0


def test_progress_callback_reaches_total(tmp_path: Path) -> None:
    f = tmp_path / "model.dat"
    _write_ansys_fixture(f)
    events: list[tuple[int, int, str]] = []
    _ = read_ansys_file(f, progress_cb=lambda c, t, s: events.append((c, t, s)))
    assert events
    assert events[-1][0] == events[-1][1]
