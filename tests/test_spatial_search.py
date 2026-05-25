from __future__ import annotations

import numpy as np

from heatflux.math_core.spatial_search import SpectraGrid


def _grid() -> SpectraGrid:
    xb = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    yb = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    return SpectraGrid(
        x_boundaries=xb,
        y_boundaries=yb,
        n_col=3,
        n_row=3,
        x_min=-1.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
    )


def test_boundaries_and_out_of_bounds() -> None:
    g = _grid()
    assert g.find_element(-1.0, -1.0) == 1
    assert g.find_element(1.0, 1.0) == 4
    assert g.find_element(1.01, 0.0) is None


def test_batch_matches_individual_results() -> None:
    g = _grid()
    xs = np.array([-0.5, 0.5, 2.0], dtype=np.float64)
    ys = np.array([-0.5, 0.5, 0.0], dtype=np.float64)
    b = g.find_elements_batch(xs, ys)
    ind = [g.find_element(float(x), float(y)) for x, y in zip(xs, ys)]
    mapped = [int(v) if v is not None else -1 for v in ind]
    assert b.tolist() == mapped
