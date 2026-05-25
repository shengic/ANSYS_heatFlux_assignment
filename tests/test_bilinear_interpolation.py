from __future__ import annotations

import numpy as np
import pytest

from heatflux.math_core.bilinear_interpolation import interpolate, solve_coefficients


def test_corner_boundary_conditions() -> None:
    coeffs = solve_coefficients(
        x1=0.0,
        x2=1.0,
        y1=0.0,
        y2=1.0,
        f11=1.0,
        f12=2.0,
        f21=3.0,
        f22=4.0,
    )
    assert np.isclose(interpolate(0.0, 0.0, coeffs), 1.0)
    assert np.isclose(interpolate(0.0, 1.0, coeffs), 2.0)
    assert np.isclose(interpolate(1.0, 0.0, coeffs), 3.0)
    assert np.isclose(interpolate(1.0, 1.0, coeffs), 4.0)


def test_uniform_field_constant_everywhere() -> None:
    coeffs = solve_coefficients(0.0, 1.0, 0.0, 1.0, 7.0, 7.0, 7.0, 7.0)
    assert np.isclose(interpolate(0.35, 0.73, coeffs), 7.0)


def test_degenerate_cell_raises() -> None:
    with pytest.raises(ValueError):
        _ = solve_coefficients(0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0)
