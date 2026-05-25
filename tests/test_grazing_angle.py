from __future__ import annotations

import numpy as np
import pytest

from heatflux.math_core.grazing_angle import compute_grazing_angle_rad


def test_normal_incidence_returns_pi_over_two() -> None:
    corners = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ]
    )
    source = np.array([0.0, 0.0, 10.0])
    centroid = np.array([0.0, 0.0, 0.0])
    g = compute_grazing_angle_rad(corners, source, centroid)
    assert np.isclose(g, np.pi / 2.0, atol=1e-12)


def test_result_is_non_negative() -> None:
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    source = np.array([0.0, 0.0, 10.0])
    centroid = np.array([0.5, 0.5, 0.5])
    g = compute_grazing_angle_rad(corners, source, centroid)
    assert g >= 0.0


def test_degenerate_zero_area_element_raises() -> None:
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]
    )
    with pytest.raises(ValueError):
        _ = compute_grazing_angle_rad(corners, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
