from __future__ import annotations

import numpy as np
import pytest

from heatflux.math_core.coordinate_transform import map_to_mrad_batch
from heatflux.math_core.geometry import build_source_geometry


def test_point_on_beam_axis_returns_zero_mrad() -> None:
    source = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 1000.0])
    horizontal = np.array([100.0, 0.0, 0.0])
    geometry = build_source_geometry(source, target, horizontal)
    c = np.array([[0.0, 0.0, 1000.0]])
    xmrad, ymrad, _ = map_to_mrad_batch(c, source, geometry)
    assert np.isclose(xmrad[0], 0.0, atol=1e-12)
    assert np.isclose(ymrad[0], 0.0, atol=1e-12)


def test_distance_is_euclidean_norm() -> None:
    source = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 1000.0])
    horizontal = np.array([100.0, 0.0, 0.0])
    geometry = build_source_geometry(source, target, horizontal)
    c = np.array([[3.0, 4.0, 12.0]])
    _, _, dist = map_to_mrad_batch(c, source, geometry)
    assert np.isclose(dist[0], 13.0, atol=1e-12)


def test_source_at_centroid_raises() -> None:
    source = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 1000.0])
    horizontal = np.array([100.0, 0.0, 0.0])
    geometry = build_source_geometry(source, target, horizontal)
    with pytest.raises(ValueError):
        _ = map_to_mrad_batch(np.array([[0.0, 0.0, 0.0]]), source, geometry)
