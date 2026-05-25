from __future__ import annotations

import numpy as np
import pytest

from heatflux.math_core.geometry import SourceGeometry, build_source_geometry


def test_unit_vectors_have_length_one() -> None:
    g = SourceGeometry.from_points(
        source=np.array([0.0, 0.0, 0.0]),
        target=np.array([0.0, 0.0, 1.0]),
        horizontal_point=np.array([1.0, 0.0, 0.0]),
    )
    assert np.isclose(np.linalg.norm(g.e_x), 1.0)
    assert np.isclose(np.linalg.norm(g.e_y), 1.0)
    assert np.isclose(np.linalg.norm(g.e_z), 1.0)


def test_basis_vectors_are_orthogonal() -> None:
    g = build_source_geometry(
        source=np.array([0.0, 0.0, 0.0]),
        target=np.array([0.0, 0.0, 1000.0]),
        horizontal_point=np.array([10.0, 0.0, 0.0]),
    )
    assert np.isclose(float(np.dot(g.e_x, g.e_y)), 0.0, atol=1e-12)
    assert np.isclose(float(np.dot(g.e_y, g.e_z)), 0.0, atol=1e-12)
    assert np.isclose(float(np.dot(g.e_x, g.e_z)), 0.0, atol=1e-12)


def test_degenerate_source_target_same_point() -> None:
    with pytest.raises(ValueError):
        build_source_geometry(
            source=np.array([0.0, 0.0, 0.0]),
            target=np.array([0.0, 0.0, 0.0]),
            horizontal_point=np.array([1.0, 0.0, 0.0]),
        )


def test_degenerate_side_colinear_with_axis() -> None:
    with pytest.raises(ValueError):
        build_source_geometry(
            source=np.array([0.0, 0.0, 0.0]),
            target=np.array([0.0, 0.0, 1.0]),
            horizontal_point=np.array([0.0, 0.0, 2.0]),
        )
