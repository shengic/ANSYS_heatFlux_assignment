from __future__ import annotations

import numpy as np
import pytest

from heatflux.math_core.unit_conversion import kw_mrad2_to_w_mm2


def test_known_value_at_1000mm() -> None:
    out = kw_mrad2_to_w_mm2(1.0, 1000.0)
    assert np.isclose(float(out), 1000.0)


def test_inverse_square_law() -> None:
    a = float(kw_mrad2_to_w_mm2(2.0, 1000.0))
    b = float(kw_mrad2_to_w_mm2(2.0, 2000.0))
    assert np.isclose(a / b, 4.0)


def test_zero_distance_raises() -> None:
    with pytest.raises(ValueError):
        _ = kw_mrad2_to_w_mm2(1.0, 0.0)
