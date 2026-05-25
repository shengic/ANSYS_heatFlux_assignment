from __future__ import annotations

from typing import NamedTuple


class BilinearCoefficients(NamedTuple):
    a0: float
    a1: float
    a2: float
    a3: float


def solve_coefficients(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    f11: float,
    f12: float,
    f21: float,
    f22: float,
) -> BilinearCoefficients:
    denom = (x1 - x2) * (y1 - y2)
    if denom == 0.0:
        raise ValueError("degenerate bilinear cell")
    a0 = (f11 * x2 * y2 - f12 * x2 * y1 - f21 * x1 * y2 + f22 * x1 * y1) / denom
    a1 = (-f11 * y2 + f12 * y1 + f21 * y2 - f22 * y1) / denom
    a2 = (-f11 * x2 + f12 * x2 + f21 * x1 - f22 * x1) / denom
    a3 = (f11 - f12 - f21 + f22) / denom
    return BilinearCoefficients(a0, a1, a2, a3)


def interpolate(xmrad: float, ymrad: float, coeffs: BilinearCoefficients) -> float:
    return coeffs.a0 + (coeffs.a1 * xmrad) + (coeffs.a2 * ymrad) + (coeffs.a3 * xmrad * ymrad)

