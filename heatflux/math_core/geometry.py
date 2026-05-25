from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SourceGeometry:
    source: np.ndarray
    target: np.ndarray
    horizontal_point: np.ndarray
    e_x: np.ndarray
    e_y: np.ndarray
    e_z: np.ndarray

    @classmethod
    def from_points(cls, source: np.ndarray, target: np.ndarray, horizontal_point: np.ndarray) -> "SourceGeometry":
        return build_source_geometry(source=source, target=target, horizontal_point=horizontal_point)


def _normalize(v: np.ndarray, label: str) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError(f"{label} is zero-length")
    return v / norm


def build_source_geometry(source: np.ndarray, target: np.ndarray, horizontal_point: np.ndarray) -> SourceGeometry:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    horizontal_point = np.asarray(horizontal_point, dtype=np.float64)

    if source.shape != (3,) or target.shape != (3,) or horizontal_point.shape != (3,):
        raise ValueError("source, target, and horizontal_point must each be shape (3,)")

    e_z = _normalize(target - source, "target - source")
    x_raw = horizontal_point - source
    _ = _normalize(x_raw, "horizontal_point - source")
    y_raw = np.cross(e_z, x_raw)
    e_y = _normalize(y_raw, "cross(z, x_raw)")
    e_x = _normalize(np.cross(e_y, e_z), "cross(y, z)")
    return SourceGeometry(
        source=source,
        target=target,
        horizontal_point=horizontal_point,
        e_x=e_x,
        e_y=e_y,
        e_z=e_z,
    )
