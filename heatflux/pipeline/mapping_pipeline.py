from __future__ import annotations

from typing import Callable

import numpy as np

from heatflux.math_core.coordinate_transform import map_to_mrad_batch
from heatflux.math_core.geometry import SourceGeometry
from heatflux.math_core.grazing_angle import compute_grazing_angle_rad
from heatflux.math_core.spatial_search import SpectraGrid
from heatflux.math_core.unit_conversion import kw_mrad2_to_w_mm2
from heatflux.model.ansys_heatflux_element import AnsysHeatFluxElement
from heatflux.model.spectra_element import SpectraElement


ProgressCallback = Callable[[int, int, str], None]


def _emit_progress(progress_cb: ProgressCallback | None, current: int, total: int, stage: str) -> None:
    if progress_cb is not None:
        progress_cb(current, total, stage)


def _centroid_from_corners(element: AnsysHeatFluxElement) -> np.ndarray:
    return np.array(
        [[node.x, node.y, node.z] for node in element.corner_nodes],
        dtype=np.float64,
    ).mean(axis=0)


def _apply_physics_to_element(
    elem: AnsysHeatFluxElement,
    spectral_element: SpectraElement | None,
    source: np.ndarray,
) -> None:
    corner_xyz = np.array([[n.x, n.y, n.z] for n in elem.corner_nodes], dtype=np.float64)
    centroid_xyz = np.array([elem.x, elem.y, elem.z], dtype=np.float64)
    grazing_angle = compute_grazing_angle_rad(corner_xyz=corner_xyz, source_xyz=source, centroid_xyz=centroid_xyz)
    elem.grazing_angle_rad = float(grazing_angle)

    if spectral_element is None:
        elem.metadata["in_grid"] = 0.0
        elem.normal_power_density_kw_mrad2 = 0.0
        elem.projected_power_density_w_mm2 = 0.0
        elem.total_power_w = 0.0
        return

    elem.metadata["in_grid"] = 1.0
    normal_kw_mrad2 = spectral_element.interpolate(elem.xmrad, elem.ymrad)
    elem.normal_power_density_kw_mrad2 = float(normal_kw_mrad2)
    normal_w_mm2 = float(kw_mrad2_to_w_mm2(normal_kw_mrad2, elem.distance_from_source_mm))
    projected_w_mm2 = float(np.sin(elem.grazing_angle_rad) * normal_w_mm2)
    elem.projected_power_density_w_mm2 = projected_w_mm2
    elem.total_power_w = float(projected_w_mm2 * elem.surface_area_mm2)


def _run_mapping_sequential(
    hf_elements: list[AnsysHeatFluxElement],
    spectra_elements: list[SpectraElement],
    grid: SpectraGrid,
    source: np.ndarray,
    geometry: SourceGeometry,
    progress_cb: ProgressCallback | None,
) -> list[AnsysHeatFluxElement]:
    total = len(hf_elements)
    if total == 0:
        _emit_progress(progress_cb, 0, 0, "Mapping complete")
        return hf_elements

    centroids = np.array([_centroid_from_corners(elem) for elem in hf_elements], dtype=np.float64)
    xmrad, ymrad, distance_mm = map_to_mrad_batch(centroids=centroids, source=source, geometry=geometry)

    for idx, elem in enumerate(hf_elements):
        elem.x, elem.y, elem.z = centroids[idx]
        elem.xmrad = float(xmrad[idx])
        elem.ymrad = float(ymrad[idx])
        elem.distance_from_source_mm = float(distance_mm[idx])
        which = grid.find_element(elem.xmrad, elem.ymrad)
        spectral_element = spectra_elements[which - 1] if which is not None else None
        _apply_physics_to_element(elem=elem, spectral_element=spectral_element, source=source)
        _emit_progress(progress_cb, idx + 1, total, "Mapping elements")

    _emit_progress(progress_cb, total, total, "Mapping complete")
    return hf_elements


def _run_mapping_vectorized(
    hf_elements: list[AnsysHeatFluxElement],
    spectra_elements: list[SpectraElement],
    grid: SpectraGrid,
    source: np.ndarray,
    geometry: SourceGeometry,
    progress_cb: ProgressCallback | None,
) -> list[AnsysHeatFluxElement]:
    total = len(hf_elements)
    if total == 0:
        _emit_progress(progress_cb, 0, 0, "Mapping complete")
        return hf_elements

    centroids = np.array([_centroid_from_corners(elem) for elem in hf_elements], dtype=np.float64)
    xmrad, ymrad, distance_mm = map_to_mrad_batch(centroids=centroids, source=source, geometry=geometry)
    indices = grid.find_elements_batch(xmrad=xmrad, ymrad=ymrad)
    batch_size = 512

    processed = 0
    while processed < total:
        end = min(processed + batch_size, total)
        for idx in range(processed, end):
            elem = hf_elements[idx]
            elem.x = float(centroids[idx, 0])
            elem.y = float(centroids[idx, 1])
            elem.z = float(centroids[idx, 2])
            elem.xmrad = float(xmrad[idx])
            elem.ymrad = float(ymrad[idx])
            elem.distance_from_source_mm = float(distance_mm[idx])
            which = int(indices[idx])
            spectral_element = spectra_elements[which - 1] if which > 0 else None
            _apply_physics_to_element(elem=elem, spectral_element=spectral_element, source=source)

        processed = end
        _emit_progress(progress_cb, processed, total, "Mapping elements")

    _emit_progress(progress_cb, total, total, "Mapping complete")
    return hf_elements


def run_mapping(
    hf_elements: list[AnsysHeatFluxElement],
    spectra_elements: list[SpectraElement],
    grid: SpectraGrid,
    geometry: SourceGeometry,
    source: np.ndarray,
    progress_cb: ProgressCallback | None = None,
    vectorized: bool = True,
) -> list[AnsysHeatFluxElement]:
    """
    vectorized=True: numpy batch path.
    vectorized=False: sequential debug path.
    """
    source = np.asarray(source, dtype=np.float64)
    if source.shape != (3,):
        raise ValueError("source must be shape (3,)")
    if vectorized:
        return _run_mapping_vectorized(
            hf_elements=hf_elements,
            spectra_elements=spectra_elements,
            grid=grid,
            source=source,
            geometry=geometry,
            progress_cb=progress_cb,
        )
    return _run_mapping_sequential(
        hf_elements=hf_elements,
        spectra_elements=spectra_elements,
        grid=grid,
        source=source,
        geometry=geometry,
        progress_cb=progress_cb,
    )
