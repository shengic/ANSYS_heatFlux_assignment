from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from heatflux.math_core.bilinear_interpolation import solve_coefficients
from heatflux.math_core.spatial_search import SpectraGrid
from heatflux.model.spectra_element import SpectraElement
from heatflux.model.spectra_node import SpectraNode


ProgressCallback = Callable[[int, int, str], None]


@dataclass(slots=True)
class SpectraParseResult:
    nodes: list[SpectraNode]
    elements: list[SpectraElement]
    grid: SpectraGrid
    n_row: int
    n_col: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    peak_power_density_kw_mrad2: float
    total_power_kw: float


def _emit_progress(progress_cb: ProgressCallback | None, current: int, total: int, stage: str) -> None:
    if progress_cb is not None:
        progress_cb(current, total, stage)


def _build_spectra_elements(
    nodes: list[SpectraNode],
    n_row: int,
    n_col: int,
    progress_cb: ProgressCallback | None = None,
    progress_start: int = 0,
    progress_total: int = 1,
    progress_stage: str = "Building SPECTRA elements",
) -> tuple[list[SpectraElement], float]:
    elements: list[SpectraElement] = []
    total_power_kw = 0.0
    if n_row < 2 or n_col < 2:
        return elements, total_power_kw

    total_cells = (n_row - 1) * (n_col - 1)
    emit_every_cells = max(1, total_cells // 200)
    built_cells = 0

    for row in range(n_row - 1):
        for col in range(n_col - 1):
            idx_ll = row * n_col + col
            idx_lr = row * n_col + (col + 1)
            idx_ul = (row + 1) * n_col + col
            idx_ur = (row + 1) * n_col + (col + 1)

            n_ll = nodes[idx_ll]
            n_lr = nodes[idx_lr]
            n_ul = nodes[idx_ul]
            n_ur = nodes[idx_ur]

            # Enforce the documented node order:
            # node1 top-left, node2 bottom-left, node3 top-right, node4 bottom-right.
            node1 = n_ul
            node2 = n_ll
            node3 = n_ur
            node4 = n_lr

            x1 = node1.xmrad
            x2 = node3.xmrad
            y1 = node2.ymrad
            y2 = node1.ymrad

            coeffs = solve_coefficients(
                x1=x1,
                x2=x2,
                y1=y1,
                y2=y2,
                f11=node2.power_density,
                f12=node1.power_density,
                f21=node4.power_density,
                f22=node3.power_density,
            )
            area_mrad2 = abs((x2 - x1) * (y2 - y1))
            avg_pd = (node1.power_density + node2.power_density + node3.power_density + node4.power_density) / 4.0
            total_power_kw += avg_pd * area_mrad2
            elements.append(
                SpectraElement(
                    nodes=[node1, node2, node3, node4],
                    area_mrad2=area_mrad2,
                    a0=coeffs.a0,
                    a1=coeffs.a1,
                    a2=coeffs.a2,
                    a3=coeffs.a3,
                )
            )
            built_cells += 1
            if built_cells == 1 or built_cells == total_cells or (built_cells % emit_every_cells == 0):
                _emit_progress(progress_cb, progress_start + built_cells, progress_total, progress_stage)

    return elements, total_power_kw


def read_spectra_file(path: str | Path, progress_cb: ProgressCallback | None = None) -> SpectraParseResult:
    """Parse SPECTRA grid file and build interpolation elements."""
    file_path = Path(path)
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        raise ValueError("SPECTRA file is empty")

    data_end_line = total_lines
    for idx in range(2, total_lines):
        if lines[idx].strip() == "":
            data_end_line = idx
            break
    data_total = max(1, data_end_line - 2)

    nodes: list[SpectraNode] = []
    n_col = 0
    previous_y: float | None = None
    emit_every_data_lines = max(1, data_total // 300)

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if line_no <= 2:
            continue
        if line == "":
            break

        tokens = line.split()
        if len(tokens) < 3:
            continue
        xmrad = float(tokens[0])
        ymrad = float(tokens[1])
        power_density = float(tokens[2])
        node_id = len(nodes) + 1
        nodes.append(SpectraNode(node_id=node_id, xmrad=xmrad, ymrad=ymrad, power_density=power_density))
        data_idx = line_no - 2
        if data_idx == 1 or data_idx == data_total or (data_idx % emit_every_data_lines == 0):
            _emit_progress(
                progress_cb,
                data_idx,
                data_total,
                f"Reading SPECTRA rows | rows={len(nodes):,}",
            )

        if previous_y is not None and n_col == 0 and ymrad != previous_y:
            n_col = node_id - 1
        previous_y = ymrad

    if not nodes:
        raise ValueError("No SPECTRA nodes found")

    if n_col == 0:
        n_col = len(nodes)
    if len(nodes) % n_col != 0:
        raise ValueError("SPECTRA grid is not rectangular")
    n_row = len(nodes) // n_col
    total_cells = max(1, (n_row - 1) * (n_col - 1))
    progress_total = data_total + total_cells

    x_boundaries = np.array([nodes[col].xmrad for col in range(n_col)], dtype=np.float64)
    y_boundaries = np.array([nodes[row * n_col].ymrad for row in range(n_row)], dtype=np.float64)
    x_min = float(np.min(x_boundaries))
    x_max = float(np.max(x_boundaries))
    y_min = float(np.min(y_boundaries))
    y_max = float(np.max(y_boundaries))

    elements, total_power_kw = _build_spectra_elements(
        nodes,
        n_row=n_row,
        n_col=n_col,
        progress_cb=progress_cb,
        progress_start=data_total,
        progress_total=progress_total,
        progress_stage=f"Building SPECTRA elements | cells={total_cells:,}",
    )

    grid = SpectraGrid(
        x_boundaries=x_boundaries,
        y_boundaries=y_boundaries,
        n_col=n_col,
        n_row=n_row,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    peak = max(node.power_density for node in nodes)
    _emit_progress(progress_cb, progress_total, progress_total, "SPECTRA parse complete")
    return SpectraParseResult(
        nodes=nodes,
        elements=elements,
        grid=grid,
        n_row=n_row,
        n_col=n_col,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        peak_power_density_kw_mrad2=peak,
        total_power_kw=total_power_kw,
    )
