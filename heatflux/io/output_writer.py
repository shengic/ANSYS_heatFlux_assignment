from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import numpy as np

_log = logging.getLogger(__name__)

from heatflux.model.ansys_heatflux_element import AnsysHeatFluxElement
from heatflux.model.heatflux_result_store import HeatFluxResultStore


def _format_line(x: float, y: float, z: float, scaled_pd: float, raw_pd: float) -> str:
    return f"{x:.4E},{y:.4E},{z:.4E},{scaled_pd:.4E},{raw_pd:.4E}\n"


def write_output_from_elements(
    path: str | Path, elements: Iterable[AnsysHeatFluxElement], total_power_ratio: float = 1.0
) -> None:
    """Write strict 5-column ANSYS external data output from element objects."""
    target = Path(path)
    if target.exists():
        _log.warning("Output file already exists, overwriting: %s", target)
    _log.info("Writing output to: %s (ratio=%.4g)", target, total_power_ratio)
    t0 = time.monotonic()
    count = 0
    with target.open("w", encoding="utf-8", newline="") as f:
        for elem in elements:
            scaled = elem.projected_power_density_w_mm2 * total_power_ratio
            f.write(_format_line(elem.x, elem.y, elem.z, scaled, elem.projected_power_density_w_mm2))
            count += 1
    _log.info("Output written: %d lines in %.2fs", count, time.monotonic() - t0)


def write_output_from_result_store(path: str | Path, store: HeatFluxResultStore, total_power_ratio: float = 1.0) -> None:
    """Write strict 5-column ANSYS external data output from numpy-backed result store."""
    target = Path(path)
    if target.exists():
        _log.warning("Output file already exists, overwriting: %s", target)
    _log.info("Writing output (result store) to: %s (ratio=%.4g)", target, total_power_ratio)
    t0 = time.monotonic()
    output = store.to_output_array(total_power_ratio)
    with target.open("w", encoding="utf-8", newline="") as f:
        np.savetxt(f, output, fmt="%.4E", delimiter=",")
    _log.info("Output written: %d lines in %.2fs", len(output), time.monotonic() - t0)

