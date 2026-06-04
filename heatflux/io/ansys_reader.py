from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

_log = logging.getLogger(__name__)

from heatflux.model.ansys_node import AnsysNode
from heatflux.model.ansys_heatflux_element import AnsysHeatFluxElement
from heatflux.model.ansys_node_store import AnsysNodeStore


ProgressCallback = Callable[[int, int, str], None]


@dataclass(slots=True)
class AnsysParseResult:
    node_store: AnsysNodeStore
    heatflux_elements: list[AnsysHeatFluxElement]
    total_elements: int


def _emit_progress(progress_cb: ProgressCallback | None, current: int, total: int, stage: str) -> None:
    if progress_cb is not None:
        progress_cb(current, total, stage)


def parse_ansys_node_line(string_line: str) -> AnsysNode:
    tokens = string_line.split()
    if len(tokens) < 4:
        raise ValueError("Invalid ANSYS node line")

    node_id = int(tokens[0])
    # NBLOCK lines often include two integer fields before XYZ.
    if len(tokens) >= 6:
        x = float(tokens[3])
        y = float(tokens[4])
        z = float(tokens[5])
    else:
        x = float(tokens[1])
        y = float(tokens[2])
        z = float(tokens[3])
    return AnsysNode(node_id=node_id, x=x, y=y, z=z)


def _quadrilateral_area_mm2(corner_nodes: list[AnsysNode]) -> float:
    p1 = np.array([corner_nodes[0].x, corner_nodes[0].y, corner_nodes[0].z], dtype=np.float64)
    p2 = np.array([corner_nodes[1].x, corner_nodes[1].y, corner_nodes[1].z], dtype=np.float64)
    p3 = np.array([corner_nodes[2].x, corner_nodes[2].y, corner_nodes[2].z], dtype=np.float64)
    p4 = np.array([corner_nodes[3].x, corner_nodes[3].y, corner_nodes[3].z], dtype=np.float64)
    v12 = p2 - p1
    v13 = p3 - p1
    v14 = p4 - p1
    return 0.5 * (np.linalg.norm(np.cross(v12, v13)) + np.linalg.norm(np.cross(v13, v14)))


def parse_ansys_heatflux_line(string_line: str, nodes_by_id: dict[int, AnsysNode]) -> AnsysHeatFluxElement:
    tokens = string_line.split()
    if len(tokens) < 13:
        raise ValueError("Invalid ANSYS heat flux line")
    element_id = int(tokens[0])
    node_ids = [int(token) for token in tokens[5:13]]

    try:
        corner_nodes = [nodes_by_id[node_id] for node_id in node_ids[:4]]
        midside_nodes = [nodes_by_id[node_id] for node_id in node_ids[4:8]]
    except KeyError as exc:
        raise ValueError(f"Heat flux line references missing node ID: {exc}") from exc
    area = _quadrilateral_area_mm2(corner_nodes)
    return AnsysHeatFluxElement(
        element_id=element_id,
        corner_nodes=corner_nodes,
        midside_nodes=midside_nodes,
        surface_area_mm2=area,
    )


def read_ansys_file(path: str | Path, progress_cb: ProgressCallback | None = None) -> AnsysParseResult:
    """Parse ANSYS APDL *.dat into node store and heat flux element list."""
    file_path = Path(path)
    file_size_mb = file_path.stat().st_size / 1_048_576
    _log.info("Parsing ANSYS file: %s (%.1f MB)", file_path, file_size_mb)
    t0 = time.monotonic()
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    total_lines = max(len(lines), 1)

    section = "nothing"
    total_elements = 0
    nodes_by_id: dict[int, AnsysNode] = {}
    heatflux_elements: list[AnsysHeatFluxElement] = []
    flux_count = 0

    emit_every_lines = 100
    last_emit_line = 0

    def emit_status(line_no: int, stage: str, force: bool = False) -> None:
        nonlocal last_emit_line
        if force or line_no == total_lines or line_no == 1 or (line_no - last_emit_line) >= emit_every_lines:
            _emit_progress(progress_cb, line_no, total_lines, stage)
            last_emit_line = line_no

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        lower = line.lower()

        if section == "node":
            stage_name = "Parsing ANSYS nodes"
        elif section == "element":
            stage_name = "Counting ANSYS elements"
        elif section == "heat flux":
            stage_name = "Parsing ANSYS heat flux elements"
        else:
            stage_name = "Scanning ANSYS sections"
        emit_status(
            line_no,
            (
                f"{stage_name} "
                f"| nodes={len(nodes_by_id):,} elements={total_elements:,} flux={flux_count:,}"
            ),
        )

        if "nodes for the whole assembly" in lower:
            _log.debug("Line %d: entering node section", line_no)
            section = "node"
            continue
        if "/com" in lower and "elements for" in lower:
            _log.debug("Line %d: entering element section", line_no)
            section = "element"
            continue
        if "create \"heat flux\"" in lower:
            _log.debug("Line %d: entering heat flux section", line_no)
            section = "heat flux"
            continue

        if line == "" or line.startswith("("):
            continue

        if line == "-1":
            section = "nothing"
            continue

        if section == "node":
            try:
                node = parse_ansys_node_line(line)
            except ValueError:
                _log.debug("Skipped node line %d: %.80r", line_no, line)
                continue
            nodes_by_id[node.node_id] = node
        elif section == "element":
            tokens = line.split()
            # EBLOCK element records can span continuation lines (e.g., 2-token tails).
            # Count only primary record lines to match ANSYS element totals.
            if tokens and tokens[0].lstrip("+-").isdigit() and len(tokens) >= 10:
                total_elements += 1
        elif section == "heat flux":
            try:
                element = parse_ansys_heatflux_line(line, nodes_by_id)
            except ValueError as exc:
                _log.warning("Skipped heat flux line %d (%s): %.80r", line_no, exc, line)
                continue
            heatflux_elements.append(element)
            flux_count += 1

    node_ids = np.array(sorted(nodes_by_id.keys()), dtype=np.int32)
    xyz = np.array([[nodes_by_id[node_id].x, nodes_by_id[node_id].y, nodes_by_id[node_id].z] for node_id in node_ids], dtype=np.float64)
    node_store = AnsysNodeStore(node_ids=node_ids, xyz=xyz)
    elapsed = time.monotonic() - t0
    n_nodes = len(nodes_by_id)
    n_flux = len(heatflux_elements)
    if n_flux == 0:
        _log.warning(
            "ANSYS parse found 0 heat flux elements in %s — check that the file contains a Heat Flux section",
            file_path.name,
        )
    if n_nodes == 0:
        _log.warning("ANSYS parse found 0 nodes in %s", file_path.name)
    _log.info(
        "ANSYS parse complete in %.2fs: nodes=%d elements=%d flux=%d",
        elapsed, n_nodes, total_elements, n_flux,
    )
    _emit_progress(
        progress_cb,
        total_lines,
        total_lines,
        (
            "ANSYS parse complete "
            f"| nodes={n_nodes:,} elements={total_elements:,} flux={n_flux:,}"
        ),
    )
    return AnsysParseResult(node_store=node_store, heatflux_elements=heatflux_elements, total_elements=total_elements)
