from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from heatflux.io.ansys_reader import AnsysParseResult
from heatflux.model.ansys_heatflux_element import AnsysHeatFluxElement
from heatflux.model.ansys_node import AnsysNode
from heatflux.model.ansys_node_store import AnsysNodeStore


_CACHE_DIR = Path(".cache") / "ansys"


@dataclass(slots=True)
class AnsysCacheEntry:
    cache_path: Path
    source_path: Path
    source_exists: bool
    source_size: int
    source_mtime_ns: int
    is_valid: bool


def _cache_path_for_source(source_path: Path) -> Path:
    resolved = str(source_path.resolve())
    key = sha256(resolved.encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{key}.npz"


def has_valid_ansys_parse_cache(source_path: Path) -> bool:
    try:
        return load_ansys_parse_cache(source_path) is not None
    except Exception:
        return False


def delete_ansys_parse_cache(source_path: Path) -> bool:
    cache_path = _cache_path_for_source(source_path.resolve())
    if cache_path.exists():
        cache_path.unlink()
        return True
    return False


def clear_all_ansys_parse_cache() -> int:
    if not _CACHE_DIR.exists():
        return 0
    count = 0
    for file in _CACHE_DIR.glob("*.npz"):
        file.unlink()
        count += 1
    return count


def delete_ansys_cache_entry_file(cache_path: Path) -> bool:
    if cache_path.exists() and cache_path.suffix.lower() == ".npz":
        cache_path.unlink()
        return True
    return False


def list_ansys_parse_cache_entries() -> list[AnsysCacheEntry]:
    if not _CACHE_DIR.exists():
        return []

    entries: list[AnsysCacheEntry] = []
    for cache_file in sorted(_CACHE_DIR.glob("*.npz")):
        try:
            with np.load(cache_file, allow_pickle=True) as data:
                source_path = Path(str(data["source_path"].item()))
                source_size = int(data["source_size"].item())
                source_mtime_ns = int(data["source_mtime_ns"].item())
        except Exception:
            entries.append(
                AnsysCacheEntry(
                    cache_path=cache_file,
                    source_path=Path("<corrupt>"),
                    source_exists=False,
                    source_size=0,
                    source_mtime_ns=0,
                    is_valid=False,
                )
            )
            continue

        source_exists = source_path.exists()
        is_valid = False
        if source_exists:
            try:
                st = source_path.stat()
                is_valid = st.st_size == source_size and st.st_mtime_ns == source_mtime_ns
            except Exception:
                is_valid = False

        entries.append(
            AnsysCacheEntry(
                cache_path=cache_file,
                source_path=source_path,
                source_exists=source_exists,
                source_size=source_size,
                source_mtime_ns=source_mtime_ns,
                is_valid=is_valid,
            )
        )
    return entries


def _quadrilateral_area_mm2(corner_nodes: list[AnsysNode]) -> float:
    p1 = np.array([corner_nodes[0].x, corner_nodes[0].y, corner_nodes[0].z], dtype=np.float64)
    p2 = np.array([corner_nodes[1].x, corner_nodes[1].y, corner_nodes[1].z], dtype=np.float64)
    p3 = np.array([corner_nodes[2].x, corner_nodes[2].y, corner_nodes[2].z], dtype=np.float64)
    p4 = np.array([corner_nodes[3].x, corner_nodes[3].y, corner_nodes[3].z], dtype=np.float64)
    v12 = p2 - p1
    v13 = p3 - p1
    v14 = p4 - p1
    return 0.5 * (np.linalg.norm(np.cross(v12, v13)) + np.linalg.norm(np.cross(v13, v14)))


def save_ansys_parse_cache(source_path: Path, result: AnsysParseResult) -> None:
    source_path = source_path.resolve()
    st = source_path.stat()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path_for_source(source_path)

    hf_count = len(result.heatflux_elements)
    hf_element_ids = np.zeros(hf_count, dtype=np.int32)
    hf_corner_node_ids = np.zeros((hf_count, 4), dtype=np.int32)
    hf_midside_node_ids = np.zeros((hf_count, 4), dtype=np.int32)

    for i, elem in enumerate(result.heatflux_elements):
        hf_element_ids[i] = elem.element_id
        hf_corner_node_ids[i, :] = [node.node_id for node in elem.corner_nodes]
        hf_midside_node_ids[i, :] = [node.node_id for node in elem.midside_nodes]

    np.savez_compressed(
        cache_path,
        source_path=np.array(str(source_path), dtype=object),
        source_size=np.array(st.st_size, dtype=np.int64),
        source_mtime_ns=np.array(st.st_mtime_ns, dtype=np.int64),
        node_ids=result.node_store.node_ids.astype(np.int32, copy=False),
        xyz=result.node_store.xyz.astype(np.float64, copy=False),
        total_elements=np.array(result.total_elements, dtype=np.int64),
        hf_element_ids=hf_element_ids,
        hf_corner_node_ids=hf_corner_node_ids,
        hf_midside_node_ids=hf_midside_node_ids,
    )


def load_ansys_parse_cache(source_path: Path) -> AnsysParseResult | None:
    source_path = source_path.resolve()
    cache_path = _cache_path_for_source(source_path)
    if not cache_path.exists():
        return None

    st = source_path.stat()
    with np.load(cache_path, allow_pickle=True) as data:
        cached_path = Path(str(data["source_path"].item()))
        if cached_path != source_path:
            return None
        cached_size = int(data["source_size"].item())
        cached_mtime_ns = int(data["source_mtime_ns"].item())
        if cached_size != st.st_size or cached_mtime_ns != st.st_mtime_ns:
            return None

        node_ids = data["node_ids"].astype(np.int32, copy=False)
        xyz = data["xyz"].astype(np.float64, copy=False)
        total_elements = int(data["total_elements"].item())
        hf_element_ids = data["hf_element_ids"].astype(np.int32, copy=False)
        hf_corner_node_ids = data["hf_corner_node_ids"].astype(np.int32, copy=False)
        hf_midside_node_ids = data["hf_midside_node_ids"].astype(np.int32, copy=False)

    node_store = AnsysNodeStore(node_ids=node_ids, xyz=xyz)
    nodes_by_id: dict[int, AnsysNode] = {}
    for i, node_id in enumerate(node_ids):
        nodes_by_id[int(node_id)] = AnsysNode(
            node_id=int(node_id),
            x=float(xyz[i, 0]),
            y=float(xyz[i, 1]),
            z=float(xyz[i, 2]),
        )

    heatflux_elements: list[AnsysHeatFluxElement] = []
    for i, element_id in enumerate(hf_element_ids):
        corner_nodes = [nodes_by_id[int(node_id)] for node_id in hf_corner_node_ids[i]]
        midside_nodes = [nodes_by_id[int(node_id)] for node_id in hf_midside_node_ids[i]]
        heatflux_elements.append(
            AnsysHeatFluxElement(
                element_id=int(element_id),
                corner_nodes=corner_nodes,
                midside_nodes=midside_nodes,
                surface_area_mm2=_quadrilateral_area_mm2(corner_nodes),
            )
        )

    return AnsysParseResult(
        node_store=node_store,
        heatflux_elements=heatflux_elements,
        total_elements=total_elements,
    )
