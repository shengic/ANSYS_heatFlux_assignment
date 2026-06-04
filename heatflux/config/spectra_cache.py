from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from heatflux.io.spectra_reader import SpectraParseResult
from heatflux.math_core.spatial_search import SpectraGrid
from heatflux.model.spectra_element import SpectraElement
from heatflux.model.spectra_node import SpectraNode

_CACHE_DIR = Path(".cache") / "spectra"


@dataclass(slots=True)
class SpectraCacheEntry:
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


def has_valid_spectra_parse_cache(source_path: Path) -> bool:
    try:
        return load_spectra_parse_cache(source_path) is not None
    except Exception:
        return False


def delete_spectra_parse_cache(source_path: Path) -> bool:
    cache_path = _cache_path_for_source(source_path.resolve())
    if cache_path.exists():
        cache_path.unlink()
        return True
    return False


def clear_all_spectra_parse_cache() -> int:
    if not _CACHE_DIR.exists():
        return 0
    count = 0
    for file in _CACHE_DIR.glob("*.npz"):
        file.unlink()
        count += 1
    return count


def delete_spectra_cache_entry_file(cache_path: Path) -> bool:
    if cache_path.exists() and cache_path.suffix.lower() == ".npz":
        cache_path.unlink()
        return True
    return False


def list_spectra_parse_cache_entries() -> list[SpectraCacheEntry]:
    if not _CACHE_DIR.exists():
        return []

    entries: list[SpectraCacheEntry] = []
    for cache_file in sorted(_CACHE_DIR.glob("*.npz")):
        try:
            with np.load(cache_file, allow_pickle=True) as data:
                source_path = Path(str(data["source_path"].item()))
                source_size = int(data["source_size"].item())
                source_mtime_ns = int(data["source_mtime_ns"].item())
        except Exception:
            entries.append(
                SpectraCacheEntry(
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
            SpectraCacheEntry(
                cache_path=cache_file,
                source_path=source_path,
                source_exists=source_exists,
                source_size=source_size,
                source_mtime_ns=source_mtime_ns,
                is_valid=is_valid,
            )
        )
    return entries


def save_spectra_parse_cache(source_path: Path, result: SpectraParseResult) -> None:
    source_path = source_path.resolve()
    st = source_path.stat()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path_for_source(source_path)

    n = len(result.nodes)
    node_ids = np.zeros(n, dtype=np.int32)
    node_xmrad = np.zeros(n, dtype=np.float64)
    node_ymrad = np.zeros(n, dtype=np.float64)
    node_power_density = np.zeros(n, dtype=np.float64)

    for i, node in enumerate(result.nodes):
        node_ids[i] = node.node_id
        node_xmrad[i] = node.xmrad
        node_ymrad[i] = node.ymrad
        node_power_density[i] = node.power_density

    elem_count = len(result.elements)
    elem_node_indices = np.zeros((elem_count, 4), dtype=np.int32)
    elem_area = np.zeros(elem_count, dtype=np.float64)
    elem_a0 = np.zeros(elem_count, dtype=np.float64)
    elem_a1 = np.zeros(elem_count, dtype=np.float64)
    elem_a2 = np.zeros(elem_count, dtype=np.float64)
    elem_a3 = np.zeros(elem_count, dtype=np.float64)

    for i, elem in enumerate(result.elements):
        elem_node_indices[i, :] = [nd.node_id - 1 for nd in elem.nodes]
        elem_area[i] = elem.area_mrad2
        elem_a0[i] = elem.a0
        elem_a1[i] = elem.a1
        elem_a2[i] = elem.a2
        elem_a3[i] = elem.a3

    np.savez_compressed(
        cache_path,
        source_path=np.array(str(source_path), dtype=object),
        source_size=np.array(st.st_size, dtype=np.int64),
        source_mtime_ns=np.array(st.st_mtime_ns, dtype=np.int64),
        n_row=np.array(result.n_row, dtype=np.int32),
        n_col=np.array(result.n_col, dtype=np.int32),
        node_ids=node_ids,
        node_xmrad=node_xmrad,
        node_ymrad=node_ymrad,
        node_power_density=node_power_density,
        elem_node_indices=elem_node_indices,
        elem_area=elem_area,
        elem_a0=elem_a0,
        elem_a1=elem_a1,
        elem_a2=elem_a2,
        elem_a3=elem_a3,
        x_boundaries=result.grid.x_boundaries,
        y_boundaries=result.grid.y_boundaries,
        peak_power_density=np.array(result.peak_power_density_kw_mrad2, dtype=np.float64),
        total_power_kw=np.array(result.total_power_kw, dtype=np.float64),
    )


def load_spectra_parse_cache(source_path: Path) -> SpectraParseResult | None:
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

        n_row = int(data["n_row"].item())
        n_col = int(data["n_col"].item())

        node_ids = data["node_ids"].astype(np.int32, copy=False)
        node_xmrad = data["node_xmrad"].astype(np.float64, copy=False)
        node_ymrad = data["node_ymrad"].astype(np.float64, copy=False)
        node_power_density = data["node_power_density"].astype(np.float64, copy=False)

        nodes: list[SpectraNode] = []
        nodes_by_id: dict[int, SpectraNode] = {}
        for i in range(len(node_ids)):
            nid = int(node_ids[i])
            node = SpectraNode(
                node_id=nid,
                xmrad=float(node_xmrad[i]),
                ymrad=float(node_ymrad[i]),
                power_density=float(node_power_density[i]),
            )
            nodes.append(node)
            nodes_by_id[nid] = node

        elem_node_indices = data["elem_node_indices"].astype(np.int32, copy=False)
        elem_area = data["elem_area"].astype(np.float64, copy=False)
        elem_a0 = data["elem_a0"].astype(np.float64, copy=False)
        elem_a1 = data["elem_a1"].astype(np.float64, copy=False)
        elem_a2 = data["elem_a2"].astype(np.float64, copy=False)
        elem_a3 = data["elem_a3"].astype(np.float64, copy=False)

        elements: list[SpectraElement] = []
        for i in range(len(elem_area)):
            indices = elem_node_indices[i]
            elem_nodes = [nodes_by_id[int(idx + 1)] for idx in indices]
            elements.append(
                SpectraElement(
                    nodes=elem_nodes,
                    area_mrad2=float(elem_area[i]),
                    a0=float(elem_a0[i]),
                    a1=float(elem_a1[i]),
                    a2=float(elem_a2[i]),
                    a3=float(elem_a3[i]),
                )
            )

        grid = SpectraGrid(
            x_boundaries=data["x_boundaries"].astype(np.float64, copy=False),
            y_boundaries=data["y_boundaries"].astype(np.float64, copy=False),
            n_col=n_col,
            n_row=n_row,
            x_min=float(np.min(data["x_boundaries"])),
            x_max=float(np.max(data["x_boundaries"])),
            y_min=float(np.min(data["y_boundaries"])),
            y_max=float(np.max(data["y_boundaries"])),
        )

        return SpectraParseResult(
            nodes=nodes,
            elements=elements,
            grid=grid,
            n_row=n_row,
            n_col=n_col,
            x_min=grid.x_min,
            x_max=grid.x_max,
            y_min=grid.y_min,
            y_max=grid.y_max,
            peak_power_density_kw_mrad2=float(data["peak_power_density"].item()),
            total_power_kw=float(data["total_power_kw"].item()),
        )
