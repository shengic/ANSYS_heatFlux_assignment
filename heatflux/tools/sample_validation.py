from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from heatflux.io.ansys_reader import read_ansys_file
from heatflux.io.output_writer import write_output_from_elements
from heatflux.io.spectra_reader import read_spectra_file
from heatflux.math_core.geometry import build_source_geometry
from heatflux.pipeline.mapping_pipeline import run_mapping


@dataclass(slots=True)
class ValidationResult:
    report_path: Path
    generated_output_path: Path
    exact_match_first5: bool
    lines: list[str]


def _read_first_5_columns(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
            if len(parts) < 5:
                continue
            rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return np.array(rows, dtype=np.float64)


def run_default_sample_validation(project_root: Path) -> ValidationResult:
    sample_dir = project_root / "test sample"
    ansys_path = sample_dir / "CU15-19A mask.dat"
    spectra_path = sample_dir / "CU15-19A spectra.data"
    ref_inp = sample_dir / "CU15-19A mask.inp"
    out_inp = sample_dir / "CU15-19A mask.python.inp"
    report_path = sample_dir / "sample_validation_report.txt"

    if not ansys_path.exists():
        raise FileNotFoundError(f"Missing ANSYS file: {ansys_path}")
    if not spectra_path.exists():
        raise FileNotFoundError(f"Missing SPECTRA file: {spectra_path}")
    if not ref_inp.exists():
        raise FileNotFoundError(f"Missing reference INP file: {ref_inp}")

    source = np.array([-21600.0, 0.0, 0.0], dtype=np.float64)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    horizontal = np.array([-21600.0, 5.0, 0.0], dtype=np.float64)

    lines: list[str] = []
    lines.append("=== Sample Validation Report ===")
    lines.append(f"sample_dir={sample_dir}")

    t0 = perf_counter()
    ansys = read_ansys_file(ansys_path)
    t1 = perf_counter()
    lines.append(f"ansys_parse_s={t1 - t0:.3f}")
    lines.append(f"ansys_nodes={len(ansys.node_store.node_ids)}")
    lines.append(f"ansys_total_elements={ansys.total_elements}")
    lines.append(f"ansys_flux_elements={len(ansys.heatflux_elements)}")

    t0 = perf_counter()
    spectra = read_spectra_file(spectra_path)
    t1 = perf_counter()
    lines.append(f"spectra_parse_s={t1 - t0:.3f}")
    lines.append(f"spectra_grid={spectra.n_col}x{spectra.n_row}")
    lines.append(f"spectra_peak_kw_mrad2={spectra.peak_power_density_kw_mrad2:.8f}")
    lines.append(f"spectra_total_power_kw={spectra.total_power_kw:.8f}")

    geometry = build_source_geometry(source, target, horizontal)

    t0 = perf_counter()
    mapped = run_mapping(
        hf_elements=ansys.heatflux_elements,
        spectra_elements=spectra.elements,
        grid=spectra.grid,
        geometry=geometry,
        source=source,
        vectorized=True,
    )
    t1 = perf_counter()
    lines.append(f"mapping_s={t1 - t0:.3f}")

    mapped_count = sum(1 for e in mapped if float(e.metadata.get("in_grid", 0.0)) > 0.5)
    out_count = len(mapped) - mapped_count
    total_power_out_kw = sum(e.total_power_w for e in mapped) / 1000.0
    lines.append(f"mapped_elements={mapped_count}")
    lines.append(f"out_of_grid_elements={out_count}")
    lines.append(f"total_power_out_kw={total_power_out_kw:.8f}")

    t0 = perf_counter()
    write_output_from_elements(out_inp, mapped, total_power_ratio=1.0)
    t1 = perf_counter()
    lines.append(f"write_s={t1 - t0:.3f}")
    lines.append(f"generated_output={out_inp}")

    ref = _read_first_5_columns(ref_inp)
    new = _read_first_5_columns(out_inp)
    lines.append(f"ref_shape={ref.shape}")
    lines.append(f"new_shape={new.shape}")

    exact_match = False
    if ref.shape == new.shape:
        diff = np.abs(ref - new)
        max_abs = diff.max(axis=0)
        mean_abs = diff.mean(axis=0)
        lines.append("max_abs_diff_cols5=" + ",".join(f"{v:.6e}" for v in max_abs))
        lines.append("mean_abs_diff_cols5=" + ",".join(f"{v:.6e}" for v in mean_abs))
        exact_match = bool(np.all(diff == 0.0))
        lines.append(f"exact_match_first5={exact_match}")
    else:
        lines.append("exact_match_first5=False")
        lines.append("reason=shape_mismatch")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ValidationResult(
        report_path=report_path,
        generated_output_path=out_inp,
        exact_match_first5=exact_match,
        lines=lines,
    )

