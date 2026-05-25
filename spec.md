# SPEC.md - Insertion-Device Python/Tk Rewrite Contract

Last updated: 2026-05-25
Project root: `K:\ANSYS heat flux assignment`

## 1. Goal

Rewrite the existing Excel/VBA (`.xlsm`) workflow into Python + Tkinter with equivalent behavior, improved reliability, and explicit progress feedback for long-running file parsing and mapping.

Primary function:
- Map SPECTRA power density onto ANSYS heat-flux elements.
- Export ANSYS External Data `*.inp` with strict 5-column format.

## 2. Required Inputs/Outputs

Inputs:
- ANSYS APDL file: `*.dat`
- SPECTRA file: `*.dta` / `*.data` / `*.dta2`
- Geometry points (mm):
  - `Source S = (Sx, Sy, Sz)`
  - `Target T = (Tx, Ty, Tz)`
  - `Horizontal point H = (Hx, Hy, Hz)`
- Output options:
  - `output_folder`
  - `output_filename` (default: `EPU66-27A power for ansys.inp`)
  - `total_power_ratio` (float, default `1.0`)

Output:
- ANSYS External Data file (`*.inp`), 5 columns per line:
  - `x, y, z, projected_pd_scaled, projected_pd_unscaled`
- Formatting: scientific notation `"{:.4E}"`.
- `total_power_ratio` scales only column 4.

## 3. Geometry Contract (Critical)

The local coordinate system must be derived from the three entered 3D points:

1. Beam vector (`z` axis):
- `z_hat = normalize(T - S)`

2. Horizontal reference (`x` axis):
- `x_raw = H - S`
- `x_hat` is derived to be orthogonal to `z_hat`:
  - `y_hat = normalize(cross(z_hat, x_raw))`
  - `x_hat = normalize(cross(y_hat, z_hat))`

3. Result:
- Right-handed orthonormal frame (`x_hat`, `y_hat`, `z_hat`) aligned with beam direction.

Validation rules:
- Reject if `|T-S| == 0`.
- Reject if `H-S` is zero.
- Reject if `cross(z_hat, x_raw)` norm is near zero (horizontal point nearly collinear with beam).
- Show explicit UI error with required correction.

## 4. Math + Mapping Contract

For each ANSYS heat-flux element:
- Compute centroid `(x, y, z)` from 4 corner nodes.
- `v = centroid - S`
- `R_mm = norm(v)`
- `xmrad = asin(dot(v, x_hat) / R_mm) * 1000`
- `ymrad = asin(dot(v, y_hat) / R_mm) * 1000`

SPECTRA lookup:
- Locate grid cell by `xmrad, ymrad` using `searchsorted`.
- Cell index is 1-based (to match legacy convention).
- Outside-grid elements must produce zero projected power density.

Interpolation + physics:
- Bilinear interpolate normal power density (kW/mrad^2).
- Convert to W/mm^2 with distance term.
- Compute grazing angle from element normal and beam incidence.
- Projected density = `sin(grazing_angle) * normal_density`.
- Total element power = projected density * element area.

## 5. UI Specification (Tkinter)

Reference mockup:
- `UI/screendump UI.png`

Main layout:
- Two-column dashboard.

Left column cards:
1. `ANSYS APDL INPUT FILE`
- File path field
- `Upload ANSYS file` button
- Parse status text + ANSYS parse progress bar (%)
- Stats:
  - total nodes
  - total elements
  - flux elements

2. `SPECTRA POWER DENSITY FILE`
- File path field
- `Upload SPECTRA file` button
- Parse status text + SPECTRA parse progress bar (%)
- Stats:
  - columns / rows
  - x range / y range (mrad)
  - peak power density
  - total power

3. `SOURCE GEOMETRY (MM)`
- 3 rows: Source (S), Target (T), Horizontal point (H)
- Columns: X, Y, Z
- `Update geometry` action (button or auto-validate on edit)

Right column cards:
1. `OUTPUT - ANSYS EXTERNAL DATA FILE`
- Output folder field + browse
- Output filename field
- Power ratio field

2. `MAPPING PROGRESS`
- Mapping progress bar (%)
- Output elements
- Total power out
- Total power (SPECTRA)

3. Actions
- Primary: `Create heat flux file`
- Secondary: `View file location`
- Danger: `Exit Session`
Notes:
- `View file location` opens output file in Explorer when available; falls back to parent folder.

## 6. Progress Bar Requirements

Two upload/parse progress bars are mandatory:

1. ANSYS progress bar
- Active during ANSYS file parsing.
- Must emit incremental updates (not only 0% and 100%).
- Recommended parse phases:
  - detect sections
  - parse nodes
  - parse elements
  - parse heatflux set
  - finalize structures

2. SPECTRA progress bar
- Active during SPECTRA file parsing.
- Must emit incremental updates while reading rows and building grid/elements.

Separate mapping progress:
- Active only during mapping/export pipeline.

Implementation contract:
- Readers accept optional callback:
  - `progress_cb(current: int, total: int, stage: str) -> None`
- UI adapts callback to percent and status text.

## 6.1 ANSYS Parse Cache (Performance)

Because ANSYS `.dat` can be very large, add a reusable parse cache:

- Cache scope:
  - ANSYS parsed model only (node store, heat flux connectivity, element count).
- Cache validity:
  - Same absolute source path.
  - Same source file size and mtime (ns).
- UI behavior on ANSYS upload:
  1. If valid cache exists, prompt user:
     - "A cached parsed ANSYS model exists for this file. Do you want to reuse it for faster loading?"
  2. If user chooses reuse:
     - Load cache and update ANSYS progress/status as completed.
  3. If user chooses re-parse:
     - Parse normally with ANSYS progress updates.
     - Refresh cache after successful parse.
- Failure policy:
  - Cache read/write failure must not block normal parse flow.
- Future enhancement:
  - Add richer cache browser UI if needed.
Implemented:
- Delete cache for current ANSYS file.
- Clear all ANSYS cache entries.

## 7. UI State Machine

State `IDLE`
- Enabled: both upload buttons, geometry fields, output settings.
- Disabled: create file, view location.

State `ANSYS_LOADING`
- Disable ANSYS upload button and create action.
- Show ANSYS progress.

State `SPECTRA_LOADING`
- Disable SPECTRA upload button and create action.
- Show SPECTRA progress.

State `READY_TO_MAP`
- Condition:
  - ANSYS loaded
  - SPECTRA loaded
  - geometry valid
  - output folder/filename valid
  - power ratio parseable float
- Enable `Create heat flux file`.

State `MAPPING`
- Disable all upload/geometry/output edits that would invalidate run.
- Show mapping progress and current stage.

State `DONE`
- Mapping finished successfully.
- Enable `View file location`.
- Keep `Create heat flux file` enabled for rerun.

State `ERROR`
- Show recoverable error dialog/status.
- Return to previous stable state with relevant controls re-enabled.

## 8. Module Plan (Python package target)

```
heatflux/
  gui/
    app_window.py
    geometry_frame.py
    progress_dialog.py
  io/
    ansys_reader.py
    spectra_reader.py
    output_writer.py
  model/
    ansys_node.py
    ansys_node_store.py
    ansys_heatflux_element.py
    spectra_node.py
    spectra_element.py
    heatflux_result_store.py
  math_core/
    geometry.py
    coordinate_transform.py
    bilinear_interpolation.py
    spatial_search.py
    grazing_angle.py
    unit_conversion.py
  pipeline/
    mapping_pipeline.py
  config/
    session_backup.py
```

Design constraints:
- `model` and `math_core` remain UI-agnostic.
- `pipeline` orchestrates I/O + math.
- `gui` only coordinates user interaction and calls services.

## 9. Known Legacy Bugs to Prevent

Must be fixed in Python rewrite:
- Bug #1: output line count/columns issue (strict 5 cols only).
- Bug #2: cross-product direction/sign issue in grazing angle.
- Bug #3/#4/#5: retain corrected logic, avoid legacy indexing/typo mistakes.

## 10. Persistence

Session backup JSON should include:
- input file paths
- geometry points
- output options
- latest stats
- timestamp

Load on startup if present; save on successful parse/map and on exit.
Naming rule:
- Backup filename uses ANSYS `.dat` stem as prefix + datetime suffix:
  - `{ansys_stem}_{YYYYMMDD_HHMMSS}.json`

## 11. Test Expectations (minimum)

Must have tests for:
- geometry validation and axis construction
- ANSYS parser correctness
- SPECTRA parser correctness
- bilinear interpolation boundary consistency
- spatial search and out-of-bounds behavior
- mapping vectorized vs sequential consistency
- output writer 5-column strict format
- progress callback invocation during ANSYS parse, SPECTRA parse, and mapping

## 12. Next Session Handoff Checklist

When resuming, do this first:
1. Confirm actual current code tree vs module plan above.
2. Implement/verify geometry validation contract in `math_core/geometry.py`.
3. Ensure both readers expose progress callbacks and GUI wires two parse progress bars.
4. Wire `READY_TO_MAP` gating and `MAPPING` lock state in `gui/app_window.py`.
5. Run tests focusing on output format bug and progress-callback coverage.
