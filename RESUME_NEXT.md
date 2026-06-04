# RESUME_NEXT.md

Last updated: 2026-06-04
Project: `K:\ANSYS heat flux assignment`
Scope: insertion-device rewrite (`.xlsm` -> Python + Tk)

## Start Prompt (copy/paste next session)

```text
Resume from spec.md and continue insertion-device Tk rewrite.
Read spec.md first, then run pytest, then continue next unfinished items.
Do not work on BM folder.
```

## Ground Rules

- Follow Markdown docs/spec as hard constraints.
- If any change would violate docs, list reason first and ask approval.
- Ignore `Implement_Ansys_BM_Power` (BM scope excluded for now).

## Current Status

- Version:
  - `v2.0` committed and pushed to `main` (`release: version 2.0`)
- Core flow implemented:
  - ANSYS upload parse + progress (background parse worker, throttled updates)
  - SPECTRA upload parse + progress (row-read + element-build phases)
  - Geometry validation from Source/Target/Horizontal
  - Mapping pipeline + output writer (5-column strict output)
- ANSYS cache:
  - Reuse prompt when valid cache exists
  - Delete current / delete all / cache browser
  - Async cache save after parse
- Session backup:
  - Save/restore wired to `session_backups/`
  - Backup filename uses ANSYS stem + datetime
  - Rerun from backup supported
- Logging / observability (added 2026-06-04):
  - `heatflux/config/app_logger.py` — rotating file logger, writes to `heatflux.log`
    (5 MB × 3 backups, DEBUG level, completely silent to UI)
  - Called once in `main.py` before Tk starts
  - All modules use `logging.getLogger(__name__)`
  - Key events logged: file parse start/end with duration, node/element/flux counts,
    grid dimensions, output path and line count, mapping out-of-grid stats,
    session backup save/load, user actions (upload, map), overwrite warnings
  - WARNING level: flux=0, node=0, SPECTRA grid <3×3, output overwrite,
    out-of-grid >5%, skipped heat flux lines (with reason)
  - ERROR level: mapping exceptions (with exc_info for traceback)
- GUI warning strip (added 2026-06-04):
  - Footer now has two labels: left = `footer_status_var` (was wired to no widget — fixed),
    right = `warn_strip_var` (orange, auto-clears after 12 s)
  - `_post_warning(msg)` / `_clear_warning()` methods in `app_window.py`
  - Triggered after mapping if out-of-grid > 5%
  - No new modal dialogs added
- Current UI labels/actions:
  - Execute button: `Map elemental power density`
  - Mapping totals formatting:
    - `Total power out: 0.000 kW`
    - `Total power (SPECTRA): 0.000 kW`
  - `Run sample validation` is removed from main UI
- Recent test baseline:
  - `28 passed`

## First Commands To Run

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py
```

## Key Files

- Spec and docs:
  - `spec.md`
  - `CONTEXT.md`
  - `AL-1605-0740_system_documentation.md` (legacy VBA reference)
  - `TDD_TEST_PLAN.md`
- App:
  - `main.py`
  - `heatflux/gui/app_window.py`
  - `heatflux/gui/geometry_frame.py`
- Logging:
  - `heatflux/config/app_logger.py`   ← rotating file logger setup
  - `heatflux.log`                     ← written at runtime (not committed)
- ANSYS cache:
  - `heatflux/config/ansys_cache.py`
- Session backup:
  - `heatflux/config/session_backup.py`
- I/O:
  - `heatflux/io/ansys_reader.py`
  - `heatflux/io/spectra_reader.py`
  - `heatflux/io/output_writer.py`
- Pipeline/math:
  - `heatflux/pipeline/mapping_pipeline.py`
  - `heatflux/math_core/geometry.py`
  - `heatflux/math_core/coordinate_transform.py`
  - `heatflux/math_core/grazing_angle.py`
  - `heatflux/math_core/spatial_search.py`
  - `heatflux/math_core/unit_conversion.py`

## Next Priorities

1. End-to-end run with real large `.dat` + `.dta` files and verify responsiveness.
2. Compare key output stats against legacy `.xlsm` reference output.
3. Final UI polish pass (spacing/borders/alignment consistency on target display scale).
4. Optional:
   - keep/refine cache browser UX
   - add dedicated formatting tests for UI status strings (if desired)

## Inputs Needed From User Next Session

- Real ANSYS file path (`*.dat`)
- Real SPECTRA file path (`*.dta`/`*.data`/`*.dta2`)
- Preferred default geometry values (S/T/H)
- Preferred output folder/filename convention
