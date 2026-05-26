# ANSYS_heatFlux_assignment

Python/Tk rewrite of the insertion-device ANSYS/SPECTRA heat-flux assignment workflow.

## Scope

- Active scope: insertion-device flow (`.xlsm` -> Python + Tk)
- Out of scope for current work: `Implement_Ansys_BM_Power` (BM implementation)

## Current Features (v2.0)

- ANSYS `.dat` upload/parse with progress
- SPECTRA `.dta/.data/.dta2` upload/parse with progress
- Source/Target/Horizontal geometry validation
- Mapping + ANSYS external data export (`*.inp`, strict 5-column format)
- ANSYS parse cache (reuse prompt, delete current/all, cache browser)
- Session backup/restore + rerun from backup

## Run

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py
```
