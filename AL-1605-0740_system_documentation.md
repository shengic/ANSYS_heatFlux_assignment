# AL-1605-0740 — SPECTRA → ANSYS Heat Flux Node Assignment
## System Documentation · Version 6.0 · Albert Sheng · 2025

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [System Architecture](#2-system-architecture)
3. [Module Reference](#3-module-reference)
4. [Complete Workflow](#4-complete-workflow)
5. [Mathematical Foundations](#5-mathematical-foundations)
6. [Data Structures](#6-data-structures)
7. [Input / Output File Formats](#7-input--output-file-formats)
8. [Known Bugs and Code Issues](#8-known-bugs-and-code-issues)
9. [Dependency Map](#9-dependency-map)

---

## 1. Purpose

This Excel VBA workbook is a **pre-processing tool for synchrotron radiation thermal analysis**. It bridges two independent software domains:

| Domain | Software | Data |
|--------|----------|------|
| Photon beam power distribution | **SPECTRA** (Tanaka & Kitamura) | Power density map on a 2D angular grid [kW/mrad²] |
| Structural/thermal finite element analysis | **ANSYS APDL** | Mesh nodes, elements, heat flux surface elements |

The tool reads both datasets, defines a geometric coordinate transformation, and produces an **ANSYS External Data** `.inp` file that maps interpolated projected heat flux [W/mm²] to every heat flux surface node in the FEA model.

**Application context:** ALS-U (Advanced Light Source Upgrade) undulator/wiggler front-end thermal analysis. Default output filename embedded in form: `EPU66-27A power for ansys.inp`, corresponding to an Elliptically Polarizing Undulator beamline component.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        form1  (UserForm)                         │
│  [ANSYS file]  [SPECTRA file]  [Geometry]  [Output settings]    │
└────────┬──────────────┬──────────────┬──────────────┬───────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
  readAnsys.bas   readSpectra.bas  sourceGeometry  main.bas
         │              │           .cls            │
         ▼              ▼              │             ▼
  ansysCoordinate  powerDensityNode   │      storeAnsysInterpolate
  ansysElement     powerDensityElement│      CoordinateAndPower()
  ansysHeatFlux    fallIntoThis       │             │
   .cls            SpectraElement     │             ▼
                   .cls           ◄───┘     ansysHeatFlux
                                             .mapAnsysCoordinate
                                             .calculateInterpolate
                                             .getGrazingAngle
                                                    │
                                                    ▼
                                          writeAnsysHeatFlux
                                          Elements()  → .inp
```

### VBA Project Structure

```
VBAProject
├── Microsoft Excel Objects
│   ├── Sheet2 (start)          — empty
│   └── ThisWorkbook            — empty
├── Forms
│   ├── form1                   — main control panel
│   └── UserForm1               — progress bar widget (unused in final flow)
├── Modules (.bas)
│   ├── main                    — globals, orchestration
│   ├── readAnsys               — ANSYS APDL file parser
│   ├── readSpectra             — SPECTRA output file parser
│   ├── readGeometry            — alternate geometry config file reader
│   ├── backupModule            — parameter persistence
│   ├── ExportAllModule         — VBA source export utility
│   └── testProgress            — progress bar test stub
└── Class Modules (.cls)
    ├── ansysCoordinate         — ANSYS node: {inode, x, y, z}
    ├── ansysElement            — ANSYS element: {element, elementNodes}
    ├── ansysHeatFlux           — heat flux element + all calculations
    ├── powerDensityNode        — SPECTRA grid node: {inode, xmrad, ymrad, powerDensity}
    ├── powerDensityElement     — SPECTRA bilinear element (4 nodes + coefficients)
    ├── fallIntoThisSpectraElement — spatial search: (xmrad,ymrad) → element index
    ├── sourceGeometry          — orthonormal basis from 3 control points
    └── oneAnsysSpectraHeatFluxType — skeleton class, transferXYZtoXYmrad() empty
```

---

## 3. Module Reference

### 3.1 `main.bas` — Global Variables and Orchestration

#### Global Variables

| Variable | Type | Description |
|----------|------|-------------|
| `spectraFileName` | String | Full path to SPECTRA .dta file |
| `ansysFileName` | String | Full path to ANSYS .dat file |
| `ansysNodes` | Collection | Key: node number (string) → `ansysCoordinate` |
| `ansysElements` | Collection | Key: element number → `ansysElement` (not used in output) |
| `ansysHeatFluxElements` | Collection | Key: sequential index → `ansysHeatFlux` |
| `SpectraPowerNodes` | Collection | Key: sequential index → `powerDensityNode` |
| `spectraPowerElements` | Collection | Key: sequential index → `powerDensityElement` |
| `geometry` | sourceGeometry | Orthonormal basis for coordinate transform |
| `xSource, ySource, zSource` | Double | Photon source point [mm] |
| `xTarget, yTarget, zTarget` | Double | Beam axis reference point [mm] |
| `xSide, ySide, zSide` | Double | Side point for X-axis definition [mm] |
| `totalPowerRatio` | Double | Scaling factor applied to output heat flux (default 1.0) |
| `spectraTotalNode` | Long | Total nodes in SPECTRA grid |
| `spectraTotalColumn` | Long | Number of x-columns in SPECTRA grid |
| `spectraTotalRow` | Long | Number of y-rows in SPECTRA grid |
| `spectraPeakPowerDensity` | Double | Peak power density [kW/mrad²] |
| `spectraTotalPower` | Double | Integrated total power [kW] |
| `minX, maxX, minY, maxY` | Double | SPECTRA grid angular extent [mrad] |

#### Key Subroutines

**`showForm()`** — Entry point. Calls `form1.Show`.

**`readAll()`** — Full sequential pipeline (used for batch/debug runs):
```
readAnsysFile → readSpectraFile → storeAnsysInterpolateCoordinateAndPower → writeAnsysHeatFluxElements
```

**`storeAnsysInterpolateCoordinateAndPower()`** — Main mapping loop. For each `ansysHeatFlux` element:
1. Compute 4-node centroid → `ansysHFluxElement.x/y/z`
2. Call `mapAnsysCoordinateToMrad()`
3. Call `fallIntoThisSpectraElement(xmrad, ymrad)` → `whichElement`
4. If > 0: retrieve `spectraPowerElements.Item(whichElement)`, call `calculateInterpolatePower()`

**`writeAnsysHeatFluxElements()`** — Output loop. Writes comma-delimited file.

**`ArrayToCommaDelimitedString(inputArray)`** — Formats each element as `0.0000E+00`.

**`estimatedLineCount(fileName)`** — Approximates line count as `fileSize ÷ 50` for progress bar.

---

### 3.2 `readAnsys.bas` — ANSYS APDL Parser

State machine with variable `nextSection ∈ {"nothing", "node", "element", "heat flux"}`.

**Trigger patterns (VBScript regex):**

| Pattern | Action |
|---------|--------|
| `"Nodes for the whole assembly"` | Switch to node mode; skip next 3 lines |
| `"/com,\*{1,} Elements for"` | Switch to element mode; skip 4 lines; count only |
| `"/com,\*{1,} Create "Heat Flux""` | Switch to heat flux mode; skip 4 lines |
| `"-1"` | Terminate current section |

**`parseAnsysNodeLine(stringLine)`**
- Input: `"  nodeID  X  Y  Z  ..."` (mm)
- Strips leading/multi-spaces via regex `^\s+` and `\s+`
- Returns `ansysCoordinate` object

**`parseAnsysHeatFluxLine(stringLine)`**
- Input: 13+ token line; token[0] = element ID; tokens[5..12] = node IDs
- Nodes retrieved from `ansysNodes` Collection by ID string key
- Returns `ansysHeatFlux` with `elementNodes` Collection and computed `surfaceArea_mm2`

---

### 3.3 `readSpectra.bas` — SPECTRA File Parser

**File format:** Space-delimited, first 2 lines are header (skipped via `i > 1` counter).

```
# line 1: header/comment
# line 2: header/comment
 -3.000  -3.000   0.12345E+02
 -2.900  -3.000   0.13456E+02
 ...
```

Columns: `xmrad  ymrad  powerDensity[kW/mrad²]`

**Auto-detection of grid dimensions:**
```
IF ymrad ≠ previousYvalue AND spectraTotalColumn = 0 AND previousYvalue ≠ "" THEN
    spectraTotalColumn = inode - 1   ← first row just completed
END IF
```
Then: `spectraTotalRow = spectraTotalNode / spectraTotalColumn`

**Post-read element construction:**
```
spectraTotalElement = (spectraTotalRow - 1) × (spectraTotalColumn - 1)
For spectraElement = 1 To spectraTotalElement
    powerDElement.initialized(spectraElement, spectraTotalColumn)
    → assigns 4 nodes, computes area, calls calculcateInterpolateCoefficients()
Next
```

---

### 3.4 Class: `ansysHeatFlux`

The central computation class. Fields:

| Field | Type | Description |
|-------|------|-------------|
| `element` | Long | Surface element number |
| `elementNodes` | Collection | 8 `ansysCoordinate` objects (4 corner + 4 midside) |
| `x, y, z` | Double | Centroid coordinates [mm] |
| `vx, vY, vz` | Double | Displacement vector from source |
| `xmrad, ymrad` | Double | Angular coordinates [mrad] |
| `distanceFromSource` | Double | `|v|` [mm] |
| `grazingAngleRadian/Degree` | Double | Beam incidence grazing angle |
| `normalPowerDensity` | Double | Interpolated power density [kW/mrad²] |
| `projectedPowerDensity` | Double | `sin(θ) × normal` [kW/mrad²] |
| `normalPowerDensityIn_W_mm2` | Double | [W/mm²] |
| `projectedPowerDensityIn_W_mm2` | Double | [W/mm²] |
| `surfaceArea_mm2` | Double | 3D surface area of element [mm²] |
| `totalPower_W` | Double | `surfaceArea × projectedPowerDensity` [W] |

**Key methods:**

`mapAnsysCoordinateToMrad()` — coordinate transformation (see §5.1)

`calculateHeatFluxSurfaceArea()` — calls `quadrilaterialArea()` on corner nodes 1–4

`getGrazingAngleInRadian()` — surface normal via cross product + dot product with beam vector (see §5.3)

`calculateInterpolatePower(spectraElement)` — bilinear interpolation + unit conversion + projection (see §5.2, §5.4)

`quadrilaterialArea(node1..4)` — splits quadrilateral into two triangles; area = ½(|v12×v13| + |v13×v14|)

---

### 3.5 Class: `powerDensityElement`

Represents one cell of the SPECTRA grid mesh (bilinear quadrilateral element).

**Node index convention:**
```
Node1(x₁,y₂) ────── Node3(x₂,y₂)
     |                     |
     |    element area      |
     |   = |Δx × Δy|        |
Node2(x₁,y₁) ────── Node4(x₂,y₁)
```

Index derivation from `elementNumber` (1-based, left-to-right, top-to-bottom):
```
j = ⌈elementNumber / (m-1)⌉          ← row (1-based)
i = elementNumber - (j-1)(m-1)        ← column (1-based)
index1 = (j-1)·m + i                  ← Node1: top-left
index2 = j·m + i                      ← Node2: bottom-left
index3 = (j-1)·m + i + 1              ← Node3: top-right
index4 = j·m + i + 1                  ← Node4: bottom-right
```
where `m = spectraTotalColumn`.

**`calculcateInterpolateCoefficients()`** — see §5.2 for full derivation.

---

### 3.6 Class: `fallIntoThisSpectraElement`

Pre-computes 1D grid boundary arrays from element corner coordinates.

```
xgrids(1..spectraTotalColumn)   ← x-boundaries of columns
ygrids(1..spectraTotalRow)      ← y-boundaries of rows
```

**`fallIntoThisSpectraElement(xmrad, ymrad) → Long`**

Linear search O(N_col + N_row):
```
IF xmrad ∈ [minX, maxX] AND ymrad ∈ [minY, maxY] THEN
    FOR i = 1 TO spectraTotalColumn-1
        IF xmrad ∈ [xgrids(i), xgrids(i+1)] THEN
            FOR j = 1 TO spectraTotalRow-1
                IF ymrad ∈ [ygrids(j), ygrids(j+1)] THEN
                    RETURN (j-1)·(spectraTotalColumn-1) + i
                END IF
            NEXT j
        END IF
    NEXT i
END IF
RETURN 0   ← outside grid
```

---

### 3.7 Class: `sourceGeometry`

Constructs an orthonormal local coordinate system from three user-supplied points.

**Input:** `Source(xₛ,yₛ,zₛ)`, `Target(xₜ,yₜ,zₜ)`, `Side(xₚ,yₚ,zₚ)` read from `form1`.

**Construction:**
```
ê_Z_raw = Target − Source                       ← beam axis direction
ê_X_raw = Side − Source                         ← horizontal reference
ê_Y = ê_Z_raw × ê_X_raw                         ← vertical (out-of-plane)
ê_X = orthonormalize(ê_X_raw)
ê_Y = orthonormalize(ê_Y)
ê_Z = orthonormalize(ê_Z_raw)
```

`orthonormalize(v)` = `v / |v|`

Basis vectors displayed on form as labels `xV`, `yV`, `zV`.

---

### 3.8 `backupModule.bas` — Parameter Persistence

On form exit (`CommandButton2_Click`), writes all form field values to:
```
<ansys_file_directory>\<ansys_filename_noext>.backup.txt
```
Includes timestamp, ANSYS/SPECTRA statistics, geometry points, output settings.

---

## 4. Complete Workflow

```
USER ACTION                          VBA RESPONSE
───────────────────────────────────────────────────────────────────────
1. Open workbook → run showForm()    form1 displayed

2. Click "Upload ANSYS file"         File picker (*.dat)
                                     → readAnsysFile()
                                       ├─ parse nodes → ansysNodes
                                       ├─ count elements → totalElementCount
                                       └─ parse heat flux → ansysHeatFluxElements
                                     Display: total nodes / elements / flux elements
                                     Enable "Upload SPECTRA file" button

3. Click "Upload SPECTRA file"       File picker (*.dta, *.data, *.dta2)
                                     → readSpectraFile()
                                       ├─ parse nodes → SpectraPowerNodes
                                       └─ build elements → spectraPowerElements
                                            └─ pre-compute bilinear coefficients
                                     → storeAnsysInterpolateCoordinateAndPower()
                                       ├─ sourceGeometry.initialized()
                                       └─ For each ansysHeatFlux element:
                                            ├─ centroid → (x,y,z)
                                            ├─ mapAnsysCoordinateToMrad → (xmrad,ymrad)
                                            ├─ fallIntoThisSpectraElement → whichElement
                                            └─ calculateInterpolatePower()
                                     Display: grid dims, mrad range, peak PD, total power

4. (Optional) Edit geometry fields   Click "Update geometry"
   and click update                  → storeAnsysInterpolateCoordinateAndPower() re-runs

5. Set output path / filename        Form validation: enables "Create heat flux file"

6. Click "Create heat flux file"     → writeAnsysHeatFluxElements()
                                       ├─ apply totalPowerRatio
                                       ├─ write x,y,z,pd_scaled,pd lines
                                       └─ display total elements + total power in kW
                                     MsgBox: file created
                                     Enable "View file location" button

7. Click "Exit"                      → backupFormParameters() → .backup.txt
                                     Unload form, End
```

---

## 5. Mathematical Foundations

### 5.1 Angular Coordinate Transformation

For each ANSYS heat flux element centroid **P** = (x, y, z) [mm]:

**Step 1 — Displacement vector from source:**
```
v = P − S = (x−xₛ,  y−yₛ,  z−zₛ)
R = |v|   [mm]
```

**Step 2 — Project onto local basis (arcsin approximation):**
```
xmrad = arcsin( v · ê_X / R ) × 1000   [mrad]
ymrad = arcsin( v · ê_Y / R ) × 1000   [mrad]
```

**Validity:** For undulator radiation `|angle| < 5 mrad`, so `sin(θ) ≈ θ` to better than 0.004%, making `arcsin(sin θ) = θ` exact. Alternative `atan2` formulation (commented out) gives identical results in this regime.

---

### 5.2 Bilinear Interpolation

Given a SPECTRA grid element with corner power densities:

```
f₁ = pd at (x₁, y₂)    f₃ = pd at (x₂, y₂)
f₂ = pd at (x₁, y₁)    f₄ = pd at (x₂, y₁)
```

The interpolating polynomial is:
```
φ(x,y) = a₀ + a₁x + a₂y + a₃xy
```

Coefficients (Δ = (x₂−x₁)(y₂−y₁)):

| Coefficient | Formula |
|-------------|---------|
| a₀ | `(f₂·x₂y₂ − f₁·x₂y₁ − f₄·x₁y₂ + f₃·x₁y₁) / Δ` |
| a₁ | `(−f₂·y₂ + f₁·y₁ + f₄·y₂ − f₃·y₁) / Δ` |
| a₂ | `(−f₂·x₂ + f₁·x₂ + f₄·x₁ − f₃·x₁) / Δ` |
| a₃ | `(f₂ − f₁ − f₄ + f₃) / Δ` |

**Boundary condition verification (all satisfied):**
```
φ(x₁, y₁) = f₂  ✓
φ(x₁, y₂) = f₁  ✓
φ(x₂, y₁) = f₄  ✓
φ(x₂, y₂) = f₃  ✓
```

**Implementation note (matrixC diagonal):** The code constructs a 4×4 `matrixA` and treats 1D `matrixB` as if it were a 4×4 matrix. Because `matrixB` is a vector, `matrixC(i,j) = Σₖ A(i,k)·B(k)` is identical for all j. Therefore `matrixC(i,i) = matrixC(i,0)` = correct coefficient aᵢ. The diagonal extraction is mathematically valid, though confusingly written.

---

### 5.3 Grazing Angle Computation

**Surface normal** at element centroid via cross product of edge vectors from corner node 2:
```
v₁ = node1 − node2   (edge j→i)
v₂ = node3 − node2   (edge j→k)
n = v₂ × v₁  [code; note: reversed from comment which says v1×v2]
n̂ = n / |n|
```
The reversal makes **n̂** point inward (toward the body rather than toward the beam source). Since `Abs()` is applied to the final result, this produces the correct grazing angle magnitude.

**Grazing angle** (complement of incidence angle from surface normal):
```
θ_grazing = |π/2 − arccos( n̂ · v̂ )|
```
where `v̂ = v / R` is the unit beam direction vector.

For a surface correctly facing the beam: `n̂ · v̂ ≈ cos(π/2 − θ) = sin θ`, so `θ_grazing = arcsin(n̂ · v̂)`, which for small θ ≈ θ itself.

---

### 5.4 Unit Conversion: kW/mrad² → W/mm²

SPECTRA reports power density in the angular domain. The conversion to spatial flux at distance R uses the inverse-square law of solid angles:

```
Ω [mrad²] = 1 mrad² = (10⁻³ rad)² = 10⁻⁶ rad²

φ [kW/mrad²] = φ [kW] / (10⁻⁶ rad²)

At distance R_m [m], solid angle element dΩ subtends area:
   dA_m² = R_m² · dΩ [m²]

Therefore:
   flux [W/m²] = φ [kW/mrad²] × 10³ [W/kW] × 10⁻⁶ [mrad²/rad²] / R_m²
               = φ × 10⁻³ / R_m²   [W/m² · m² = W/rad²/m² ... ]
```

Expanding correctly:
```
flux [W/m²] = φ [kW/mrad²] × 10³ [W/kW] / (10⁻³)² [mrad²→rad²] / R_m²
            = φ × 10³ × 10⁶ / R_m²
            = φ × 10⁹ / R_m²   [W/m²]

Convert to W/mm² (÷ 10⁶):
   flux [W/mm²] = φ × 10³ / R_m²

Substituting R_m = R_mm / 1000:
   flux [W/mm²] = φ × 10³ / (R_mm / 1000)²
               = φ × 10³ × 10⁶ / R_mm²
               = φ × 10⁹ / R_mm²
```

**Code implementation (confirmed correct):**
```vba
normalPowerDensityIn_W_mm2 = normalPowerDensity * 1000 / (distanceFromSource / 1000#) ^ 2
```
= φ × 10³ / (R_mm/10³)² = φ × 10⁹ / R_mm² ✓

**Projected flux onto inclined surface:**
```
φ_projected [W/mm²] = sin(θ_grazing) × φ_normal [W/mm²]
```

**Total power on element:**
```
P [W] = A_surface [mm²] × φ_projected [W/mm²]
```

---

## 6. Data Structures

### Collection Key Conventions

| Collection | Key Type | Key Value |
|------------|----------|-----------|
| `ansysNodes` | String | Node number, e.g. `"12345"` |
| `ansysHeatFluxElements` | (sequential) | Integer index 1…N |
| `SpectraPowerNodes` | String | Sequential `"1"`, `"2"`, … |
| `spectraPowerElements` | String | Sequential `"1"`, `"2"`, … |
| `interpolateCoefficients` | String | `"1"` = a₀, `"2"` = a₁, `"3"` = a₂, `"4"` = a₃ |
| `fourNodes` (per element) | String | `"1"`, `"2"`, `"3"`, `"4"` |

---

## 7. Input / Output File Formats

### 7.1 ANSYS APDL Input File (*.dat)

Excerpt of expected structure:
```
/com,*** Nodes for the whole assembly
NBLOCK, ...
(header lines × 3)
  12345  1.23456E+02  0.00000E+00  5.67890E+01
  12346  1.23460E+02  ...
  -1
/com,*** Elements for the solver ...
EBLOCK, ...
(header lines × 4)
  ...two-line element records...
  -1
/com,*** Create "Heat Flux" on surface ...
CMBLOCK, ...
(header lines × 4)
  67890   ...  node1  node2  node3  node4  node5  node6  node7  node8
  -1
```

### 7.2 SPECTRA Output File (*.dta / *.data)

```
# header line 1
# header line 2
  -3.0000  -3.0000   1.23456E+01
  -2.9000  -3.0000   1.34567E+01
  ...
  (empty line or EOF terminates reading)
```

Column 1: horizontal angle [mrad], Column 2: vertical angle [mrad], Column 3: power density [kW/mrad²].

Data scanned in row-major order: all x values at y = y_min, then next y row, etc.

### 7.3 ANSYS External Data Output File (*.inp)

```
1.2346E+02,    0.0000E+00,    5.6789E+01,    4.5678E-02,    4.5678E-02,    0.0000E+00, ...
```

Columns per line:
1. `x` [mm] — element centroid x
2. `y` [mm] — element centroid y
3. `z` [mm] — element centroid z
4. `projectedPowerDensityIn_W_mm2 × totalPowerRatio` [W/mm²]
5. `projectedPowerDensityIn_W_mm2` (unscaled) [W/mm²]
6–11. Zeros (⚠ **Bug #1** — spurious columns from oversized array declaration)

This file is imported into ANSYS via **External Data** (Mechanical) or **LDREAD** (APDL) to apply heat flux boundary conditions.

---

## 8. Known Bugs and Code Issues

### Bug #1 — Output file has 6 extra zero columns ⚠ HIGH

**File:** `main.bas` · **Sub:** `writeAnsysHeatFluxElements()`

```vba
' CURRENT (wrong):
Dim arrayString(10) As String   ' declares indices 0..10
' indices 5..10 never assigned → formatted as "0.0000E+00"

' CORRECT:
Dim arrayString(4) As String    ' indices 0..4 only
```

**Impact:** Every line of the `.inp` output file contains 6 extra trailing zero columns. Whether ANSYS ignores these depends on the External Data import configuration. If column positions are fixed (not "by header"), the import may succeed but wastes file space. If ANSYS validates column count against a template, import will fail.

---

### Bug #2 — Cross product direction reversed in `getGrazingAngleInRadian()` ⚠ MEDIUM

**File:** `ansysHeatFlux.cls`

```vba
' First block (correct, ji × jk) — OVERWRITTEN:
nx = vx1 * vy2 - vy1 * vx2
ny = vy1 * vz2 - vz1 * vy2
nz = vz1 * vx2 - vx1 * vz2

' Second block (incorrect, jk × ji = −n) — THIS IS WHAT EXECUTES:
nx = (vy2 * vz1) - (vz2 * vy1)   ' = -(v1×v2)_x
ny = (vz2 * vx1) - (vx2 * vz1)
nz = (vx2 * vy1) - (vy2 * vx1)
```

**Mathematical analysis:**
```
n_code = v₂ × v₁ = −(v₁ × v₂) = −n_correct

n_code · v̂ = −(n_correct · v̂) = −cos α

arccos(−cos α) = π − α

θ_grazing = π/2 − (π − α) = α − π/2   [negative]

Abs(α − π/2) = |π/2 − α|   = correct magnitude ✓
```

`Abs()` on the final line fully compensates. However, the dead first block and the reversed second block are a maintenance hazard.

**Correct fix:**
```vba
' Remove first block entirely. Fix second block:
nx = vx1 * vy2 - vy1 * vx2   ' v1 × v2, correct outward normal
ny = vy1 * vz2 - vz1 * vy2
nz = vz1 * vx2 - vx1 * vz2
nd = Sqr(nx^2 + ny^2 + nz^2)
nx = nx/nd : ny = ny/nd : nz = nz/nd
' Abs() can be retained as safety guard but is no longer needed
```

---

### Bug #3 — `dotProduct()` function uses wrong index ⚠ MEDIUM (latent)

**File:** `ansysHeatFlux.cls`

```vba
' CURRENT (wrong):
dotProduct = v1(0)*v2(0) + v1(1)*v2(1) + v1(3)*v2(3)
'                                              ^ should be 2

' CORRECT:
dotProduct = v1(0)*v2(0) + v1(1)*v2(1) + v1(2)*v2(2)
```

**Impact:** The function is defined but never called in the current codebase. All dot products are computed inline with correct indexing. If this function is ever used in a refactoring, it will silently produce wrong results (missing the z-component, using an out-of-bounds index that may be 0).

---

### Bug #4 — Dead accumulation loop with typo in `putElementNodes()` ⚠ LOW

**File:** `powerDensityElement.cls`

```vba
' CURRENT:
averagePowerDensity = 0#
For ii = 1 To 4
    Set powerDNode = fourNodes.Item(ii)
    averagePowerDensity = averagePowreDensity + powerDNode.powerDensity / 4#
    '                     ^ typo: "averagePowreDensity" ≠ "averagePowerDensity"
    '                       In VBA without Option Explicit, creates a new variable = 0
    '                       Loop result is always just powerDNode.powerDensity / 4 (last node)
Next ii

' Then IMMEDIATELY overwritten with correct formula:
averagePowerDensity = (fourNodes.Item(1).powerDensity + fourNodes.Item(2).powerDensity + _
                      fourNodes.Item(3).powerDensity + fourNodes.Item(4).powerDensity) / 4#
```

**Fix:** Delete the for-loop. Add `Option Explicit` at module top to prevent silent variable creation from typos.

---

### Bug #5 — Wrong variable in `ReDim` inside `getVZ()` ⚠ LOW (no effect)

**File:** `sourceGeometry.cls`

```vba
Public Function getVZ() As Double()
  ReDim getVY(2)     ' ← should be: ReDim getVZ(2)
  getVZ = zV         ' this line overwrites, so bug has no effect
End Function
```

**Fix:** `ReDim getVZ(2)`

---

### Confirmed Correct (previously questioned)

| Item | Status | Reason |
|------|--------|--------|
| Bilinear interpolation diagonal extraction | ✅ Correct | matrixB is 1D vector; all columns of matrixC are equal; diagonal = correct coefficient values |
| Unit conversion kW/mrad² → W/mm² | ✅ Correct | Verified: `pd × 1000 / (R_mm/1000)²` = `pd × 10⁹ / R_mm²` |
| arcsin angle mapping | ✅ Correct for application | Undulator radiation < 5 mrad; arcsin ≈ atan2 to < 0.004% |

---

## 9. Dependency Map

```
showForm()
└── form1
    ├── uploadAnsysFileButton_Click()
    │   └── readAnsysFile()
    │       ├── estimatedLineCount()
    │       ├── parseAnsysNodeLine()  →  ansysCoordinate
    │       ├── [parseAnsysElementLine() — disabled]
    │       └── parseAnsysHeatFluxLine()
    │           ├── ansysCoordinate.x/y/z (lookup)
    │           └── ansysHeatFlux.calculateHeatFluxSurfaceArea()
    │               └── quadrilaterialArea()
    │
    ├── uploadSpectraFileButton_Click()
    │   ├── readSpectraFile()
    │   │   ├── powerDensityNode.initialized()
    │   │   └── powerDensityElement.initialized()
    │   │       ├── putElementNodes()
    │   │       └── calculcateInterpolateCoefficients()
    │   └── storeAnsysInterpolateCoordinateAndPower()
    │       ├── sourceGeometry.initialized()
    │       │   └── orthonormalize()
    │       ├── ansysHeatFlux.mapAnsysCoordinateToMrad()
    │       │   └── sourceGeometry.getVX() / getVY()
    │       ├── fallIntoThisSpectraElement.fallIntoThisSpectraElement()
    │       │   └── [determineGrid() called once at initialize()]
    │       └── ansysHeatFlux.calculateInterpolatePower()
    │           ├── powerDensityElement.interpolateCoefficients
    │           ├── ansysHeatFlux.getGrazingAngleInRadian()
    │           └── [unit conversion inline]
    │
    ├── createExternalFileButton_Click()
    │   └── writeAnsysHeatFluxElements()
    │       └── ArrayToCommaDelimitedString()
    │
    ├── updateGeometryButton_Click()
    │   └── storeAnsysInterpolateCoordinateAndPower()  [re-run]
    │
    └── CommandButton2_Click()  (Exit)
        └── backupFormParameters()
```

---

*Document generated from VBA source extraction of AL-1605-0740_Ansys_Spectra_data_heat_flux_nodes_assignment_with_form-v6_0.xlsm*
