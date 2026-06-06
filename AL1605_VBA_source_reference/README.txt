AL-1605-0740 — SPECTRA → ANSYS Heat Flux Node Assignment
VBA Source Code Reference Archive
Version 6.0 · Albert Sheng · 2025
=====================================================================

This archive contains all VBA source code extracted from:
  AL16050740_Ansys_Spectra_data_heat_flux_nodes_assignment_with_formv6_0.xlsm

Extracted using olevba (oletools) from the binary vbaProject.bin stream.

FILE INVENTORY
--------------
Modules (.bas)
  main.bas                      Global variables + orchestration pipeline
  readAnsys.bas                 ANSYS APDL file parser (state machine)
  readSpectra.bas               SPECTRA .dta file parser
  readGeometry.bas              Alternate geometry config reader (non-primary path)
  backupModule.bas              Form parameter persistence to .backup.txt
  ExportAllModule.bas           VBA source export utility
  testProgress.bas              Progress bar test stub (not production code)

Class Modules (.cls)
  ansysCoordinate.cls           ANSYS node: {inode, x, y, z}
  ansysElement.cls              ANSYS element: {element, elementNodes}
  ansysHeatFlux.cls             Heat flux element + all calculations (centroid,
                                  mrad transform, grazing angle, interpolation,
                                  unit conversion, projection)
  powerDensityNode.cls          SPECTRA grid node: {inode, xmrad, ymrad, powerDensity}
  powerDensityElement.cls       SPECTRA bilinear element (4 nodes + coefficients)
  fallIntoThisSpectraElement.cls  Spatial search: (xmrad,ymrad) -> element index
  sourceGeometry.cls            Orthonormal basis from 3 control points
  oneAnsysSpectraHeatFluxType.cls Skeleton class (transferXYZtoXYmrad() empty)
  Sheet2.cls                    Excel sheet object (empty)
  ThisWorkbook.cls              Excel workbook object (empty)

Forms (.frm)
  form1.frm                     Main control panel UserForm
  UserForm1.frm                 Progress bar widget (empty in extracted form)

KNOWN BUGS (documented in AL-1605-0740_system_documentation.md)
---------------------------------------------------------------
Bug #1 HIGH   main.bas writeAnsysHeatFluxElements()
              Dim arrayString(10) -> 6 extra zero columns in output

Bug #2 MEDIUM ansysHeatFlux.cls getGrazingAngleInRadian()
              Cross product direction reversed (v2xv1 instead of v1xv2)
              Compensated by Abs() - fragile

Bug #3 MEDIUM ansysHeatFlux.cls dotProduct()
              v1(3)*v2(3) should be v1(2)*v2(2) - never called in production

Bug #4 LOW    powerDensityElement.cls putElementNodes()
              Typo averagePowreDensity creates silent new variable
              Dead loop - result immediately overwritten correctly

Bug #5 LOW    sourceGeometry.cls getVZ()
              ReDim getVY(2) should be ReDim getVZ(2) - no runtime effect

CONFIRMED CORRECT (previously questioned)
  - Bilinear interpolation diagonal extraction (matrixC)
  - Unit conversion kW/mrad2 -> W/mm2
  - arcsin angle mapping (valid for <5 mrad synchrotron radiation)

REFERENCE DOCUMENTS
-------------------
  AL-1605-0740_system_documentation.md  Full system documentation
  AL1605_project_structure.md           Python/Tk port structure
  CONTEXT.md                            Python port business context
  TDD_TEST_PLAN.md                      97 TDD test specifications
