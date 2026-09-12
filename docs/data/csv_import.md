# Import CSV boundary conditions in foampilot

## Goal

Add CSV-based time-varying boundary condition support to foampilot, working in both steady-state and transient modes with pandas DataFrames as input.

## Constraints

- Must work with OpenFOAM 13 (solver modules, not standalone binaries)
- Use pandas DataFrames as input interface
- Support both scalar and vector boundary conditions
- Support both uniform (table-based) and spatial (interpolated) CSV BCs
- CSVs written headerless to `<case>/constant/` for OpenFOAM `CsvTableReader` compatibility
- Use `solver_name` = `"incompressibleFluid"` (OpenFOAM 13 module pattern)

## Done

### Core implementation

- Created `foampilot/src/foampilot/boundaries/csv_boundary_condition.py` with:
  - `CsvTimeSeries` class for loading and querying time-varying data
  - `write_csv_table()` for writing CSV files in OpenFOAM-compatible format
  - `make_uniform_fixed_value_bc()` for scalar uniform BCs
  - `make_uniform_fixed_value_vector_bc()` for vector uniform BCs
  - `set_csv_condition()` high-level helper for uniform BCs
  - `set_spatial_csv_condition()` for spatial BCs with SciPy interpolation
  - Support for wide format, long format, and point-cloud CSV formats

### Bug fixes in CSV BC module

- Fixed `mergeSeparators` issue: removed invalid keyword for OpenFOAM 13, added explicit `mergeSeparators: False`
- Fixed `_write_spatial_field_from_template()` to use line-by-line parser instead of regex for reliable patch value replacement
- Added automatic missing patch BC generation based on mesh patch types (e.g., `fixedValue` + `value uniform 0` for `patch` type)
- Fixed spatial BC registration order: `write_boundary_conditions()` must be called BEFORE `set_spatial_csv_condition()`

### OpenFOAMDirectReader fixes

- Fixed `_read_field()` boundary parsing:
  - Don't stop at first `}` (patch closing brace), only break on `boundaryField` closing brace
  - Handle blank lines in boundary parsing
  - Fixed patch name detection when line doesn't end with `{`
- Added `_parse_nonuniform_list()` to parse OpenFOAM nonuniform lists (scalar and vector)
- Extended `read_field()` to reconstruct full cell array from `internalField` + boundary values when internal is uniform

### Solver selection fix

- Changed `solver.py` `_update_solver()` to always keep `"incompressibleFluid"` regardless of `energy_activated`
- Previously, setting `energy_activated=True` would switch to `"fluid"` solver which doesn't exist in OpenFOAM 13

### Examples

- Created `examples/csv_example/run_uniform_scalar.py` (steady + transient)
- Created `examples/csv_example/run_uniform_vector.py`
- Created `examples/csv_example/run_spatial.py`
- Created `examples/csv_example/run_spatial_steady.py`

### Verification

- `verify_csv_post.py` passes for all 5 cases:
  - `case_uniform_scalar_steady`: OK
  - `case_uniform_scalar`: OK
  - `case_uniform_vector`: OK
  - `case_spatial`: OK
  - `case_spatial_steady`: OK

### Visualization

- `visualize_csv_direct.py` uses `OpenFOAMDirectReader` for spatial cases (reads nonuniform boundary values correctly)
- Falls back to `FoamPostProcessing` / `foamToVTK` for uniform cases
- Generates slices, contours, and vector magnitude plots

## Current status

All 5 example cases run successfully and produce correct field files. The `OpenFOAMDirectReader` can now read spatial BC values from field files. The main limitation is that `incompressibleFluid` without energy doesn't write `T` to VTK, so uniform scalar cases need either `foamToVTK` with the solver that writes T, or the field files need to be inspected directly.

## Next steps

- Verify visualization works end-to-end for all cases
- Consider adding `scalarTransportFoam` as an alternative solver for passive scalar transport
- Potentially add more interpolation methods for spatial BCs
