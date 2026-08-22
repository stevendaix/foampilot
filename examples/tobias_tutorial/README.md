# Tobias Holzmann tutorials ported with FoamPilot

Each tutorial is an independent directory. A directory contains a `run.py` that recreates the OpenFOAM case through FoamPilot, the local geometry/assets required by the case, and a `report.md` describing the physical objective, source version, every generation step, every OpenFOAM 13 adaptation, and the execution evidence.

The original Tobias shell `run` files are never used as the implementation. They are source references only. Text dictionaries are emitted from Python through FoamPilot; OpenFOAM utilities are launched through FoamPilot's solver command boundary. A case may use a supplied STL, UNV or HDF asset because Tobias's public GitHub repository omits large geometry files, but its case setup remains controlled by the Python runner.

## Validation rule

A case is **validated** only when its `run.py` has been executed with OpenFOAM 13 and all intended generation and calculation stages exit successfully. A report must include the relevant log markers, final mesh evidence, and solver completion evidence. A case with missing geometry, unavailable external software, a known OpenFOAM dialect conflict, or an unexecuted runner is **not validated**; it must be recorded as blocked or pending.

For long production cases, a short smoke run may be recorded separately from the full training calculation. The report must state the shortened `endTime`, the reason for shortening, and must not claim that the full reference run was reproduced.

## Current directories

| Directory | Tobias case | Scope | Status |
| --- | --- | --- | --- |
| `2d_rotational_axis_symmetric` | 2D Rotational Axis-Symmetric Meshing | `blockMesh` → `surfaceFeatures` → `snappyHexMesh` → `extrudeMesh` → `createPatch` | Validated on OpenFOAM 13 |
| `pitot_tube` | Pitot Tube | UNV conversion → `snappyHexMesh` → `changeDictionary` → `extrudeMesh` → `foamRun` | Validated with a documented short calculation |
| `fluidic_oscillator` | Fluidic Oscillator | UNV conversion → transforms → feature extraction → snappy meshing → flatten/extrude → `topoSet` → `setFields` → `foamRun` | Validated with a documented short calculation; run `fetch_assets.py` first |
| `falling_droplets` | Falling Droplets | UNV conversion → `changeDictionary` → `setFields` → `foamRun` | Validated with a documented short calculation; run `fetch_assets.py` first |

The remaining Tobias training cases are being ported incrementally in the same format. They are not marked validated until their individual Python runner has completed successfully.
