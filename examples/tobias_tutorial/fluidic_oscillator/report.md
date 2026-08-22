# Tobias — Fluidic Oscillator

## Identification and objective

This directory ports Tobias Holzmann’s **Bifluidic/Fluidic Oscillator** training case. The case is an incompressible VoF calculation using `incompressibleVoF`, a scalar-transport function object for `S`, `fvModels`, and a background UNV mesh with a triangulated oscillator surface. The full `fluidicOscillator-12.tar.gz` archive is required because the public GitHub repository omits the large background mesh and STL input.

The physical objective is to reproduce the transient two-phase flow and passive scalar transport through the oscillator. The original training workflow performs background-mesh conversion, geometric translation, feature extraction, snappy meshing with layers, 2-D flattening/extrusion, a cell-zone construction, alpha initialization, and transient VoF calculation.

## FoamPilot implementation

`run.py` creates a clean case and emits every text dictionary and initial field through `OpenFOAMDictAddFile.write_raw`. It then calls each OpenFOAM utility through the FoamPilot `Solver.run_command` boundary. The source shell `run` file is not invoked. The UNV and STL files remain local assets because their archive sizes exceed ordinary Git repository file limits; the runner documents the required layout.

Three OpenFOAM 13 input adaptations were required. The `extrudeMesh` linear-normal coefficients receive `nLayers` and `expansionRatio` inside `linearNormalCoeffs`. The alpha convection entry becomes `Gauss interfaceCompression vanLeer 1`, which is the OpenFOAM 13 interface-compression syntax. The obsolete `cAlpha` entry is removed from the alpha solver block because OpenFOAM 13 rejects it as deprecated and unused.

The runner shortens `endTime` from 2 seconds to 0.002 seconds for a bounded smoke calculation. It does not claim to reproduce the overnight production run described by Tobias.

## Execution and evidence

| Stage | FoamPilot-launched command | Evidence |
| --- | --- | --- |
| Background conversion | `ideasUnvToFoam cad/backgroundMesh.unv` | `Read 435120 cells and 95564 boundary faces`; `End` |
| Coordinate preparation | `transformPoints translate=(-0.0005 0 0)` | Completed |
| Feature extraction | `surfaceFeatures` | Completed |
| Surface meshing | `snappyHexMesh -overwrite` | `Layer mesh: cells:1621498 faces:5017970 points:1776159`; `End` |
| 2-D preparation | `flattenMesh`, then `extrudeMesh` | `Writing mesh to "constant/region0"`; `End` |
| Final translation | `transformPoints translate=(0 -0.005 0)` | Completed |
| Passive scalar zone | `topoSet` | `Created cellZoneSet c0` |
| VoF initialization | `setFields` | `Setting internal values of volScalarField alpha.water`; `End` |
| Calculation | `foamRun -solver incompressibleVoF` | `PIMPLE: Converged`; `End` |

The case is therefore **validated: short calculation run**. The calculation reached time `0.00216642 s`, completed the scalar-transport function object, respected the interface Courant controls, and ended without a fatal error. The full 2-second training run remains intentionally unexecuted because the original case is designed for a much longer transient campaign.

## References

[1]: https://holzmann-cfd.com/community/training-cases/fluidic-oscillator — Tobias Holzmann, Fluidic Oscillator.

[2]: https://github.com/shor-ty/OpenFOAMTutorials/tree/main/cases/openfoam.org/fluidicOscillator — source case and original workflow.

[3]: https://openfoam.org/download/13-ubuntu/ — OpenFOAM Foundation 13 Ubuntu installation instructions.
