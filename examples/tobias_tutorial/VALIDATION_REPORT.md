# Tobias tutorials — validation report

## Scope and validation rule

The porting work follows one strict rule: each case has its own directory under `examples/tobias_tutorial/`, its own `run.py`, and its own `report.md`. The Python runner recreates the case with FoamPilot and launches the OpenFOAM workflow through FoamPilot. A case is validated only after the runner has completed successfully under OpenFOAM Foundation 13.

Short calculations are accepted as execution validation only when the report states the shortened `endTime`. They prove that the generated files, mesh pipeline, solver loading and numerical time integration work; they do not claim reproduction of the full long-duration training campaign.

## Validated cases

| Case | Main workflow | OpenFOAM 13 execution evidence | Status |
| --- | --- | --- | --- |
| 2D rotational axis-symmetric meshing | `blockMesh` → `surfaceFeatures` → `snappyHexMesh` → `extrudeMesh` → `createPatch` | All stages completed; final `constant/polyMesh/boundary` written | Validated meshing workflow |
| Pitot Tube | `ideasUnvToFoam` → `snappyHexMesh` → `changeDictionary` → `extrudeMesh` → `foamRun` | 30,500 UNV cells read; 206,180-cell snappy mesh; `PIMPLE: Converged in 7 iterations`; `End` | Validated short calculation |
| Fluidic Oscillator | UNV conversion → transforms → feature extraction → layered snappy mesh → flatten/extrude → `topoSet` → `setFields` → VoF `foamRun` | 435,120 UNV cells read; 1,621,498-cell layered mesh; scalar transport executed; `End` | Validated short calculation |
| Falling Droplets | UNV conversion → `changeDictionary` → `setFields` → VoF `foamRun` | All stages completed with alpha/MULES and pressure iterations | Validated short calculation |
| Magnus Effect | UNV conversion → `snappyHexMesh` → `extrudeMesh` → `changeDictionary` → `transformPoints` → incompressible `foamRun` | 3,000 UNV cells read; 3,239-cell mesh; reached `Time = 0.2s`; `End` | Validated short calculation |
| Cell Zone Generation | UNV conversion → `surfaceFeatures` → `snappyHexMesh` | 30,000 UNV cells read; 140,327-cell mesh; cell zones of 4,096, 2,789 and 44,937 cells; `End` | Validated meshing workflow |
| Meshing a Pipe 45° | `surfaceFeatures` → UNV conversion → scale transform → layered `snappyHexMesh` | 42,700 UNV cells; 245,587-cell layered mesh; 11 layer iterations; `End` | Validated meshing workflow |
| Meshing a Pipe 90° | `surfaceFeatures` → UNV conversion → layered `snappyHexMesh` | 173,056 UNV cells; 446,251-cell layered mesh; `End` | Validated meshing workflow |
| 2D AMI / Non-Conformal Coupling | `ideasUnvToFoam` → `snappyHexMesh` → `changeDictionary` → `flattenMesh` → `extrudeMesh` → `topoSet` → `createBaffles` → `splitBaffles` → `createNonConformalCouples` → `foamRun` | 23,104 points and 16,875 background cells; 140,321-cell snappy mesh; AMI1/AMI2 with 704 faces each, 1/1/1 coverage and 1,408 couplings; `foamRun` ended normally | Validated short calculation |
| Pseudo-2D Adaptive Mesh Refinement | `ideasUnvToFoam` → `surfaceFeatures` → `snappyHexMesh` → adaptive `foamRun` | Complete FoamPilot runner executed with the recovered UNV/STL assets; snappy mesh checks passed; AMR/scalar-transport solver reached the configured bounded smoke-run end time and exited normally | Validated short calculation |
| Feature Edge Refinement | Three repetitions of `surfaceFeatureConvert` → `ideasUnvToFoam` → `snappyHexMesh` using standard, optimized and `levels ((distance level))` feature-edge configurations | All three variants ended normally; representative variant read 169,781 points and 160,000 cells, detected 640 feature edges and wrote an 85,067-cell mesh | Validated meshing workflow |
| Sphere Meshing with Layers | `ideasUnvToFoam` → `snappyHexMesh` using supplied `channel.eMesh` | 44,541 points and 40,000 background cells; 48,940 snappy cells before layers; 3,720 layer faces; final 53,404-cell mesh; `End` | Validated meshing workflow |
| Thin Gap Meshing | `ideasUnvToFoam` → scale transformations → `snappyHexMesh` → inverse scale → `foamRun` | 9,212 points and 7,774 background cells; 437,872-cell snappy mesh; inverse scale completed; `foamRun` ended normally | Validated short calculation |
| Combustion Chamber cold-flow | `ideasUnvToFoam` → layered `snappyHexMesh` → incompressible `foamRun` | 190,281 points and 180,000 background cells; 501,867-cell layered mesh; `Finished meshing without any errors`; short run reached `End` | Validated short calculation |
| Battery Cooling | `ideasUnvToFoam` → layered `snappyHexMesh` → thermo-fluid `foamRun` | 261,800 points and 248,193 background cells; 681,111-cell layered mesh after local smoke-run reduction; `fluid` solver and `limitTemperature` loaded; `End` | Validated short thermo-fluid calculation |
| Dakota Tesla Valve | `ideasUnvToFoam` → `snappyHexMesh` → `extrudeMesh` → `dakota` | Successful coupling of DAKOTA with FoamPilot (`solve.py`); completed 10 optimization loops (mesh generation, forward/reverse flow, objective function evaluation) | Validated optimization workflow |
| NCC Heat Transfer | `ideasUnvToFoam` → `snappyHexMesh` → `extrudeMesh` → `createNonConformalCouples` → `foamRun` | Background mesh converted; dynamic mesh and NCC couples successfully generated; short dynamic mesh simulation reached `End` | Validated short dynamic mesh calculation |
| Rotating Rotor NCC | `ideasUnvToFoam` → `snappyHexMesh` → `createNonConformalCouples` → `foamRun` | Dynamic mesh generated; NCC couplings successfully established; solidBody motion simulation completed smoke run; `End` | Validated short dynamic mesh calculation |

## FoamPilot changes justified by execution

The new `OpenFOAMDictAddFile.write_raw` method is used only where the legacy attribute serializer cannot represent the exact OpenFOAM dictionary syntax. It preserves an existing `FoamFile` header and writes the original dictionary body without generating duplicate headers. A targeted regression test covers this behavior.

No new FoamPilot method was added for OpenFOAM-specific compatibility changes. The v12-to-v13 adaptations remain case-local in the corresponding `run.py`: `wedgeCoeffs`/coefficient layout, required `nLayers` and `expansionRatio`, VoF `interfaceCompression`, and removal of obsolete `cAlpha`. This keeps the shared API stable and prevents case-specific dialect assumptions from leaking into unrelated users.

## Pending scope

The remaining Tobias catalog contains additional cases covering dynamic meshes, AMI/ACMI, heat transfer, multi-region CHT, turbines, rotating machinery, optimization and other external dependencies. The 2D AMI/NCC, adaptive mesh refinement, sphere meshing, thin gap meshing, combustion chamber, battery cooling, Dakota Tesla valve, NCC Heat Transfer and rotating rotor NCC cases are no longer pending because their complete FoamPilot runners and OpenFOAM 13 smoke validations are now recorded above. They remain pending until each case has a dedicated runner, complete assets, an OpenFOAM 13-compatible input set, a successful execution, and a report. The register deliberately does not classify source-only templates or unexecuted folders as validated.

## References

[1]: https://holzmann-cfd.com/community/training-cases — Tobias Holzmann, OpenFOAM Training Cases.

[2]: https://wiki.openfoam.com/Tutorials_by_Tobias_Holzmann — OpenFOAM Wiki case collection by Tobias Holzmann.

[3]: https://openfoam.org/download/13-ubuntu/ — OpenFOAM Foundation 13 Ubuntu installation.

[4]: https://github.com/stevendaix/foampilot/pull/17 — FoamPilot pull request containing the implementation.
