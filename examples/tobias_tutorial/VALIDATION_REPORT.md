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

## FoamPilot changes justified by execution

The new `OpenFOAMDictAddFile.write_raw` method is used only where the legacy attribute serializer cannot represent the exact OpenFOAM dictionary syntax. It preserves an existing `FoamFile` header and writes the original dictionary body without generating duplicate headers. A targeted regression test covers this behavior.

No new FoamPilot method was added for OpenFOAM-specific compatibility changes. The v12-to-v13 adaptations remain case-local in the corresponding `run.py`: `wedgeCoeffs`/coefficient layout, required `nLayers` and `expansionRatio`, VoF `interfaceCompression`, and removal of obsolete `cAlpha`. This keeps the shared API stable and prevents case-specific dialect assumptions from leaking into unrelated users.

## Pending scope

The Tobias catalog contains additional cases covering dynamic meshes, AMI/ACMI, heat transfer, multi-region CHT, turbines, rotating machinery, optimization and other external dependencies. They remain pending until each case has a dedicated runner, complete assets, an OpenFOAM 13-compatible input set, a successful execution, and a report. The register deliberately does not classify source-only templates or unexecuted folders as validated.

## References

[1]: https://holzmann-cfd.com/community/training-cases — Tobias Holzmann, OpenFOAM Training Cases.

[2]: https://wiki.openfoam.com/Tutorials_by_Tobias_Holzmann — OpenFOAM Wiki case collection by Tobias Holzmann.

[3]: https://openfoam.org/download/13-ubuntu/ — OpenFOAM Foundation 13 Ubuntu installation.

[4]: https://github.com/stevendaix/foampilot/pull/17 — FoamPilot pull request containing the implementation.
