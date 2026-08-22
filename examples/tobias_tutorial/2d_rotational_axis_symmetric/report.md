# Tobias — 2D rotational axis-symmetric meshing

## Identification

This case corresponds to Tobias Holzmann’s **2D Rotational Axis-Symmetric Meshing** training case. The training page describes a three-dimensional `snappyHexMesh` workflow followed by `extrudeMesh` to obtain a 2-D rotational axis-symmetric mesh. The full archive used here is `rotationalAxisSymmetricMeshing-12.tar.gz`, downloaded from the Tobias training-case page and ported to OpenFOAM Foundation 13.

## Objective

The purpose of the case is to demonstrate how a complex rotational axis-symmetric domain can be meshed in three stages. First, `blockMesh` creates the background mesh. Second, `surfaceFeatures` and `snappyHexMesh` conform the mesh to the triangulated `join.stl` geometry. Third, `extrudeMesh` creates the wedge mesh and `createPatch` removes empty or obsolete patches.

## FoamPilot implementation

`run.py` recreates the case in a clean `case/` directory. The nine source system dictionaries are embedded as Python templates. They are written through `OpenFOAMDictAddFile.write_raw`, which generates the case files from Python while preserving OpenFOAM syntax and comments. The STL geometry is placed under `constant/triSurface/join.stl`, then all utilities are launched through the FoamPilot `Solver.run_command` interface. The original shell `run` script is not executed.

The Python port applies one documented OpenFOAM 13 compatibility change. The OpenFOAM 12 source stores the wedge parameters in `sectorCoeffs`; the official OpenFOAM 13 template expects those parameters in `wedgeCoeffs`. The port changes only that dictionary key and leaves `axisPt`, `axis`, `angle`, `nLayers`, and the remaining case data unchanged.

## Execution protocol

| Step | FoamPilot/OpenFOAM action | Result |
| --- | --- | --- |
| 1 | Recreate `system/` and `constant/` from Python templates | Passed |
| 2 | Install `join.stl` under `constant/triSurface` | Passed |
| 3 | `blockMesh` | Passed |
| 4 | `surfaceFeatures` | Passed |
| 5 | `snappyHexMesh -overwrite` | Passed |
| 6 | Rename the OpenFOAM 12 `minZ` patch to `front` | Passed |
| 7 | `extrudeMesh` with OpenFOAM 13 `wedgeCoeffs` | Passed |
| 8 | `createPatch -overwrite` | Passed |
| 9 | Confirm `constant/polyMesh/boundary` exists | Passed |

## Validation evidence

The case was run with OpenFOAM 13 on Linux using:

```bash
source /opt/openfoam13/etc/bashrc
python3 run.py
```

The run completed successfully. The final `createPatch` log reports `End`, and the resulting `constant/polyMesh/boundary` file exists. The log also confirms that the zero-sized `axi` patch was removed, matching the behavior described by the source case’s final note.

This case is therefore **validated for the meshing workflow**. It is not a flow-solver tutorial: the source case contains no initial fields or solver run, so no physical time integration is claimed.

## References

[1]: https://holzmann-cfd.com/community/training-cases/2d-rotational-axis-symmetric-meshing — Tobias Holzmann, 2D Rotational Axis-Symmetric Meshing.

[2]: https://github.com/shor-ty/OpenFOAMTutorials/tree/main/cases/openfoam.org/2dAxisSymmetricMeshing — source case and original run workflow.

[3]: https://github.com/OpenFOAM/OpenFOAM-13/blob/master/etc/caseDicts/mesh/generation/extrudeMeshDict — official OpenFOAM 13 `extrudeMeshDict` template.
