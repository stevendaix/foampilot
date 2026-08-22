# Tobias — Meshing a Pipe at 90°

## Objective

This case reproduces the 90-degree bent pipe meshing tutorial documented by Tobias Holzmann. It demonstrates feature-edge extraction, `snappyHexMesh` and boundary-layer generation on the right-angle pipe geometry.

## FoamPilot workflow

`run.py` generates the dictionaries with `OpenFOAMDictAddFile.write_raw`, copies the official UNV background mesh and STL surface, then executes `surfaceFeatures`, `ideasUnvToFoam` and `snappyHexMesh` through FoamPilot. The original shell script is not executed. The embedded `salome.py` file is retained as source asset documentation but is not required for this OpenFOAM execution.

## Validation evidence

The OpenFOAM 13 run read **173,056 cells and 28,288 boundary faces**, produced a snapped mesh with **187,083 cells**, completed layer addition, and wrote a final layered mesh with **446,251 cells, 1,367,004 faces and 475,659 points**. The log ended successfully. This case is **validated as a meshing workflow**.

## Reference

[1]: https://holzmann-cfd.com/community/training-cases/meshing-a-pipe — Tobias Holzmann, Meshing a Pipe.
