# Tobias — Meshing a Pipe at 45°

## Objective

This case reproduces the 45-degree bent pipe meshing tutorial documented by Tobias Holzmann. The focus is `snappyHexMesh`, feature-edge extraction and boundary-layer generation. Tobias documents the geometric difficulty at the outlet and reports an expected layer coverage below 100% for the 45-degree variant.

## FoamPilot workflow

`run.py` generates the OpenFOAM dictionaries with `OpenFOAMDictAddFile.write_raw`, copies the official UNV background mesh and STL surface, then executes `surfaceFeatures`, `ideasUnvToFoam`, `transformPoints scale=(1000 1000 1000)` and `snappyHexMesh` through FoamPilot. The original shell script is not executed.

## Validation evidence

The OpenFOAM 13 run read **42,700 cells and 8,110 boundary faces**, produced a snapped mesh with **168,393 cells**, then completed eleven layer-addition iterations and wrote a final layered mesh with **245,587 cells, 770,938 faces and 284,928 points**. The log ended successfully. This case is **validated as a meshing workflow**.

## Reference

[1]: https://holzmann-cfd.com/community/training-cases/meshing-a-pipe — Tobias Holzmann, Meshing a Pipe.
