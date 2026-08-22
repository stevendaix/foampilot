# Tobias — Pseudo-2D Adaptive Mesh Refinement

## Objective

This case reproduces Tobias Holzmann’s pseudo-2D adaptive mesh refinement tutorial. The documented case uses a passive scalar `S` as the refinement criterion; OpenFOAM’s dynamic refinement is inherently 3-D, so the geometry is pseudo-2D.

## FoamPilot implementation

`run.py` generates the complete case through FoamPilot, places the official UNV background mesh and cylinder STL, and launches `ideasUnvToFoam`, `surfaceFeatures`, `snappyHexMesh` and `foamRun`. The missing cylinder STL was recovered from the official v12 archive because the repository case references `constant/geometry/cylinder.stl` without including it.

## Execution status

The mesh stages completed under OpenFOAM 13. The generated snappy mesh contained 3,300 cells and passed the final mesh checks. The solver started successfully and executed the AMR/scalar-transport loop, but even the bounded smoke-test duration remained computationally expensive: after approximately 889 seconds the log was still advancing and reported `Selected 0 cells for refinement out of 19057`. The process was stopped to avoid an unbounded background calculation.

This case is therefore **prepared but not validated**. It must not be counted among the validated tutorials until `foamRun` reaches its configured `endTime` and exits successfully.

## Reference

[1]: https://holzmann-cfd.com/community/training-cases/adaptive-mesh-refinement — Tobias Holzmann, Pseudo-2D Adaptive Mesh Refinement.
