# Tobias — Pitot Tube

## Identification and objective

This directory ports Tobias Holzmann’s **Pitot Tube** training case, identified on the training page as a `pimpleFoam` case from the 2019 OpenFOAM community Christmas competition. The objective is to reproduce the transient flow around the Pitot geometry and the pressure/velocity analysis described by the case.

The full archive `pitotTube-12.tar.gz` was used because the GitHub repository intentionally removes large geometry and background-mesh files. The port uses the `2D_0msto50ms` case and retains the supplied UNV background mesh, STL surfaces and initial fields as local case inputs.

## FoamPilot implementation

`run.py` creates a clean case and writes all text dictionaries and fields through `OpenFOAMDictAddFile.write_raw`. The original shell script is not executed. The runner then uses FoamPilot’s `Solver.run_command` for `ideasUnvToFoam`, `snappyHexMesh`, `changeDictionary`, `extrudeMesh` and `foamRun`.

Two OpenFOAM 12-to-13 input adaptations were required. First, `extrudeModel linearNormal` requires `nLayers` and `expansionRatio` inside `linearNormalCoeffs` in OpenFOAM 13. Second, the source snappy dictionary’s STL paths resolve under `constant/geometry`, whereas the archive stores the files under `constant/triSurface`; the runner writes explicit relative paths to the preserved `triSurface` directory. The source `endTime` is shortened to `0.0005` for a reproducible smoke calculation; this is documented as a short validation run and is not claimed to reproduce the multi-day 50 m/s production run.

## Execution and evidence

| Stage | Command through FoamPilot | Evidence |
| --- | --- | --- |
| Background mesh conversion | `ideasUnvToFoam cad/backgroundMesh.unv` | `Read 30500 cells and 31744 boundary faces`; `End` |
| Surface meshing | `snappyHexMesh -overwrite` | `Snapped mesh: cells:206180 faces:669942 points:259200`; `End` |
| Boundary correction | `changeDictionary` | `Writing modified boundary`; `End` |
| 2-D extrusion | `extrudeMesh` | `Writing mesh to "constant/region0"`; `End` |
| Transient calculation | `foamRun -solver incompressibleFluid` | `PIMPLE: Converged in 7 iterations`; `End` |

The case therefore has a real successful mesh and solver execution under OpenFOAM 13. Its status is **validated: short calculation run**. The full 300-second production calculation from the original case is intentionally not run in this validation because the source description estimates multiple days for the 50 m/s target.

## References

[1]: https://holzmann-cfd.com/community/training-cases/pitot-tube — Tobias Holzmann, Pitot Tube training case.

[2]: https://github.com/shor-ty/OpenFOAMTutorials/tree/main/cases/openfoam.org/pitotTube — source case and original workflow.

[3]: https://openfoam.org/download/13-ubuntu/ — OpenFOAM Foundation 13 Ubuntu installation instructions.
