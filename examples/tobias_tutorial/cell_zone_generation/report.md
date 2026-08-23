# Tobias — Cell Zone Generation

## Objective

This case reproduces Tobias Holzmann’s training case on creating cell zones during `snappyHexMesh` meshing. The case demonstrates how closed STL surfaces can assign distinct cell zones for later porosity or source-term modeling.

## FoamPilot workflow

`run.py` generates the text dictionaries through `OpenFOAMDictAddFile.write_raw`, copies the complete UNV background mesh and four STL surfaces, then launches `ideasUnvToFoam`, `surfaceFeatures` and `snappyHexMesh -overwrite` through FoamPilot. This is a meshing tutorial; the source does not include a physical solver calculation, so validation is defined as successful completion of the documented meshing workflow and production of `constant/polyMesh/cellZones`.

The STL paths are adapted from the source’s `constant/geometry` expectation to the generated case’s `constant/triSurface` layout. No shared FoamPilot API change was required.

## Validation evidence

| Stage | Evidence |
| --- | --- |
| Background conversion | `Read 30000 cells and 6800 boundary faces`; `End` |
| Feature extraction | `surfaceFeatures`; `End` |
| Snappy meshing | `Snapped mesh: cells:140327 faces:439141 points:158954`; `End` |
| Zone assignment | `cellZone1 size:4096`, `cellZone2 size:2789`, `cellZone3 size:44937` |
| Output | `constant/polyMesh/cellZones` and `faceZones` exist |

The case is **validated as a meshing/cell-zone workflow** under OpenFOAM Foundation 13.

## References

[1]: https://holzmann-cfd.com/community/training-cases/cell-zone-generation — Tobias Holzmann, Cell Zone Generation.

[2]: https://github.com/shor-ty/OpenFOAMTutorials/tree/main/cases/openfoam.org/cellZoneGeneration — source case and original workflow.
