# Complete VOF-to-DPM project map

The complete implementation is bundled in FoamPilot under `examples/openfoam13/vof_to_dpm/`.

| Area | Location |
|---|---|
| Python converter | `src/foampilot/utilities/vof_to_dpm.py` |
| Python tests | `test/test_vof_to_dpm.py` |
| Teaching example | `examples/course_vof_to_dpm.py` |
| PDF generator | `examples/generate_vof_to_dpm_technical_note.py` |
| Native offline extractor | `examples/openfoam13/vof_to_dpm/applications/vofToDpm/` |
| Incompressible bridge | `examples/openfoam13/vof_to_dpm/applications/incompressibleVoFClouds/` |
| Compressible bridge | `examples/openfoam13/vof_to_dpm/applications/compressibleVoFClouds/` |
| Original statisticalDPMFoam sources | `examples/openfoam13/vof_to_dpm/statisticalDPMFoam/` |
| OpenFOAM 13 tests | `examples/openfoam13/vof_to_dpm/test/openfoam13/` |
| Technical note and bibliography | `docs/fr/vof_to_dpm_technical_note.pdf`, `docs/fr/vof_to_dpm.bib` |

The main runnable cases are `vofToDpmSingleCell`, `vofToDpmParcelInBox`, `incompressibleVoFCloudsDamBreak` and `compressibleVoFCloudsDamBreak`. Start with [installation and execution](vof_to_dpm_openfoam13.md), then run the Python tests before compiling the OpenFOAM components.

The French and Chinese versions of this guide are available at `docs/fr/vof_to_dpm_openfoam13.md` and `docs/zh/vof_to_dpm_openfoam13.md`.
