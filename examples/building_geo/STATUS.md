# VoxCity → OpenFOAM Pipeline Status

## Goal
- Generate a CFD case from VoxCity HDF5 (`output/voxcity.h5`) with buildings correctly represented as a `buildings` patch in the OpenFOAM mesh.
- Produce post-processing images with only `buildings` and `ground` patches colored, others transparent.

## Progress

### Done
- Reset `config.json` `margin` from `100.0` to `50.0` to match documented behavior
- Ran full pipeline: `generate.py` → Gmsh meshing → OpenFOAM export → simulation → post-processing
- Verified solver converged with first attempt (residuals ~1e-9, no divergence)
- Identified post-processing slice origin bug: slices used `(0,0,z)` instead of mesh center, producing empty images
- Fixed `voxcity_postprocess.py` to compute slice origins from actual mesh bounds
- Added `buildings_only.png` visualization
- Identified root cause of misaligned geometry: `build123_geometry.py` created fluid box centered at origin in local coords, causing domain offset
- Fixed `build123_geometry.py` so fluid box spans `[0,dx]×[0,dy]×[0,dz]` in local coordinates before placement
- Rebuilt case; new mesh bounds now match VoxCity buildings: `x=[450193, 450541]`, `y=[5410944, 5411377]`, `z=[-5, 77]`

### In Progress
- Investigating solver crash at `t=1s` caused by poor mesh quality

### Blocked
- `checkMesh` reports 2 failures: max aspect ratio `877744` (10 cells) and max skewness `8.225` (1 highly skew face), causing immediate continuity errors and solver instability

## Key Decisions
- Use `margin=50.0` (not `100.0`) to keep buildings visible and domain size reasonable
- Generate all post-processing images from VTK files in `neighborhood_case/VTK/`
- Fix geometry origin in `build123_geometry.py` rather than shifting buildings post-hoc

## Next Steps
- Investigate and fix high-aspect-ratio and skew cells in Gmsh mesh (likely need mesh size constraints, boundary layer settings, or building surface mesh refinement)
- Rerun simulation after mesh quality passes `checkMesh`
- Regenerate final post-processing images once simulation completes

## Critical Context
- VoxCity buildings are at UTM coords `x=[450242.6, 450491.2]`, `y=[5410994.8, 5411327.1]`, `z=[0, 27]`
- Previous mesh was offset by ~250m in X/Y and too small; now corrected
- Current case: 9512 tetrahedra, 7 patches (`buildings`, `ground`, `inlet`, `outlet`, `side_left`, `side_right`, `top`)
- Solver log shows huge initial continuity errors (~636) and `k` bounding to `3.86e+10`, likely mesh-induced
- `foamToVTK` must be rerun after each simulation to regenerate boundary VTK files (`VTK/buildings/buildings_2000.vtk` etc.)

## Relevant Files
- `/home/steven/foampilot/examples/building_geo/neighborhood_demo/config.json`: margin reset to 50.0
- `/home/steven/foampilot/examples/building_geo/voxcity_export_work/src/build123_geometry.py`: fixed fluid box origin alignment
- `/home/steven/foampilot/examples/building_geo/voxcity_postprocess.py`: fixed slice origins to use mesh bounds center
- `/home/steven/foampilot/examples/building_geo/neighborhood_demo/neighborhood_case`: current broken case directory with misaligned mesh and solver crash
- `/home/steven/foampilot/examples/building_geo/neighborhood_demo/output/voxcity.h5`: VoxCity source data
- `/home/steven/foampilot/examples/building_geo/neighborhood_demo/generate.py`: main pipeline script
- `/home/steven/foampilot/foampilot/src/foampilot/postprocess/openfoam_pyvista.py`: `FoamPostProcessing` class used for VTK loading and visualization
