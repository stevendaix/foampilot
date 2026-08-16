# Patient 58 CFD Example Case

Complete aortic CFD simulation case for patient 58 (COA - Coarctation of the Aorta), from medical imaging to flow results. Uses foampilot modules and a corrected centerline extraction pipeline.

## Pipeline Overview

### 0. Centerline extraction (surface → centerline)

**Module**: `examples/coa/cad_reconstruction/centerline_extractor.py`

**Bug fixed**: The original code used `mesh.vertices[0]` and `mesh.vertices[-1]` as source/target points for the VMTK centerline algorithm. These are arbitrary surface vertices — not the actual inlet/outlet centers. In this case, `vertices[0]` was near the outlet (not the inlet) and `vertices[-1]` was near the inlet (not the outlet). They were **backwards AND arbitrary**.

**Fix**: Added `_detect_inlet_outlet()` method that:
1. Computes PCA on mesh vertices to find the principal axis
2. Projects all vertices onto the principal axis
3. Selects vertices within 1% of the projection range at each end
4. Returns the centroid of those end vertices as the inlet/outlet centers

**Output**: `centerline.npy` — 76 points, 2mm spacing, 151.43 mm total length

| Endpoint | X (mm) | Y (mm) | Z (mm) | X (m) | Y (m) | Z (m) |
|----------|--------|--------|--------|-------|-------|-------|
| Inlet    | 282.0  | 317.5  | 45.0   | 0.282 | 0.3175| 0.045 |
| Outlet   | 233.0  | 174.5  | 54.0   | 0.233 | 0.1745| 0.054 |

The centerline goes from inlet → outlet with 100% monotonic progression along the PCA axis.

### 1. STL preparation (mm → m)

The original STL is in mm at: `data_preproc/tbad_stl_output/tbad_TL_walls.stl`

The pipeline copies it to `constant/triSurface/tbad_TL_walls.stl` and scales to meters.
No normal fixing needed — snappyHexMesh handles inconsistent normals internally.

### 2. Volume mesh (snappyHexMesh)

Key settings:
- `locationInMesh (0.251947 0.229793 0.050520)` — first centerline point confirmed inside STL by trimesh (point #46 of 76; `trimesh.contains()` is unreliable on medical STLs, documented in PATIENT58_CFD_SUMMARY.md)
- `refinementSurfaces level (3 3)` — needed for full z-coverage
- `nCellsBetweenLevels 4` — smooth level transition
- `tolerance 4.0` — 4mm snap tolerance

Result: 25,585 cells, Mesh OK.

### 3. Patch splitting (INLET/OUTLET/WALL)

single `wall_aorta` patch split by z-coordinate:
- INLET: 376 faces (bottom 10% by z)
- OUTLET: 2512 faces (top 10% by z)
- WALL: 7705 faces (middle 80%)

### 4. CFD simulation (simpleFoam)

Boundary conditions:
- INLET: fixedValue U = (0.4 0 0) m/s
- OUTLET: fixedValue p = 0
- WALL: noSlip

Parameters:
- Kinematic viscosity: 3.77e-6 m²/s (blood)
- Density: 1060 kg/m³
- Reynolds number: ~4700 (transitional)
- Solver: simpleFoam (steady-state)
- Convergence: residual < 1e-5

## Files

```
patient58_cfd_example/
├── centerline.npy                    # Centerline points in meters (76 points)
├── PATIENT58_CFD_SUMMARY.md          # Technical notes and troubleshooting
├── README.md                         # This file
├── patient58_stl_3d.png              # 3D STL surface with centerline overlay
├── patient58_stl_projections.png     # Axial/sagittal/coronal projections with centerline
├── patient58_proj_axial.png          # Axial view (X-Y plane)
├── patient58_proj_sagittal.png       # Sagittal view (X-Z plane)
├── patient58_proj_coronal.png        # Coronal view (Y-Z plane)
├── patient58_centerline_overlay.png  # 3D semi-transparent STL with centerline
├── patient58_mesh_cross_section.png  # Mesh cross-section showing patches
├── patient58_post_processing.png     # Convergence + velocity profile
├── constant/
│   ├── polyMesh/                     # Volume mesh (points, faces, owner, neighbour, boundary)
│   ├── triSurface/tbad_TL_walls.stl  # Aortic wall STL (meters)
│   ├── transportProperties           # Newtonian blood (nu=3.77e-6 m²/s)
│   └── turbulenceProperties          # Laminar
├── system/
│   ├── blockMeshDict                 # Background mesh (0.15-0.40 × 0.10-0.41 × 0.00-0.09)
│   ├── snappyHexMeshDict             # STL surface refinement
│   ├── controlDict                   # simpleFoam, 100 iterations
│   ├── fvSchemes                     # Gauss linear schemes
│   └── fvSolution                    # PIMPLE, GAMG/smoothSolver
├── 0/
│   ├── U                             # Velocity field with BCs
│   └── p                             # Pressure field with BCs
├── scripts/
│   └── run_pipeline.py               # Full pipeline launcher
├── report/                           # foampilot post-processing output
│   ├── patient58_cfd_report.html     # Interactive HTML report with stats
│   ├── simulation_statistics.json    # Machine-readable convergence stats
│   ├── cell_data_500.csv             # Cell-center U, p at t=500
│   ├── INLET_data_500.csv            # Inlet patch field data
│   ├── OUTLET_data_500.csv           # Outlet patch field data
│   └── WALL_data_500.csv             # Wall patch field data
├── VTK/                              # VTK exports for ParaView
│   ├── patient58_cfd_example_500.vtk # Internal mesh (p, U)
│   ├── INLET/INLET_500.vtk
│   ├── OUTLET/OUTLET_500.vtk
│   └── WALL/WALL_500.vtk
└── [time directories 0/, 5/, 50/, 500/ with results]
```

## Simulation Results

- **Mesh**: 25,585 cells, 32,223 points — checkMesh PASSED
- **Convergence**: CONVERGED (all residuals ~1e-6, target 1e-5)
- **Final residuals**: Ux 9.99e-07, Uy 9.11e-07, Uz 9.73e-07
- **Reynolds number**: ~4700 (transitional)

## Running

```bash
# Source OpenFOAM
source /opt/openfoam13/etc/bashrc

# Run full pipeline (centerline → STL → mesh → patches → CFD → post)
PYTHONPATH=/home/steven/foampilot/foampilot/src \
    python3 scripts/run_pipeline.py --all

# Only generate mesh
PYTHONPATH=/home/steven/foampilot/foampilot/src \
    python3 scripts/run_pipeline.py --mesh-only

# Only run CFD simulation
PYTHONPATH=/home/steven/foampilot/foampilot/src \
    python3 scripts/run_pipeline.py --run-sim

# Only post-process
PYTHONPATH=/home/steven/foampilot/foampilot/src \
    python3 scripts/run_pipeline.py --post-process
```

Individual steps:
```bash
blockMesh
snappyHexMesh -overwrite -case .
simpleFoam -case .
checkMesh -case .
paraview -case .
```

### Post-processing (foampilot native)

Use the foampilot `OpenFOAMDirectReader` to read mesh + fields directly — no `foamToVTK` or `foamLog` required:

```bash
source /opt/openfoam13/etc/bashrc
PYTHONPATH=/home/steven/foampilot/foampilot/src \
    python3 scripts/postprocess_foampilot.py
```

This generates:
- `report/velocity_magnitude_3d.png` — 3D render with velocity coloring + centerline
- `report/pressure_3d.png` — 3D render with pressure coloring
- `report/velocity_glyphs.png` — velocity vector glyphs
- `report/wall_velocity.png` — wall velocity distribution
- `report/convergence_history.png` — residual convergence plot
- `report/postprocess_statistics.json` — machine-readable stats
- `report/field_data_500.csv` — per-point field data (X/Y/Z, U, p, U_mag)
