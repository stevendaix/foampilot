# Plan: Fix Gmsh Volume Creation from Medical STL

## Root Cause

The STL from `marching_cubes` (step 1) has ~260,560 triangles with **degraded quality**
(min edge = 4e-6 mm, min gamma = 1.5e-11). This causes **all** of these Gmsh operations to fail:
- `classifySurfaces` → produces 1 discrete surface
- `createTopology()` → produces no volume entity
- `createGeometry()` → "Invalid exterior boundary mesh for parametrization"
- `addSurfaceLoop` → "Unknown OpenCASCADE surface with tag X"

Decimation with `pyfqmr` **worsens** the problem (introduces non-manifold edges).
Trimesh smoothing doesn't improve the worst triangles.

## Strategy

1. **Remesh the STL surface using VTK** for high-quality triangle generation
2. **Attempt Gmsh OCC volume creation** on the clean mesh
3. **Fallback to snappyHexMesh** if Gmsh still fails

## Implementation

### Phase 1: Add `remesh_stl_with_vtk()` to `mesh_utils.py`

New function that uses VTK's pipeline:
- `vtkPolyDataReader` → read STL
- `vtkCleanPolyData` → remove duplicate vertices
- `vtkTriangleFilter` → ensure all triangles
- `vtkDecimatePro` → reduce face count (target 15k–20k)
- `vtkWindowedSincPolyDataFilter` → high-quality smoothing (preserves shape)
- `vtkPolyDataNormals` → fix winding
- Export: `.stl` → `.vtp` (VTK PolyData) for Gmsh import

**Why VTK?** Its `vtkDecimatePro` with `PreserveTopologyOn` and `SplittingOn`
produces manifold, water-tight meshes unlike `pyfqmr` which breaks topology.
`vtkWindowedSincPolyDataFilter` produces far better triangle quality than
trimesh's Taubin smoother.

### Phase 2: Update `step3_mesh_gmsh()` in `run_full_pipeline.py`

Replace the STL preprocessing block with:

```python
from mesh_utils import remesh_stl_with_vtk

clean_stl = case_dir / "stl_vtk_clean.stl"
remesh_stl_with_vtk(stl_path, clean_stl, target_faces=20000)

# Import into Gmsh
gmsh.merge(str(clean_stl))
gmsh.model.mesh.classifySurfaces(angle=360*pi/180, exportDiscrete=True)
gmsh.model.mesh.createTopology()
gmsh.model.occ.synchronize()

# Check for volumes
volumes = gmsh.model.getEntities(dim=3)
if volumes:
    # Direct volume — mesh it
    ...
else:
    # Try createGeometry
    try:
        gmsh.model.mesh.createGeometry()
        ...
    except:
        # Fallback: snappyHexMesh
        ...
```

### Phase 3: snappyHexMesh fallback (already partially implemented)

If Gmsh OCC fails:
- Use decimated STL (already works for snappy)
- Set `locationInMesh` from centerline midpoint
- Set refinement levels based on `lc_min`/`lc_max`
- Add boundary layers via `add_layer()`

### Phase 4: Image generation at each step

Add PNG visualization after each step:
- **Step 1**: 3D scatter of STL vertices + centerline overlay
- **Step 2**: Loft surface with centerline path
- **Step 3**: Mesh visualization (surface + volume cells) or snappyHexMesh result

Use `matplotlib` (already in deps) with `mpl_toolkits.mplot3d` for quick renders.

## Validation Steps

1. `remesh_stl_with_vtk()`: verify output is watertight, no degenerate triangles
2. Gmsh volume creation: verify `getEntities(dim=3)` returns ≥1 volume
3. snappyHexMesh: verify mesh has wall_aorta patch
4. `checkMesh`: verify topology and geometry OK
5. Tests: run `test_direct_openfoam_export.py` + existing test suite

## File Changes

| File | Change |
|------|--------|
| `mesh_utils.py` | Add `remesh_stl_with_vtk()` function |
| `run_full_pipeline.py` | Rewrite step 3 preprocessing + add image generation |
| `test/test_vtk_remesh.py` | New test for VTK remeshing |

## Timeline

1. **Phase 1**: 5 min — implement + test `remesh_stl_with_vtk()`
2. **Phase 2**: 10 min — integrate into step 3, test Gmsh volume creation
3. **Phase 3**: 10 min — finalize snappyHexMesh fallback
4. **Phase 4**: 5 min — add image generation
5. **Validation**: 10 min — run tests + pipeline
