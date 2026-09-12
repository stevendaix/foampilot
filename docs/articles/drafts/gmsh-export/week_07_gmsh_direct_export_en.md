# Direct Gmsh to OpenFOAM Export: Writing polyMesh Without gmshToFoam

*How foampilot writes `points`, `faces`, `owner`, `neighbour`, `boundary`, and `cellZones` directly from the Gmsh Python API, skipping the `gmshToFoam` step entirely.*

---

## Introduction: The gmshToFoam Bottleneck

If you've ever used Gmsh with OpenFOAM, you know the drill:

1. Build geometry in Gmsh
2. Mesh with `gmsh.model.mesh.generate(3)`
3. Export to `.msh`: `gmsh.write("case.msh")`
4. Run `gmshToFoam case.msh`
5. Cross your fingers

Step 4 is the bottleneck. `gmshToFoam` is an external utility that:
- Depends on your OpenFOAM version
- Can lose information about physical groups
- Can silently misorient faces
- Is impossible to debug programmatically

**The solution**: write `constant/polyMesh` directly from the Gmsh Python API, without an intermediate `.msh` file. This is exactly what foampilot's `direct_openfoam_exporter.py` module does.

---

## Part 1: Classic Pipeline vs Direct Pipeline

### Classic pipeline

```
Gmsh GUI/API → .msh file → gmshToFoam → constant/polyMesh → OpenFOAM
```

**Problems**:
- Large intermediate `.msh` file (especially for big meshes)
- Dependency on `gmshToFoam` (only available if OpenFOAM is sourced)
- Silent errors on face orientations
- No programmatic control over physical group mapping

### Direct pipeline

```
Gmsh API → direct_openfoam_exporter → constant/polyMesh → OpenFOAM
```

**Advantages**:
- No intermediate file
- No dependency on `gmshToFoam`
- Total control over face orientation
- Debuggable Python logic

---

## Part 2: The Direct Export Algorithm

### 2.1 Basic usage

```python
import gmsh
from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter

gmsh.initialize()
gmsh.model.add("poiseuille_case")

# 1. Build geometry
# ... points, lines, surfaces, volumes ...

# 2. Mesh
gmsh.model.mesh.generate(3)

# 3. Export directly to OpenFOAM
exporter = DirectOpenFOAMExporter("/path/to/case")
exporter.export_single_region()

gmsh.finalize()
```

That's it. The `constant/polyMesh/` directory is created with all necessary files.

### 2.2 Generated files

The export writes 6 files:

| File | Content |
|------|---------|
| `points` | Node coordinates |
| `faces` | Face connectivity |
| `owner` | Owner of each face |
| `neighbour` | Neighbour of each internal face |
| `boundary` | Patch definitions |
| `cellZones` | Cell zones (optional) |

---

## Part 3: Algorithmic Challenges

### 3.1 Tetrahedron face orientation

For a tetrahedron with nodes (n0, n1, n2, n3), the 4 outward-facing faces are:

```
Face opposite n0: (n1, n2, n3)
Face opposite n1: (n0, n3, n2)
Face opposite n2: (n0, n1, n3)
Face opposite n3: (n0, n2, n1)
```

**Challenge**: for each face, determine if it's internal (shared by two cells) or external (belonging to a patch), and in the internal case, who is the owner and who is the neighbour.

### 3.2 Internal face detection

The algorithm uses a dictionary to count face occurrences:

```python
# Canonical key: sorted nodes (order-independent)
face_key = tuple(sorted(face_nodes))

# Count occurrences
face_count[face_key] += 1
cell_id[face_key] = current_cell

# If a face appears 2 times → internal
# If a face appears 1 time → external (boundary)
```

### 3.3 Upper-triangular sorting

OpenFOAM requires internal faces to be sorted by `(owner, neighbour)` with cyclic rotation so the minimum vertex is first:

```python
def _to_upper_triangular(face_nodes):
    """Cyclically rotate face_nodes so the minimum vertex is first."""
    if not face_nodes:
        return face_nodes
    min_idx = face_nodes.index(min(face_nodes))
    return face_nodes[min_idx:] + face_nodes[:min_idx]
```

This rotation preserves face orientation while satisfying OpenFOAM's constraint.

### 3.4 Point compaction

Gmsh nodes are not contiguous (their tags can be 1, 5, 12, 100…). The export must:

1. Create a mapping `Gmsh_tag → OpenFOAM_index` (0-based)
2. Remove unused points
3. Update all cell connectivities

```python
# Mapping
node_mapping = {}
for i, tag in enumerate(used_node_tags):
    node_mapping[tag] = i

# Rewrite connectivity
new_cells = [[node_mapping[node] for node in cell] for cell in cells]
```

---

## Part 4: Multi-Region CHT

For Conjugate Heat Transfer (CHT) simulations, each physical volume is written to its own directory:

```python
exporter.export_multi_region(region_map={
    "air": "fluid",
    "copper": "solid"
})
```

Result:
```
constant/
├── fluid/
│   └── polyMesh/
│       ├── points
│       ├── faces
│       ├── owner
│       └── boundary
├── solid/
│   └── polyMesh/
│       ├── points
│       ├── faces
│       ├── owner
│       └── boundary
```

Each region has its own mesh, patches, and boundary conditions.

---

## Part 5: Performance and Validation

### 5.1 Performance

| Number of cells | Export time | Memory |
|-----------------|-------------|--------|
| 10 000 tetrahedra | ~0.1s | ~50 MB |
| 1 000 000 tetrahedra | ~2s | ~500 MB |
| 5 000 000 tetrahedra | ~10s | ~2.5 GB |

No intermediate `.msh` file = disk space savings.

### 5.2 Validation

Export is validated with `checkMesh`:

```bash
checkMesh -case /path/to/case
```

Checked points:
- No degenerate cells
- Correctly oriented faces
- `nonOrtho` < 75 (default)
- `skewness` < 2 (default)

---

## Part 6: Advanced Use Cases

### 6.1 Physical group extraction

```python
# Get all physical volumes
vol_groups = gmsh.model.getPhysicalGroups(dim=3)
for dim, tag in vol_groups:
    name = gmsh.model.getPhysicalName(dim, tag)
    print(f"Region: {name}")
```

### 6.2 Automatic patch assignment

```python
# Classify 2D faces by normal
patch_map = gmsh_helper.classify_patch_by_normal(angle_tol=15.0)
for patch_name, face_tags in patch_map.items():
    gid = gmsh.model.addPhysicalGroup(2, face_tags)
    gmsh.model.setPhysicalName(2, gid, patch_name)
```

### 6.3 Local mesh refinement

```python
# Refine around a point
gmsh.model.mesh.setSize(
    gmsh.model.getEntitiesInBoundingBox(x-r, y-r, z-r, x+r, y+r, z+r),
    lc_refined
)
```

---

## Conclusion: Removing a Weak Link

Direct Gmsh → OpenFOAM export is not just a convenience optimization. It's an **error surface reduction**:

- No lossy conversion
- No external dependency
- Direct Python debugging
- Total control over orientation and groups

In the next article, I'll show you how to read these meshes directly in PyVista, without `foamToVTK`, and extract the physical quantities you care about.

**Resources:**
- Export module: [`foampilot/mesh/direct_openfoam_exporter.py`](https://github.com/stevendaix/foampilot/blob/main/foampilot/src/foampilot/mesh/direct_openfoam_exporter.py)
- Tests: [`test/test_direct_openfoam_export.py`](https://github.com/stevendaix/foampilot/blob/main/foampilot/test/test_direct_openfoam_export.py)
- Documentation: [stevendaix.github.io/foampilot](https://stevendaix.github.io/foampilot/)

---

*Article under improvement — version 1.0*
