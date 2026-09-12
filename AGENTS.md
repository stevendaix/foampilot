# AGENTS.md

## Project Overview

foampilot is a Python library for automating OpenFOAM case setup, meshing,
simulation, and post-processing. It wraps OpenFOAM utilities and case files
in a Pythonic API, and integrates Gmsh for geometry generation and meshing.

## Environment

- Python 3.10+ (`/home/steven/venv/bin/python3`)
- Gmsh 4.14 (pip package + CLI binary)
- OpenFOAM 13 (`/opt/openfoam13`)

## Common Commands

### Run tests

```bash
cd /home/steven/foampilot/foampilot
PYTHONPATH=src python3 -m pytest test/ -v
```

### Run the direct Gmsh-to-OpenFOAM export tests

```bash
cd /home/steven/foampilot/foampilot
PYTHONPATH=src python3 -m pytest test/test_direct_openfoam_export.py -v
```

### Check OpenFOAM mesh quality

```bash
checkMesh -case /path/to/case [-region <regionName>]
```

### Generate the Gmsh mesh and write directly to polyMesh

```python
import gmsh
from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter

gmsh.initialize()
gmsh.model.add("case")
# ... build geometry, assign physical groups, mesh.generate(3) ...
DirectOpenFOAMExporter("/path/to/case").export_single_region()
# or for CHT:
DirectOpenFOAMExporter("/path/to/case").export_multi_region()
gmsh.finalize()
```

## Direct Export Module

The `direct_openfoam_exporter` module (`foampilot/mesh/direct_openfoam_exporter.py`)
provides:

- `DirectOpenFOAMExporter.export_single_region()` — writes
  `constant/polyMesh/` (points, faces, owner, neighbour, boundary, cellZones)
  from the current Gmsh model without invoking `gmshToFoam`.

- `DirectOpenFOAMExporter.export_multi_region(region_map=None)` — writes one
  `constant/<region>/polyMesh/` directory per 3-D physical group, suitable
  for conjugate heat-transfer (CHT) cases with `chtMultiRegionFoam`.

Key design notes:

- Gmsh node tags are remapped to contiguous 0-based OpenFOAM point
  indices.
- Tetrahedron faces are extracted from cell connectivity and oriented
  outward using a geometric centroid–normal check.
- Internal faces are sorted by (owner, neighbour) and cyclically rotated
  so the minimum vertex is first ("upper-triangular" ordering required
  by OpenFOAM).
- Boundary faces are matched to surface physical groups for patch naming.
- Unused points are compacted away per region.
