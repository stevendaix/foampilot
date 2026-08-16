# CAD Reconstruction TBAD

Pipeline VMTK-like pour reconstruire la géométrie CAD de cas TBAD à partir de STL, puis mailler et exporter vers OpenFOAM.

## Installation

```bash
pip install numpy scipy trimesh vtk gmsh geomdl
```

## Utilisation rapide

```python
from pathlib import Path
from cad_reconstruction import CADReconstruction

stl = Path("tbad_TL_walls.stl")
case_dir = Path("patient58")
recon = CADReconstruction(case_dir=case_dir, centerline_spacing_mm=2.0)
result = recon.run(stl)
print(result)
```

## Pipeline

1. **Centerlines** : extraction Voronoi/Dijkstra
2. **Sections** : coupes perpendiculaires
3. **B-spline** : fitting des contours
4. **CAD** : loft OCC
5. **Maillage** : Gmsh volume
6. **Export** : OpenFOAM polyMesh

## Tests

```bash
pytest cad_reconstruction/test_validation.py -v
```
