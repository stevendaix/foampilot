# Post-traitement 5 : Qualité du maillage après simulation

## Objectif
S'assurer que le maillage reste valide après déformation ou adaptation, et identifier les zones problématiques.

## Méthode
- Relancer `checkMesh` sur le cas final.
- Extraire `nonOrthogonality`, `skewness`, `aspect ratio`.
- Visualiser les cellules de mauvaise qualité avec PyVista.

## Code minimal
```python
import meshio

points, cells, *_ = meshio.read("/tmp/voxcity_vector_demo3/constant/polyMesh/points"), ...
# Plus simple : utiliser OpenFOAM directement
import subprocess
result = subprocess.run(
    ["checkMesh", "-case", "/tmp/voxcity_vector_demo3"],
    capture_output=True, text=True
)
print(result.stdout)
```

## Points clés
- Max non-orthogonalité < 70° pour `simpleFoam` / `incompressibleFluid`.
- Skewness < 1 pour des tétraèdres, < 2 sinon.
- Si des cellules sont dégradées, raffiner localement avec `gmsh` et réexporter.

## Automatisation
On pourra intégrer ces vérifications dans `voxcity_vector_example.py` pour rejeter un maillage non conforme automatiquement.
