# Post-traitement 2 : Coefficients de pression sur les façades

## Objectif
Calculer Cp sur les faces des bâtiments pour identifier les zones de surpression et dépression.

## Formule
Cp = (p - p_ref) / (0.5 * rho * U_inf^2)

## Méthode
1. Extraire les faces du patch `buildings` avec `foamToVTK` ou `sample`.
2. Récupérer `p` et la normale de surface.
3. Projeter la vitesse locale sur la normale pour obtenir la pression dynamique.

## Code minimal
```python
import pyvista as pv
import numpy as np

mesh = pv.read("/tmp/voxcity_vector_demo3/2000/p.vtk")
buildings = mesh.extract_geometry()

# Centre des cellules
centers = buildings.cell_centers()
p = centers["p"]

# Paramètres
rho = 1.225
U_inf = 10.0
p_ref = 0.0

Cp = (p - p_ref) / (0.5 * rho * U_inf**2)

centers["Cp"] = Cp
centers.plot(cmap="RdBu_r", scalars="Cp", title="Cp sur les façades")
```

## À surveiller
- Les gradients de Cp près des arêtes vives : besoin de raffinement local ?
- La symétrie du Cp sur les façades opposées pour un écoulement quasi-stationnaire.
