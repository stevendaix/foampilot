# Post-traitement 3 : Visualisation des champs avec PyVista / ParaView

## Objectif
Inspecter visuellement `U`, `p`, `k`, `epsilon` dans le domaine fluide.

## Avec PyVista
```python
import pyvista as pv

# Charge un champ à un instant donné
mesh = pv.read("/tmp/voxcity_vector_demo3/2000/U.vtk")

# Glyphes de vitesse
mesh["Umag"] = np.linalg.norm(mesh["U"], axis=1)

# Plan de coupe
slice_y = mesh.slice(normal="y", origin=(0, 0, 0))

p = pv.Plotter()
p.add_mesh(slice_y, scalars="Umag", cmap="viridis")
p.add_title("Vitesse — plan y=0")
p.show()
```

## Avec ParaView
- Ouvrir le cas `voxcity_vector_demo3`.
- Utiliser `Glyph` pour les vecteurs `U`.
- `Clip` ou `Slice` pour les coupes.
- `Plot Over Line` pour des profils.

## Pièges
- PyVista en mode off-screen peut produire des images noires si le backend n'est pas bien configuré.
- Préférer ParaView pour des rendus fiables.
