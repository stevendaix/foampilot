
# Simulation CFD d'un silencieux à l'aide de FoamPilot et ClassyBlocks

Ce document détaille un script Python utilisant les bibliothèques `foampilot`, `classy_blocks`, `pyvista` et `numpy` pour simuler un écoulement incompressible dans une géométrie de silencieux. Le script couvre la création de la géométrie, le maillage, la définition des conditions aux limites, la simulation et le post-traitement.

---

## Table des matières
1. [Introduction](#introduction)
2. [Importation des bibliothèques](#importation-des-bibliothèques)
3. [Initialisation du solveur et des répertoires](#initialisation-du-solveur-et-des-répertoires)
4. [Définition de la géométrie](#définition-de-la-géométrie)
5. [Génération du maillage](#génération-du-maillage)
6. [Gestion des conditions aux limites](#gestion-des-conditions-aux-limites)
7. [Exécution de la simulation](#exécution-de-la-simulation)
8. [Post-traitement](#post-traitement)
9. [Visualisation et analyse des résultats](#visualisation-et-analyse-des-résultats)
10. [Conclusion](#conclusion)

---

## Introduction
Ce script automatise la création d'une géométrie de silencieux, génère un maillage adapté, définit les conditions aux limites, exécute une simulation CFD (Computational Fluid Dynamics) avec OpenFOAM via `foampilot`, et effectue un post-traitement des résultats à l'aide de `pyvista`.

Cet exemple est disponible dans exemple/muffler

---

## Importation des bibliothèques

```python
from foampilot import incompressibleFluid, Meshing, commons, utilities, postprocess
import numpy as np
import classy_blocks as cb
import pyvista as pv
from pathlib import Path
from foampilot.utilities.manageunits import Quantity
```

- **foampilot** : Interface Python pour OpenFOAM, permettant de gérer les cas de simulation, le maillage, les conditions aux limites et le post-traitement.
- **numpy** : Utilisé pour les calculs numériques et la manipulation de tableaux.
- **classy_blocks** : Bibliothèque pour la création de géométries et de maillages structurés.
- **pyvista** : Bibliothèque de visualisation 3D pour les maillages et les champs de résultats.
- **Path** : Pour la gestion des chemins de fichiers.
- **Quantity** : Pour la gestion des unités physiques.

---

## Initialisation du solveur et des répertoires

```python
current_path = Path.cwd() / 'exemple2'
solver = incompressibleFluid(path_case=current_path)
system_dir = solver.system.write()
system_dir = solver.constant.write()
solver.system.fvSchemes.to_dict()
```

- `current_path` : Chemin vers le répertoire de travail où seront stockés les fichiers de la simulation.
- `solver` : Instance du solveur pour un fluide incompressible.
- `system_dir` : Écriture des répertoires `system` et `constant` nécessaires à OpenFOAM.
- `fvSchemes.to_dict()` : Affiche les schémas numériques utilisés pour la simulation.

---

## Définition de la géométrie

### Paramètres géométriques

```python
pipe_radius = 0.05      # Rayon du tuyau
muffler_radius = 0.08   # Rayon du silencieux
ref_length = 0.1        # Longueur de référence pour les segments
cell_size = 0.015       # Taille uniforme des cellules du maillage
shapes = []             # Liste pour stocker les formes géométriques
```

### Création des formes géométriques

1. **Cylindre d'entrée** :
   ```python
   shapes.append(cb.Cylinder([0, 0, 0], [3 * ref_length, 0, 0], [0, pipe_radius, 0]))
   shapes[-1].chop_axial(start_size=cell_size)
   shapes[-1].chop_radial(start_size=cell_size)
   shapes[-1].chop_tangential(start_size=cell_size)
   shapes[-1].set_start_patch("inlet")
   ```
   - Crée un cylindre de l'entrée, avec un maillage structuré axial, radial et tangentiel.
   - La face de départ est nommée `"inlet"`.

2. **Extension du cylindre** :
   ```python
   shapes.append(cb.Cylinder.chain(shapes[-1], ref_length))
   shapes[-1].chop_axial(start_size=cell_size)
   ```

3. **Anneau extrudé (début du silencieux)** :
   ```python
   shapes.append(cb.ExtrudedRing.expand(shapes[-1], muffler_radius - pipe_radius))
   shapes[-1].chop_radial(start_size=cell_size)
   ```

4. **Corps du silencieux** :
   ```python
   shapes.append(cb.ExtrudedRing.chain(shapes[-1], ref_length))
   shapes[-1].chop_axial(start_size=cell_size)
   ```

5. **Fin du silencieux** :
   ```python
   shapes.append(cb.ExtrudedRing.chain(shapes[-1], ref_length))
   shapes[-1].chop_axial(start_size=cell_size)
   ```

6. **Remplissage de l'anneau** :
   ```python
   shapes.append(cb.Cylinder.fill(shapes[-1]))
   shapes[-1].chop_radial(start_size=cell_size)
   ```

7. **Coudure (coude)** :
   ```python
   elbow_center = shapes[-1].sketch_2.center + np.array([0, 2 * muffler_radius, 0])
   shapes.append(cb.Elbow.chain(shapes[-1], np.pi / 2, elbow_center, [0, 0, 1], pipe_radius))
   shapes[-1].chop_axial(start_size=cell_size)
   shapes[-1].set_end_patch("outlet")
   ```
   - Crée un coude à 90° et nomme la face de sortie `"outlet"`.

---

## Génération du maillage

```python
mesh = cb.Mesh()
for shape in shapes:
    mesh.add(shape)
mesh.set_default_patch("walls", "wall")
mesh.write(current_path / "system" / "blockMeshDict", current_path / "debug.vtk")
```

- `cb.Mesh()` : Initialise un objet maillage.
- `mesh.add(shape)` : Ajoute chaque forme au maillage.
- `set_default_patch("walls", "wall")` : Définit les surfaces non nommées comme des parois.
- `mesh.write()` : Génère les fichiers `blockMeshDict` (pour OpenFOAM) et `debug.vtk` (pour la visualisation).

---

## Exécution du maillage avec OpenFOAM

```python
meshing = Meshing(path_case=current_path)
meshing.run_blockMesh()
```

- `Meshing` : Classe pour exécuter l'outil `blockMesh` d'OpenFOAM.
- `run_blockMesh()` : Génère le maillage à partir du fichier `blockMeshDict`.

---

## Gestion des conditions aux limites

### Initialisation

```python
solver.boundary.initialize_boundary()
```

### Conditions aux limites

1. **Vitesse d'entrée** :
   ```python
   solver.boundary.set_velocity_inlet(
       pattern="inlet",
       velocity=(Quantity(10,"m/s"), Quantity(0,"m/s"), Quantity(0,"m/s")),
       turbulence_intensity=0.05
   )
   ```
   - Vitesse d'entrée de 10 m/s selon l'axe x, avec une intensité turbulente de 5%.

2. **Pression de sortie** :
   ```python
   solver.boundary.set_pressure_outlet(
       pattern="outlet",
       velocity=(Quantity(10,"m/s"), Quantity(0,"m/s"), Quantity(0,"m/s")),
   )
   ```

3. **Paroi sans glissement** :
   ```python
   solver.boundary.set_wall(
       pattern="walls",
       velocity=(Quantity(0,"m/s"), Quantity(0,"m/s"), Quantity(0,"m/s"))
   )
   ```

### Écriture des fichiers

```python
fields = ["U", "p", "k", "epsilon", "nut"]
for field in fields:
    solver.boundary.write_boundary_file(field)
```

- Génère les fichiers de conditions aux limites pour les champs de vitesse (`U`), pression (`p`), énergie cinétique turbulente (`k`), dissipation turbulente (`epsilon`) et viscosité turbulente (`nut`).

---

## Exécution de la simulation

```python
solver.run_simulation()
```

- Lance la simulation CFD avec les paramètres définis.

---

## Post-traitement des résidus

```python
residuals_post = utilities.ResidualsPost(current_path / "log.incompressibleFluid")
residuals_post.process(export_csv=True, export_json=True, export_png=True, export_html=True)
```

- Extrait et exporte les résidus de la simulation dans différents formats.

---

## Visualisation et analyse des résultats

### Chargement des résultats

```python
foam_post = postprocess.FoamPostProcessing(case_path=current_path)
foam_post.foamToVTK()
time_steps = foam_post.get_all_time_steps()
latest_time_step = time_steps[-1]
structure = foam_post.load_time_step(latest_time_step)
cell_mesh = structure["cell"]
boundaries = structure["boundaries"]
```

- Convertit les résultats OpenFOAM au format VTK pour la visualisation.
- Charge le dernier pas de temps et les maillages des cellules et des frontières.

### Visualisation

1. **Tranche (slice)** :
   ```python
   pl_slice = pv.Plotter(off_screen=True)
   y_slice = cell_mesh.slice(normal='z')
   pl_slice.add_mesh(y_slice, scalars='U', lighting=False, scalar_bar_args={'title': 'U'})
   pl_slice.add_mesh(cell_mesh, color='w', opacity=0.25)
   for name, mesh in boundaries.items():
       pl_slice.add_mesh(mesh, opacity=0.5)
   foam_post.export_plot(pl_slice, current_path / "slice_plot.png")
   ```

2. **Contours de pression** :
   ```python
   pl_contour = pv.Plotter(off_screen=True)
   pl_contour.add_mesh(cell_mesh, scalars='p', show_scalar_bar=True)
   foam_post.export_plot(pl_contour, current_path / "contour_plot.png")
   ```

3. **Vecteurs de vitesse** :
   ```python
   pl_vectors = pv.Plotter(off_screen=True)
   cell_mesh.set_active_vectors('U')
   arrows = cell_mesh.glyph(orient='U', factor=0.001)
   pl_vectors.add_mesh(arrows, color='blue')
   foam_post.export_plot(pl_vectors, current_path / "vector_plot.png")
   ```

4. **Style de maillage** :
   ```python
   pl_mesh_style = pv.Plotter(off_screen=True)
   pl_mesh_style.add_mesh(cell_mesh, style='wireframe', show_edges=True, color='red')
   foam_post.export_plot(pl_mesh_style, current_path / "mesh_style_plot.png")
   ```

### Analyse

1. **Critère Q** :
   ```python
   mesh_with_q = foam_post.calculate_q_criterion(mesh=cell_mesh, velocity_field="U")
   ```

2. **Vorticité** :
   ```python
   mesh_with_vorticity = foam_post.calculate_vorticity(mesh=cell_mesh, velocity_field="U")
   ```

3. **Statistiques** :
   ```python
   pressure_stats = foam_post.get_scalar_statistics(mesh=cell_mesh, scalar_field="p")
   time_series = foam_post.get_time_series_data(scalar_field="p", point_coordinates=[0.0, 0.0, 0.0])
   mesh_stats = foam_post.get_mesh_statistics(cell_mesh)
   cell_region_stats = foam_post.get_region_statistics(structure, "cell", "U")
   ```

4. **Export des données** :
   ```python
   foam_post.export_region_data_to_csv(structure, "cell", ["U", "p"], current_path / "cell_data.csv")
   foam_post.export_statistics_to_json(all_stats, current_path / "all_stats.json")
   foam_post.create_animation(scalars='U', filename='animation_test.gif', fps=5)
   ```

---

## Conclusion

Ce script illustre une chaîne complète de simulation CFD :
- Création de géométrie et maillage structuré avec `classy_blocks`.
- Simulation avec OpenFOAM via `foampilot`.
- Post-traitement et visualisation avec `pyvista`.

Les résultats incluent des visualisations 2D/3D, des statistiques sur les champs, et des animations. Le code est modulaire et peut être adapté à d'autres géométries ou conditions de simulation.

---

## Pour aller plus loin

- **Personnalisation** : Modifier les paramètres géométriques ou les conditions aux limites pour étudier d'autres configurations.
- **Optimisation** : Utiliser les résultats pour optimiser la géométrie du silencieux.
- **Automatisation** : Intégrer ce script dans un pipeline de simulation plus large.

---
