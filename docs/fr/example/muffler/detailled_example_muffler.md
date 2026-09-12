# Exemple CFD Silencieux – FoamPilot

## Vue d'ensemble

Cet exemple démontre un **workflow CFD complet** utilisant **FoamPilot** et **OpenFOAM** pour simuler un écoulement incompressible dans une géométrie de silencieux. Il sert de **référence** pour illustrer la philosophie de FoamPilot :

- Modélisation physique explicite (fluides, unités)
- Géométrie paramétrique et maillage structuré
- Gestion robuste des conditions aux limites
- Exécution automatisée des simulations
- Post-traitement et visualisation avancés
- Génération automatique de rapports PDF

📁 **Emplacement** : `examples/muffler`

---

## 1. Prérequis

Avant d'exécuter cet exemple, assurez-vous que :

- OpenFOAM est correctement installé et accessible
- FoamPilot est installé
- Les dépendances Python suivantes sont disponibles :
  - `classy_blocks`
  - `pyvista`
  - `numpy`
  - `pandas`

---

## 2. Initialisation du cas

Définition du répertoire de travail et initialisation du solveur FoamPilot.

```python
from foampilot.solver import Solver
from pathlib import Path

current_path = Path.cwd() / "cas_test"
solver = Solver(current_path)

solver.compressible = False
solver.with_gravity = False
```

Le `Solver` est l'objet central qui orchestre :

- Les dictionnaires OpenFOAM
- Les conditions aux limites
- L'exécution de la simulation

---

## 3. Propriétés du fluide

FoamPilot utilise l'API `FluidMechanics` pour définir explicitement les fluides.

```python
from foampilot import FluidMechanics, ValueWithUnit

fluides_disponibles = FluidMechanics.get_available_fluids()

fluide = FluidMechanics(
    fluides_disponibles["Water"],
    temperature=ValueWithUnit(293.15, "K"),
    pressure=ValueWithUnit(101325, "Pa")
)

proprietes = fluide.get_fluid_properties()
nu = proprietes["kinematic_viscosity"]
```

La viscosité cinématique est ensuite injectée dans la configuration OpenFOAM :

```python
solver.constant.transportProperties.nu = nu
```

---

## 4. Écriture des fichiers de configuration OpenFOAM

```python
solver.system.write()
solver.constant.write()
solver.system.fvSchemes.to_dict()
```

FoamPilot génère automatiquement :

- `controlDict`
- `fvSchemes`
- `fvSolution`
- `transportProperties`

---

## 5. Définition de la géométrie (ClassyBlocks)

### 5.1 Paramètres géométriques

```python
pipe_radius = 0.05
muffler_radius = 0.08
ref_length = 0.1
cell_size = 0.015
```

### 5.2 Construction de la géométrie

La géométrie est construite comme une **séquence de formes paramétriques** :

1. Tuyau d'entrée (cylindre)
2. Anneau d'expansion (corps du silencieux)
3. Section remplie
4. Coude de sortie à 90°

Exemple pour le cylindre d'entrée :

```python
import classy_blocks as cb

shapes = []

shapes.append(cb.Cylinder(
    [0, 0, 0],
    [3 * ref_length, 0, 0],
    [0, pipe_radius, 0]
))

shapes[-1].chop_axial(start_size=cell_size)
shapes[-1].chop_radial(start_size=cell_size)
shapes[-1].chop_tangential(start_size=cell_size)
shapes[-1].set_start_patch("inlet")
```

Les patches sont définis **au niveau de la géométrie**, garantissant la cohérence entre le maillage et les conditions aux limites.

---

## 6. Génération du maillage

```python
mesh = cb.Mesh()
for shape in shapes:
    mesh.add(shape)

mesh.set_default_patch("walls", "wall")
mesh.write(
    current_path / "system" / "blockMeshDict",
    current_path / "debug.vtk"
)
```

Exécution du maillage avec OpenFOAM :

```python
from foampilot import Meshing

meshing = Meshing(current_path, mesher="blockMesh")
meshing.mesher.run()
```

---

## 7. Conditions aux limites

FoamPilot fournit une API générique utilisant des motifs (wildcards) :

```python
solver.boundary.initialize_boundary()
```

### 7.1 Entrée de vitesse

```python
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(
        ValueWithUnit(10, "m/s"),
        ValueWithUnit(0, "m/s"),
        ValueWithUnit(0, "m/s")
    ),
    turbulence_intensity=0.05
)
```

### 7.2 Sortie de pression

```python
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet"
)
```

### 7.3 Parois

```python
solver.boundary.apply_condition_with_wildcard(
    pattern="walls",
    condition_type="wall"
)
```

### 7.4 Écriture des fichiers de conditions

```python
solver.boundary.write_boundary_conditions()
```

---

## 8. Exécution de la simulation

```python
solver.run_simulation()
```

FoamPilot gère automatiquement le solveur, l'exécution et le logging.

---

## 9. Post-traitement des résidus

```python
from foampilot.utilities import ResidualsPost

residuals = ResidualsPost(current_path / "log.incompressibleFluid")
residuals.process(
    export_csv=True,
    export_json=True,
    export_png=True,
    export_html=True
)
```

---

## 10. Visualisation et analyse des résultats

### Chargement des résultats

```python
from foampilot import postprocess

foam_post = postprocess.FoamPostProcessing(case_path=current_path)
foam_post.foamToVTK()
time_steps = foam_post.get_all_time_steps()
latest_time_step = time_steps[-1]
structure = foam_post.load_time_step(latest_time_step)
cell_mesh = structure["cell"]
boundaries = structure["boundaries"]
```

### Visualisation

- Tranches (slice)
- Contours de pression
- Vecteurs de vitesse
- Maillage filaire

### Analyses

- Critère Q
- Vorticité
- Statistiques du maillage et des champs
- Export CSV / JSON
- Animation

```python
foam_post.calculate_q_criterion(mesh=cell_mesh, velocity_field="U")
foam_post.calculate_vorticity(mesh=cell_mesh, velocity_field="U")
foam_post.create_animation(
    scalars="U",
    filename=current_path / "animation.gif",
    fps=5
)
```

---

## 11. Génération automatique du rapport PDF

```python
from foampilot import latex_pdf

doc = latex_pdf.LatexDocument(
    title="Rapport Simulation : Silencieux",
    author="Rapport Automatique",
    output_dir=current_path
)

doc.add_title()
doc.add_toc()
doc.add_abstract(
    "Ce rapport résume la simulation CFD incompressible d'un silencieux."
)

doc.generate_document(output_format="pdf")
```

Le rapport inclut :

- Propriétés du fluide
- Statistiques du maillage
- Statistiques des champs
- Figures et visualisations
- Annexes de données

---

## 12. Résumé

Cet exemple illustre une **chaîne complète de simulation CFD** :

- Géométrie paramétrique et maillage structuré avec `classy_blocks`
- Simulation CFD avec OpenFOAM via `foampilot`
- Post-traitement et visualisation avancés avec `pyvista`
- Reporting PDF automatisé

