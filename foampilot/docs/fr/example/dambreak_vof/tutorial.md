# DamBreak VOF — Tutoriel FoamPilot

## Aperçu

Ce tutoriel simule le **balancement d'une colonne d'eau** dans un domaine rectangulaire 2D en utilisant le modèle **VOF (Volume of Fluid)** et le solveur `interFoam`.

FoamPilot automatise :

- la configuration de la fraction volumique VOF (`alpha.water`)
- les propriétés matériaux diphasiques
- l'activation de la gravité

📁 **Emplacement**: `foampilot/tutorials/04_damBreak_multiphase/`

---

## 1. Prérequis

- OpenFOAM installé
- FoamPilot installé

---

## 2. Physique du cas

- **Domaine** : cuve rectangulaire 2D (5 m × 2 m × 0.1 m)
- **Phases** : eau (alpha = 1) et air (alpha = 0)
- **Modèle VOF** : Volume of Fluid pour le suivi d'interface
- **Gravité** : active (9.81 m/s², direction -Y)
- **Turbulence** : laminaire (faible Re)

### 2.1 Équation de transport VOF

$$
\frac{\partial \alpha}{\partial t} + \nabla \cdot (\mathbf{u} \, \alpha) = 0
$$

La fraction volumique `α` = 1 dans l'eau, 0 dans l'air, 0–1 à l'interface.

### 2.2 Équations de quantité de mouvement

$$
\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \mathbf{u}) = -\nabla p + \mu \nabla^2 \mathbf{u} + \rho \mathbf{g} + \sigma \kappa \nabla \alpha
$$

Où :
- `σ` — coefficient de tension de surface
- `κ` — courbure de l'interface
- `g` — vecteur gravité

### 2.3 Conditions initiales

- Colonne d'eau : 2 m × 1 m à gauche du domaine
- Reste du domaine : rempli d'air
- Vitesse nulle partout initialement

---

## 3. Flux de travail

### 3.1 Initialisation du solveur

```python
from foampilot.solver import Solver
from foampilot import ValueWithUnit

solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = False
solver.is_vof = True
solver.turbulence_model = "laminar"
```

La définition `solver.is_vof = True` active automatiquement :

- le solveur `interFoam`
- la configuration de `transportProperties` diphasique
- la création du champ `alpha.water`

### 3.2 Conditions aux limites

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
)
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet",
)
solver.boundary.apply_condition_with_wildcard(
    pattern="walls",
    condition_type="wall",
)
```

### 3.3 Exécution

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. Post-traitement

### 4.1 Suivi de l'interface

Le champ VOF `alpha.water` suit l'interface eau-air :

```python
from foampilot import postprocess

foam_post = postprocess.FoamPostProcessing(case_path=case_path)
foam_post.foamToVTK()
```

### 4.2 Visualisation

```python
import pyvista as pv

mesh = pv.read("VTK/0/cellular.vtk")
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh, scalars="alpha.water", cmap="Blues")
plotter.screenshot("dam_break_interface.png")
```

### 4.3 Génération de rapport

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="DamBreak VOF Simulation Report",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_abstract("Two-phase flow simulation using VOF model.")
doc.add_equation(r"\frac{\partial \alpha}{\partial t} + \nabla \cdot (\mathbf{u} \alpha) = 0")
doc.add_section("Interface Evolution")
doc.add_figure("dam_break_interface.png", "Water-air interface at t=2.0s")
doc.generate_document(output_format="pdf")
```

---

## 5. Résultats attendus

| Quantité | Attendu |
|----------|---------|
| Vitesse du front d'eau | ~4.4 m/s (√(2gh), h=1m) |
| Temps pour atteindre la paroi droite | ~3 s |
| Réflexion d'onde | Visible après l'impact |

---

## 6. Exécution

```bash
cd foampilot/tutorials/04_damBreak_multiphase
python run.py
python report_generator.py
```
