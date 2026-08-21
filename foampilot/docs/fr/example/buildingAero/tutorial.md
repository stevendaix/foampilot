# Aérodynamique des bâtiments — Tutoriel FoamPilot

## Aperçu

Ce tutoriel simule le **flux turbulent autour des bâtiments** dans un environnement urbain en utilisant `simpleFoam` (k-omega SST). Il montre la manipulation avancée du maillage à l'aide de `topoSet` et `createPatch`.

FoamPilot automatise :

- `topoSet` et `createPatch` (exécution)
- Conditions aux limites urbaines (écoulement libre, bâtiments)
- Analyse des charges dues au vent

📁 **Emplacement**: `foampilot/tutorials/06_buildingAero/`

---

## 1. Prérequis

- OpenFOAM installé
- FoamPilot installé

---

## 2. Physique du cas

- **Domaine** : canyon urbain avec plusieurs bâtiments (10 m × 10 m × 3 m)
- **Écoulement** : Incompressible, turbulent, stationnaire
- **Vitesse d'entrée** : 10 m/s (50% urban turbulence intensity)
- **Modèle de turbulence** : k-omega SST
- **Gravité** : désactivée (écoulement poussé par pression, RANS stationnaire)

### 2.1 Profil de la couche limite urbaine

Vitesse d'entrée (profil logarithmique) :

$$
u(y) = u_* \frac{\ln(y / y_0)}{\kappa}
$$

Où :
- `u*` — vitesse de frottement
- `κ` — constante de von Kármán (0.41)
- `y0` — hauteur de rugosité

FoamPilot simplifie ceci avec `velocityInlet` et `turbulence_intensity` :

```python
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(10, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    turbulence_intensity=0.15,
)
```

### 2.2 Effet du vent dans le canyon

Les bâtiments génèrent des **vortex de canyon urbain** en aval. Le rapport d'aspect du canyon (hauteur du bâtiment / largeur de la rue) détermine le régime d'écoulement :

$$
AR = \frac{H_{building}}{W_{street}}
$$

Pour AR ≈ 1 (ce tutoriel), l'écoulement est dans le régime "critique" avec une forte recirculation à l'intérieur du canyon.

---

## 3. Flux de travail

### 3.1 Initialisation du solveur

```python
solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = False
solver.turbulence_model = "kOmegaSST"
```

### 3.2 Manipulation du maillage avec topoSet + createPatch

FoamPilot encapsule les outils de topologie d'OpenFOAM :

```python
# topoSet pour définir les ensembles de cellules des bâtiments
solver.system.run_topoSet()

# createPatch pour renommer les patches de frontière
solver.system.run_createPatch()
```

Cela exécute :

- `system/topoSetDict` — définit des ensembles de cellules/zones pour les bâtiments
- `system/createPatchDict` — renomme les faces en patches nommés

### 3.3 Conditions aux limites

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(10, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    turbulence_intensity=0.15,
)
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet",
)
solver.boundary.apply_condition_with_wildcard(
    pattern=".*building.*",
    condition_type="wall",
)
solver.boundary.write_boundary_conditions()
```

### 3.4 Exécution

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. Post-traitement

### 4.1 Visualisation de l'écoulement dans le canyon

PyVista visualization of velocity field:

```python
import pyvista as pv
from pathlib import Path

mesh = pv.read(str(Path("VTK/latest/cellular.vtk")))
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh.slice("z"), scalars="U", cmap="viridis")
plotter.screenshot("canyon_velocity.png")
```

### 4.2 Génération de rapport

```python
from foampilot.report.latex_pdf import LatexDocument
from foampilot.report.report_generator import CFDReportGenerator

# Rapport HTML
report = CFDReportGenerator(
    case_path=case_path,
    title="Rapport d'aérodynamique des bâtiments",
    author="FoamPilot",
)
report.add_statistic("U_inlet", 10.0, "m/s", "Vitesse d'entrée")
report.add_statistic("I_inlet", 0.15, "", "Intensité de turbulence")
report.save_html_report(filename="building_report.html")

# Rapport LaTeX
doc = LatexDocument(
    title="Aérodynamique des bâtiments urbains",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_abstract("Simulation de l'écoulement du vent autour de bâtiments urbains.")
doc.add_section("Écoulement dans le canyon", "")
doc.add_figure("canyon_velocity.png", "Champ de vitesse dans le canyon urbain")
doc.generate_document(output_format="pdf")
```

---

## 5. Résultats attendus

| Quantité | Attendu |
|----------|---------|
| Accélération du vent au niveau des toits | 1.2–1.5× U_inlet |
| Zone de recirculation dans le canyon | Visible derrière chaque bâtiment |
| Coefficient de pression Cp | -0.5 to +1.0 |
| Vitesse au niveau des piétons | < 0.2 U_inlet |

---

## 6. Exécution

```bash
cd foampilot/tutorials/06_buildingAero
python run.py
python report_generator.py
```
