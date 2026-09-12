# Transport scalaire — Tutoriel FoamPilot

## Aperçu

Ce tutoriel simule le **transport scalaire passif** (champ de température) dans un écoulement de canal laminaire en utilisant `buoyantSimpleFoam` avec `energy_activated`.

FoamPilot automatise :

- Activation de l'équation d'énergie
- Conditions aux limites pour le scalaire
- Configuration du solveur `scalarTransportFoam`

📁 **Emplacement**: `foampilot/tutorials/05_scalarTransport/`

---

## 1. Prérequis

- OpenFOAM installé
- FoamPilot installé

---

## 2. Physique du cas

- **Domaine** : canal 2D (1 m × 0.1 m)
- **Écoulement** : laminaire, incompressible
- **Scalaire** : Température T (scalaire passif)
- **Température à l'entrée** : 300 K
- **Température au mur** : 350 K (paroi inférieure chauffée)

### 2.1 Équation du transport scalaire

$$
\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \alpha \nabla^2 T + S_T
$$

Où :
- `α` — diffusivité thermique (α = ν/Pr)
- `S_T` — terme source (optionnel)

### 2.2 Conditions aux limites

- **Entrée** : Température fixée `T = 300 K`
- **Sortie** : Gradient nul `∂T/∂n = 0`
- **Parois** : Température fixée `T = 350 K` (fond), adiabatique (haut)
- **Symétrie** : Gradient nul partout

---

## 3. Flux de travail

### 3.1 Initialisation du solveur

```python
from pathlib import Path
from foampilot.solver import Solver
from foampilot import ValueWithUnit

case_path = Path.cwd()

solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = False
solver.turbulence_model = "laminar"
solver.energy_activated = True
```

Le fait de définir `energy_activated = True` active :

- Équation d'énergie dans `fvSchemes`
- Initialisation du champ de température `T`
- Couplage de la flottabilité si `with_gravity = True`

### 3.2 Conditions aux limites

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(1, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
)
solver.boundary.apply_condition_with_wildcard(
    pattern="walls",
    condition_type="wall",
)
solver.boundary.write_boundary_conditions()
```

### 3.3 Exécution

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. Post-traitement

### 4.1 Statistiques de température

```python
from foampilot.report.report_generator import CFDReportGenerator

report = CFDReportGenerator(
    case_path=case_path,
    title="Scalar Transport Report",
    author="FoamPilot",
)

report.add_statistic("T_inlet", 300.0, "K", "Inlet temperature")
report.add_statistic("T_wall", 350.0, "K", "Wall temperature")
report.add_statistic("Pr", 0.71, "", "Prandtl number (air)")
```

### 4.2 Rapport LaTeX

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="Scalar Transport Analysis",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_abstract("Passive scalar transport in a laminar channel flow.")
doc.add_equation(
    r"\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \alpha \nabla^2 T",
    caption="Scalar transport equation",
)
doc.add_section("Boundary Conditions", "")
doc.add_table(
    [["Inlet", "300", "K"], ["Wall", "350", "K"], ["Outlet", "zeroGradient", ""]],
    headers=["Patch", "Condition", "Value"],
    caption="Temperature boundary conditions",
)
doc.generate_document(output_format="pdf")
```

### 4.3 Rapport Typst

```python
from foampilot.report.typst_pdf import ScientificDocument, TypstRenderer

doc = ScientificDocument("Scalar Transport", "FoamPilot")
doc.add_section("Introduction", "Passive scalar transport analysis.")
doc.add_equation(r"Pe = UL/\alpha", caption="Peclet number", label="eq:pe")
doc.add_table(
    [["Parameter", "Value"], ["Re", "100"], ["Pe", "71"]],
    caption="Flow parameters",
)
renderer = TypstRenderer()
renderer.compile_pdf(doc)
```

---

## 5. Résultats attendus

| Quantité | Formule | Valeur attendue |
|----------|---------|-----------------|
| T moyenne d'écoulement | $T_{bulk} = \frac{1}{L} \int_0^L T dy$ | ~325 K |
| Flux de chaleur au mur | $q'' = -k \frac{dT}{dy}\big|_{wall}$ | ~500 W/m² |
| Distribution de T à la sortie | — | Profil parabolique |

---

## 6. Exécution

```bash
cd foampilot/tutorials/05_scalarTransport
python run.py
python report_generator.py
```
