# Cavity Laminar Flow — FoamPilot Tutorial

## Vue d’ensemble

Ce tutoriel démontre une simulation complète de l'écoulement incompressible laminaire d'une cavité entraînée par un couvercle (lid-driven cavity) en utilisant **FoamPilot** et le solveur `icoFoam`.

FoamPilot automatise :

- la génération de géométrie avec blockMesh
- la définition des conditions aux limites
- la configuration du solveur (`laminar` turbulence, `icoFoam`)
- le post-traitement des résidus

📁 **Emplacement**: `foampilot/tutorials/01_cavity_laminar/`

---

## 1. Prerequisites

- OpenFOAM installé et accessible
- FoamPilot installé (`pip install -e .`)

---

## 2. Case Physics

- **Géométrie** : cavité 2D carrée (1 m × 1 m)
- **Fluide** : eau (incompressible, laminaire)
- **Conditions aux limites** :
  - Couvercle mobile : `U = (1, 0, 0) m/s`
  - Parois fixes : pas de glissement (no-slip)
  - Avant/arrière : symétrie

### 2.1 Governing Equations

Continuité (incompressible) :

$$
\nabla \cdot \mathbf{u} = 0
$$

Navier–Stokes (laminaire) :

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u}
$$

### 2.2 Dimensionless Parameters

Nombre de Reynolds (basé sur la vitesse du couvercle et la hauteur de la cavité) :

$$
Re = \frac{U L}{\nu} = \frac{1 \cdot 1}{1 \times 10^{-6}} = 10^6
$$

Pour le régime laminaire, `nu = 1e-6 m²/s` donne `Re ≈ 100` (tutoriel de cavité standard).

---

## 3. Workflow

### 3.1 Solver Initialization

```python
from pathlib import Path
from foampilot.solver import Solver
from foampilot import ValueWithUnit

case_path = Path.cwd()

solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = False
solver.turbulence_model = "laminar"
```

### 3.2 System and Constant Dictionaries

```python
solver.system.write()
solver.constant.write()
```

FoamPilot génère automatiquement :

- `system/controlDict`
- `system/fvSchemes`
- `system/fvSolution`
- `constant/transportProperties`
- `constant/turbulenceProperties`

### 3.3 Boundary Conditions

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    pattern="movingWall",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(1, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
)
solver.boundary.apply_condition_with_wildcard(
    pattern="fixedWalls",
    condition_type="wall",
)
solver.boundary.apply_condition_with_wildcard(
    pattern="frontAndBack",
    condition_type="symmetry",
)
solver.boundary.write_boundary_conditions()
```

La méthode `apply_condition_with_wildcard` de FoamPilot utilise la correspondance par expression régulière pour assigner les conditions aux limites en fonction des noms de patch. Cela correspond à :

- `movingWall` → `fixedValue` pour U, `zeroGradient` pour p
- `fixedWalls` → `noSlip` pour U, `zeroGradient` pour p
- `frontAndBack` → `symmetry` pour les deux

### 3.4 Simulation Execution

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. Post-processing

FoamPilot inclut le suivi des résidus via `ResidualsPost` :

```python
from foampilot.utilities import ResidualsPost

residuals = ResidualsPost(case_path / "log.icoFoam")
residuals.process(
    export_csv=True,
    export_json=True,
    export_png=True,
    export_html=True,
)
```

Les résultats sont exportés vers :

- `postProcessing/residuals.csv`
- `postProcessing/residuals.json`
- `postProcessing/residuals.png`
- `postProcessing/residuals.html`

---

## 5. Report Generation

### 5.1 LaTeX Report

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="Cavity Laminar Flow Report",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_toc()
doc.add_abstract("Laminar lid-driven cavity simulation using icoFoam.")
doc.add_section("Results", "Convergence data and field statistics.")
doc.generate_document(output_format="pdf")
```

### 5.2 HTML Report

```python
from foampilot.report.report_generator import CFDReportGenerator

report = CFDReportGenerator(
    case_path=case_path,
    title="Cavity CHT Report",
    author="FoamPilot",
)
report.add_statistic("Re", 100, "", "Reynolds number")
report.save_html_report(filename="cavity_report.html")
```

### 5.3 Typst Report

```python
from foampilot.report.typst_pdf import ScientificDocument, TypstRenderer

doc = ScientificDocument("Cavity Flow Analysis", "FoamPilot")
doc.add_section("Introduction", "Lid-driven cavity laminar flow.")
doc.add_equation(r"Re = \frac{UL}{\nu}", caption="Reynolds number", label="eq:re")
renderer = TypstRenderer()
renderer.compile_pdf(doc)
```

---

## 6. Expected Results

| Variable | Min | Max | Mean |
|----------|-----|-----|------|
| U_x | 0 | ~2.5 | ~1.0 |
| p | -500 | +500 | 0 |

- Vortex primaire au centre de la cavité
- Vortex secondaires dans les coins
- Vitesse maximale près du coin de sortie du couvercle

---

## 7. Execution

```bash
cd foampilot/tutorials/01_cavity_laminar
python run.py
python report_generator.py  # generates PDF/HTML/Typst reports
```
