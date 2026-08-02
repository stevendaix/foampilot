# Cavity Laminar Flow — FoamPilot Tutorial

## Overview

This tutorial demonstrates a **complete laminar incompressible flow** simulation of a
lid-driven cavity using **FoamPilot** and the `icoFoam` solver.

FoamPilot automates:

- blockMesh geometry generation
- Boundary condition definition
- Solver configuration (`laminar` turbulence, `icoFoam`)
- Residual post-processing

📁 **Location**: `foampilot/tutorials/01_cavity_laminar/`

---

## 1. Prerequisites

- OpenFOAM installed and accessible
- FoamPilot installed (`pip install -e .`)

---

## 2. Case Physics

- **Geometry**: 2D square cavity (1 m × 1 m)
- **Fluid**: Water (incompressible, laminar)
- **Boundary conditions**:
  - Moving lid: `U = (1, 0, 0) m/s`
  - Fixed walls: no-slip
  - Front/back: symmetry

### 2.1 Governing Equations

Continuity (incompressible):

$$
\nabla \cdot \mathbf{u} = 0
$$

Navier–Stokes (laminar):

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u}
$$

### 2.2 Dimensionless Parameters

Reynolds number (based on lid velocity and cavity height):

$$
Re = \frac{U L}{\nu} = \frac{1 \cdot 1}{1 \times 10^{-6}} = 10^6
$$

For laminar regime, `nu = 1e-6 m²/s` gives `Re ≈ 100` (standard cavity tutorial).

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

FoamPilot auto-generates:

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

FoamPilot's `apply_condition_with_wildcard` uses regex pattern matching to assign
boundary conditions based on patch names. This maps to:

- `movingWall` → `fixedValue` for U, `zeroGradient` for p
- `fixedWalls` → `noSlip` for U, `zeroGradient` for p
- `frontAndBack` → `symmetry` for both

### 3.4 Simulation Execution

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. Post-processing

FoamPilot includes residual tracking via `ResidualsPost`:

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

Results are exported to:

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

- Primary vortex in cavity center
- Secondary vortices in corners
- Maximum velocity near lid outlet corner

---

## 7. Execution

```bash
cd foampilot/tutorials/01_cavity_laminar
python run.py
python report_generator.py  # generates PDF/HTML/Typst reports
```
