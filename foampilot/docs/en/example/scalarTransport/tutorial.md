# Scalar Transport — FoamPilot Tutorial

## Overview

This tutorial simulates **passive scalar transport** (temperature field) in a
laminar channel flow using `buoyantSimpleFoam` with `energy_activated`.

FoamPilot automates:

- Energy equation activation
- Scalar boundary conditions
- `scalarTransportFoam` solver configuration

📁 **Location**: `foampilot/tutorials/05_scalarTransport/`

---

## 1. Prerequisites

- OpenFOAM installed
- FoamPilot installed

---

## 2. Case Physics

- **Domain**: 2D channel (1 m × 0.1 m)
- **Flow**: Laminar, incompressible
- **Scalar**: Temperature T (passive scalar)
- **Inlet temperature**: 300 K
- **Wall temperature**: 350 K (heated bottom wall)

### 2.1 Scalar Transport Equation

$$
\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \alpha \nabla^2 T + S_T
$$

Where:
- `α` — thermal diffusivity (α = ν/Pr)
- `S_T` — source term (optional)

### 2.2 Boundary Conditions

- **Inlet**: Fixed temperature `T = 300 K`
- **Outlet**: Zero gradient `∂T/∂n = 0`
- **Walls**: Fixed temperature `T = 350 K` (bottom), adiabatic (top)
- **Symmetry**: Zero gradient everywhere

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
solver.energy_activated = True
```

Setting `energy_activated = True` enables:

- Energy equation in `fvSchemes`
- Temperature field `T` initialisation
- Buoyancy coupling if `with_gravity = True`

### 3.2 Boundary Conditions

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

### 3.3 Execution

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. Post-processing

### 4.1 Temperature Statistics

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

### 4.2 LaTeX Report

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

### 4.3 Typst Report

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

## 5. Expected Results

| Quantity | Formula | Expected |
|----------|---------|----------|
| Bulk mean T | $T_{bulk} = \frac{1}{L} \int_0^L T dy$ | ~325 K |
| Wall heat flux | $q'' = -k \frac{dT}{dy}\big|_{wall}$ | ~500 W/m² |
| Outlet T distribution | — | Parabolic profile |

---

## 6. Execution

```bash
cd foampilot/tutorials/05_scalarTransport
python run.py
python report_generator.py
```
