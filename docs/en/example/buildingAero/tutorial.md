# Building Aerodynamics — FoamPilot Tutorial

## Overview

This tutorial simulates **turbulent flow around buildings** in an urban
environment using `simpleFoam` (k-omega SST). It demonstrates advanced
mesh manipulation using `topoSet` and `createPatch`.

FoamPilot automates:

- `topoSet` and `createPatch` execution
- Urban boundary conditions (freestream, buildings)
- Wind loading analysis

📁 **Location**: `foampilot/tutorials/06_buildingAero/`

---

## 1. Prerequisites

- OpenFOAM installed
- FoamPilot installed

---

## 2. Case Physics

- **Domain**: Urban canyon with multiple buildings (10 m × 10 m × 3 m)
- **Flow**: Incompressible, turbulent, steady
- **Inlet velocity**: 10 m/s (50% urban turbulence intensity)
- **Turbulence model**: k-omega SST
- **Gravity**: Off (pressure-driven, steady RANS)

### 2.1 Urban Boundary Layer Profile

Inlet velocity (logarithmic profile):

$$
u(y) = u_* \frac{\ln(y / y_0)}{\kappa}
$$

Where:
- `u*` — friction velocity
- `κ` — von Kármán constant (0.41)
- `y0` — roughness height

FoamPilot simplifies this with `velocityInlet` and `turbulence_intensity`:

```python
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(10, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    turbulence_intensity=0.15,
)
```

### 2.2 Canyon Wind Effect

Buildings create **street canyon vortices** downstream. The canyon aspect
ratio (building height / street width) determines flow regime:

$$
AR = \frac{H_{building}}{W_{street}}
$$

For AR ≈ 1 (this tutorial), the flow is in the "critical" regime with
strong recirculation inside the canyon.

---

## 3. Workflow

### 3.1 Solver Initialization

```python
solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = False
solver.turbulence_model = "kOmegaSST"
```

### 3.2 Mesh Manipulation with topoSet + createPatch

FoamPilot wraps OpenFOAM topology tools:

```python
# topoSet for defining building cell zones
solver.system.run_topoSet()

# createPatch for renaming boundary patches
solver.system.run_createPatch()
```

This executes:

- `system/topoSetDict` — defines cell/zone sets for buildings
- `system/createPatchDict` — renames faces to named patches

### 3.3 Boundary Conditions

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

### 3.4 Execution

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. Post-processing

### 4.1 Canyon Flow Visualization

PyVista visualization of velocity field:

```python
import pyvista as pv
from pathlib import Path

mesh = pv.read(str(Path("VTK/latest/cellular.vtk")))
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh.slice("z"), scalars="U", cmap="viridis")
plotter.screenshot("canyon_velocity.png")
```

### 4.2 Report Generation

```python
from foampilot.report.latex_pdf import LatexDocument
from foampilot.report.report_generator import CFDReportGenerator

# HTML report
report = CFDReportGenerator(
    case_path=case_path,
    title="Building Aero Report",
    author="FoamPilot",
)
report.add_statistic("U_inlet", 10.0, "m/s", "Inlet velocity")
report.add_statistic("I_inlet", 0.15, "", "Turbulence intensity")
report.save_html_report(filename="building_report.html")

# LaTeX report
doc = LatexDocument(
    title="Urban Building Aerodynamics",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_abstract("Wind flow simulation around urban buildings.")
doc.add_section("Canyon Flow", "")
doc.add_figure("canyon_velocity.png", "Velocity field in urban canyon")
doc.generate_document(output_format="pdf")
```

---

## 5. Expected Results

| Quantity | Expected |
|----------|----------|
| Wind speed-up at roof level | 1.2–1.5× U_inlet |
| Canyon recirculation zone | Visible behind each building |
| Pressure coefficient Cp | -0.5 to +1.0 |
| Pedestrian-level velocity | < 0.2 U_inlet |

---

## 6. Execution

```bash
cd foampilot/tutorials/06_buildingAero
python run.py
python report_generator.py
```
