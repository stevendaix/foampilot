# DamBreak VOF — FoamPilot Tutorial

## Overview

This tutorial simulates **sloshing of a water column** through a 2D
rectangular domain using the **VOF (Volume of Fluid)** model and
`interFoam` solver.

FoamPilot automates:

- VOF phase fraction (`alpha.water`) setup
- Two-phase material properties
- Gravity activation

📁 **Location**: `foampilot/tutorials/04_damBreak_multiphase/`

---

## 1. Prerequisites

- OpenFOAM installed
- FoamPilot installed

---

## 2. Case Physics

- **Domain**: 2D rectangular tank (5 m × 2 m × 0.1 m)
- **Phases**: Water (alpha = 1) and air (alpha = 0)
- **VOF model**: Volume of Fluid for interface tracking
- **Gravity**: Active (9.81 m/s², -Y direction)
- **Turbulence**: Laminar (low Re)

### 2.1 VOF Transport Equation

$$
\frac{\partial \alpha}{\partial t} + \nabla \cdot (\mathbf{u} \, \alpha) = 0
$$

Phase fraction `α` = 1 in water, 0 in air, 0–1 at the interface.

### 2.2 Momentum Equations

$$
\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \mathbf{u}) = -\nabla p + \mu \nabla^2 \mathbf{u} + \rho \mathbf{g} + \sigma \kappa \nabla \alpha
$$

Where:
- `σ` — surface tension coefficient
- `κ` — interface curvature
- `g` — gravity vector

### 2.3 Initial Conditions

- Water column: 2 m × 1 m at the left of the domain
- Rest of domain: filled with air
- Zero velocity everywhere initially

---

## 3. Workflow

### 3.1 Solver Initialization

```python
from foampilot.solver import Solver
from foampilot import ValueWithUnit

solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = False
solver.is_vof = True
solver.turbulence_model = "laminar"
```

Setting `solver.is_vof = True` automatically:

- Enables `interFoam` solver
- Sets up two-phase `transportProperties`
- Creates `alpha.water` field

### 3.2 Boundary Conditions

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

### 3.3 Execution

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. Post-processing

### 4.1 Interface Tracking

The VOF `alpha.water` field tracks the water-air interface:

```python
from foampilot import postprocess

foam_post = postprocess.FoamPostProcessing(case_path=case_path)
foam_post.foamToVTK()
```

### 4.2 Visualization

```python
import pyvista as pv

mesh = pv.read("VTK/0/cellular.vtk")
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh, scalars="alpha.water", cmap="Blues")
plotter.screenshot("dam_break_interface.png")
```

### 4.3 Report Generation

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

## 5. Expected Results

| Quantity | Expected |
|----------|----------|
| Water front velocity | ~4.4 m/s (√(2gh), h=1m) |
| Time to hit right wall | ~3 s |
| Wave reflection | Visible after impact |

---

## 6. Execution

```bash
cd foampilot/tutorials/04_damBreak_multiphase
python run.py
python report_generator.py
```
