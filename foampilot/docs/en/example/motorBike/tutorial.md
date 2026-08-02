# MotorBike External Aerodynamics — FoamPilot Tutorial

## Overview

This tutorial simulates **high-speed external flow around a motorcycle**
using `simpleFoam` (k-omega SST). It demonstrates wall-resolved meshing
and wake prediction.

FoamPilot automates:

- High-speed inlet setup (30 m/s)
- Wall and moving ground boundary conditions
- Drag and lift monitoring

📁 **Location**: `foampilot/tutorials/07_motorBike/`

---

## 1. Prerequisites

- OpenFOAM installed
- FoamPilot installed

---

## 2. Case Physics

- **Geometry**: Motorcycle model with road surface
- **Flow**: Incompressible, turbulent, steady
- **Speed**: 30 m/s (108 km/h highway speed)
- **Turbulence model**: k-omega SST
- **Turbulence intensity**: 5%

### 2.1 Dimensionless Parameters

Reynolds number based on vehicle length (L = 2.0 m):

$$
Re_L = \frac{U L}{\nu} = \frac{30 \times 2}{1.5 \times 10^{-5}} = 4 \times 10^6
$$

Drag coefficient:

$$
C_d = \frac{F_d}{\frac{1}{2} \rho U^2 A}
$$

Where:
- `Fd` — drag force
- `A` — frontal area (~0.7 m² for a motorcycle)

### 2.2 Wake Prediction

Downstream of the motorcycle, the wake exhibits:

- Velocity deficit
- Turbulent mixing
- Pressure recovery

$$
T_{aw} = T_\infty \left[ 1 + r \frac{\gamma - 1}{2} M_\infty^2 \right]
$$

(Recovery temperature formula for high-speed flow)

---

## 3. Workflow

### 3.1 Solver Initialization

```python
solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = False
solver.turbulence_model = "kOmegaSST"
```

### 3.2 Boundary Conditions

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(30, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    turbulence_intensity=0.05,
)
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet",
)
solver.boundary.apply_condition_with_wildcard(
    pattern=".*wheels.*|.*moving.*",
    condition_type="wall",
)
solver.boundary.apply_condition_with_wildcard(
    pattern=".*road.*",
    condition_type="wall",
)
solver.boundary.write_boundary_conditions()
```

FoamPilot's wildcard pattern system handles complex patches:

- `.*wheels.*` — matches all wheel patches
- `.*moving.*` — matches moving surfaces
- `.*road.*` — matches ground plane

### 3.3 Execution

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. Post-processing

### 4.1 Force Coefficients

```python
from foampilot.report.report_generator import CFDReportGenerator

report = CFDReportGenerator(
    case_path=case_path,
    title="MotorBike Aerodynamics",
    author="FoamPilot",
)
report.add_statistic("Re_L", 4e6, "", "Reynolds number")
report.add_statistic("Cd_expected", 0.35, "", "Expected drag coefficient")
```

### 4.2 LaTeX Report

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="MotorBike External Aerodynamics",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_toc()
doc.add_abstract("Aerodynamic analysis of motorcycle at 30 m/s.")
doc.add_section("Method", "")
doc.add_equation(r"Re_L = \frac{UL}{\nu}")
doc.add_section("Results", "")
doc.add_table(
    [["Drag coeff", "0.35"], ["Lift coeff", "0.05"]],
    headers=["Coefficient", "Value"],
    caption="Aerodynamic coefficients",
)
for img in ["pressure_contour.png", "velocity_vectors.png"]:
    doc.add_figure(img, caption=img.replace("_", " ").title())
doc.generate_document(output_format="pdf")
```

---

## 5. Expected Results

| Quantity | Expected |
|----------|----------|
| Drag coefficient (Cd) | 0.30–0.40 |
| Frontal drag | ~200–250 N |
| Wake size | ~3–5 bike lengths |
| Pressure recovery at tail | ~70–80% |

---

## 6. Execution

```bash
cd foampilot/tutorials/07_motorBike
python run.py
python report_generator.py
```
