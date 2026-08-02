# SimpleCar Turbulent Flow — FoamPilot Tutorial

## Overview

This tutorial demonstrates a **steady RANS turbulent flow** simulation around a
simplified car geometry using **FoamPilot** and the `simpleFoam` solver with the
**k-omega SST** turbulence model.

FoamPilot automates:

- Turbulent boundary condition setup with turbulence intensity
- Function objects (field average, run-time controls)
- Force and pressure coefficient monitoring

📁 **Location**: `foampilot/tutorials/02_simpleCar_turbulent/`

---

## 1. Prerequisites

- OpenFOAM installed
- FoamPilot installed
- `classy_blocks` (optional, for geometry)

---

## 2. Case Physics

- **Geometry**: Simplified car external aerodynamics
- **Flow**: Incompressible, turbulent, steady-state
- **Inlet velocity**: 30 m/s (108 km/h headwind)
- **Turbulence model**: k-omega SST
- **Turbulence intensity**: 5%

### 2.1 Governing Equations

RANS with Boussinesq approximation:

$$
\nabla \cdot \mathbf{u} = 0
$$

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nabla \cdot \left[ \nu_{eff} \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right) \right]
$$

### 2.2 k-omega SST Model

Turbulent kinetic energy:

$$
\frac{\partial (\rho k)}{\partial t} + \frac{\partial (\rho u_j k)}{\partial x_j} = P_k - \beta^* \rho k \omega
$$

Specific dissipation rate:

$$
\frac{\partial (\rho \omega)}{\partial t} + \frac{\partial (\rho u_j \omega)}{\partial x_j} = \alpha S_\omega
$$

### 2.3 Dimensionless Parameters

Wind tunnel Reynolds number (based on car length L = 4.5 m):

$$
Re_L = \frac{U L}{\nu} = \frac{30 \times 4.5}{1.5 \times 10^{-5}} \approx 9 \times 10^6
$$

---

## 3. Workflow

### 3.1 Solver Setup

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
```

FoamPilot's `velocityInlet` with `turbulence_intensity` automatically computes
`k` and `omega` inlet values:

$$
k = \frac{3}{2} (I \cdot U)^2, \quad \omega = \frac{\sqrt{k}}{L_{ref} \cdot 0.016}
$$

### 3.3 Function Objects

FoamPilot supports adding function objects for monitoring:

```python
solver.system.functions.velocity_field_average = {
    "type": "fieldAverage",
    "enabled": True,
    "fields": [("U", "U_mean", "U_rms")],
}
```

### 3.4 Execution

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. Post-processing

### 4.1 Force Coefficients

FoamPilot monitors drag and lift coefficients via function objects:

```
forces {
    type            forces;
    functionObjectLibs ("libforces.so");
    patches          (car body walls);
    rho            rhoInf;  // incompressible
    liftDir        (0 0 1);
    dragDir        (1 0 0);
    CofR           (0 0 0);
}
```

### 4.2 Pressure Coefficient

$$
C_p = \frac{p - p_\infty}{\frac{1}{2} \rho U_\infty^2}
$$

### 4.3 Report Generation

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="SimpleCar Aerodynamics Report",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_abstract("External aerodynamics simulation of a simplified car.")
doc.add_section("Drag Coefficient", f"Cd = {cd_value:.4f}")
doc.add_section("Pressure Distribution", "")
for img in ["pressure_contour.png", "velocity_contour.png"]:
    doc.add_figure(img, caption=img.replace("_", " ").title())
doc.generate_document(output_format="pdf")
```

---

## 5. Expected Results

| Quantity | Expected Value |
|----------|----------------|
| Drag coefficient (Cd) | 0.25–0.35 |
| Lift coefficient (Cl) | 0.1–0.2 |
| Max. Cp | ~1.2 |
| Reattachment length behind car | ~2–3 car lengths |

---

## 6. Execution

```bash
cd foampilot/tutorials/02_simpleCar_turbulent
python run.py
python report_generator.py
```
