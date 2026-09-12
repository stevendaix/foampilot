# Thermal Buoyancy (buoyantSimpleFoam) — FoamPilot Tutorial

## Overview

This tutorial simulates **natural convection** in a heated room using
`buoyantSimpleFoam` with the **Boussinesq approximation**. It demonstrates
the coupling of fluid flow and heat transfer under gravity.

FoamPilot automates:

- Gravity activation and Boussinesq buoyancy
- Isothermal wall temperature patches
- Energy equation configuration

📁 **Location**: `foampilot/tutorials/08_thermalBuoyancy/`

---

## 1. Prerequisites

- OpenFOAM installed
- FoamPilot installed

---

## 2. Case Physics

- **Domain**: Room (4 m × 4 m × 3 m)
- **Fluid**: Air (incompressible Boussinesq)
- **Hot wall**: 350 K (left wall)
- **Cold wall**: 300 K (right wall)
- **Other walls**: Adiabatic (zero gradient)
- **Gravity**: 9.81 m/s² (-Z direction)

### 2.1 Boussinesq Approximation

The density variation is modeled as:

$$
\rho = \rho_0 [1 - \beta (T - T_0)]
$$

Where:
- `ρ₀` — reference density
- `β` — thermal expansion coefficient
- `T₀` — reference temperature

The buoyancy term in the momentum equation:

$$
\frac{\partial (rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \mathbf{u}) = -\nabla p_{rgh} + \nabla \cdot (\mu_{eff} \nabla \mathbf{u}) + \rho \mathbf{g}
$$

### 2.2 Rayleigh Number

$$
Ra = \frac{g \beta \Delta T L^3}{\nu \alpha}
$$

For this case (ΔT = 50 K, L = 4 m):

$$
Ra = \frac{9.81 \times 3.2 \times 10^{-3} \times 50 \times 4^3}{1.5 \times 10^{-5} \times 2.2 \times 10^{-5}} \approx 9.7 \times 10^9
$$

This is in the **turbulent natural convection** regime (Ra > 1e9), confirming
the need for a turbulence model (k-epsilon).

### 2.3 Governing Equations

Energy:

$$
\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \alpha \nabla^2 T
$$

Pressure (hydrostatic-modified):

$$
p_{rgh} = p - \rho \mathbf{g} \cdot \mathbf{h}
$$

---

## 3. Workflow

### 3.1 Solver Initialization

```python
solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = True
solver.turbulence_model = "kEpsilon"
```

Setting `solver.with_gravity = True` enables:

- `buoyantSimpleFoam` solver
- Boussinesq density in momentum equation
- `p_rgh` pressure variable

### 3.2 Boundary Conditions

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(ValueWithUnit(0.1, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
)
solver.boundary.apply_condition_with_wildcard(
    pattern="walls",
    condition_type="wall",
)

# Hot wall at 350 K
solver.boundary.set_raw_condition("hotWall", "T", {"type": "fixedValue", "value": "350"})
# Cold wall at 300 K
solver.boundary.set_raw_condition("coldWall", "T", {"type": "fixedValue", "value": "300"})
```

FoamPilot's `set_raw_condition` allows direct OpenFOAM dictionary specification
for complex cases.

### 3.3 Execution

```python
solver.run_simulation(nb_proc=1)
```

---

## 4. Post-processing

### 4.1 Natural Convection Cells

```python
from foampilot.report.report_generator import CFDReportGenerator

report = CFDReportGenerator(
    case_path=case_path,
    title="Thermal Buoyancy Report",
    author="FoamPilot",
)
report.add_statistic("Ra", 9.7e9, "", "Rayleigh number")
report.add_statistic("T_hot", 350.0, "K", "Hot wall temperature")
report.add_statistic("T_cold", 300.0, "K", "Cold wall temperature")
report.save_html_report(filename="buoyancy_report.html")
```

### 4.2 LaTeX Report

```python
from foampilot.report.latex_pdf import LatexDocument

doc = LatexDocument(
    title="Natural Convection in a Heated Room",
    author="FoamPilot",
    output_dir=case_path,
)
doc.add_title()
doc.add_toc()
doc.add_abstract("Boussinesq buoyancy simulation with buoyantSimpleFoam.")
doc.add_section("Governing Equations", "")
doc.add_equation(
    r"Ra = \frac{g \beta \Delta T L^3}{\nu \alpha}",
    caption="Rayleigh number",
)
doc.add_equation(
    r"p_{rgh} = p - \rho \mathbf{g} \cdot \mathbf{h}",
    caption="Modified pressure",
)
doc.add_section("Boundary Conditions", "")
doc.add_table(
    [["hotWall", "350", "K"], ["coldWall", "300", "K"], ["Other walls", "adiabatic", ""]],
    headers=["Patch", "Temperature", "Condition"],
    caption="Wall boundary conditions",
)
doc.generate_document(output_format="pdf")
```

### 4.3 Typst Scientific Document

```python
from foampilot.report.typst_pdf import ScientificDocument, TypstRenderer

doc = ScientificDocument("Natural Convection", "FoamPilot")
doc.add_section("Introduction", "Buoyancy-driven flow analysis.")
doc.add_equation(
    r"Ra = g \beta \Delta T L^3 / (\nu \alpha)",
    caption="Rayleigh number",
    label="eq:rayleigh",
)
doc.add_table(
    [["T_hot", "350 K"], ["T_cold", "300 K"], ["g", "9.81 m/s²"]],
    headers=["Parameter", "Value"],
    caption="Simulation parameters",
)
renderer = TypstRenderer()
renderer.compile_pdf(doc)
```

---

## 5. Expected Results

| Quantity | Expected |
|----------|----------|
| Natural convection cells | 2–4 circulating cells |
| Hot air rise velocity | ~0.1–0.3 m/s |
| Temperature profile at mid-plane | Linear from 350 K to 300 K |
| Velocity near hot wall | Upward (0.05–0.15 m/s) |

---

## 6. Execution

```bash
cd foampilot/tutorials/08_thermalBuoyancy
python run.py
python report_generator.py
```
