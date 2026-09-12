# PitzDaily Backward-Facing Step — FoamPilot Tutorial

## Overview

This tutorial simulates **turbulent flow over a backward-facing step**
using **FoamPilot** and `simpleFoam` (k-omega SST). The case validates
the recirculation zone and reattachment length.

📁 **Location**: `foampilot/tutorials/03_pitzDaily_step/`

---

## 1. Prerequisites

- OpenFOAM installed
- FoamPilot installed

---

## 2. Case Physics

- **Geometry**: 2D channel with backward-facing step (step height H = 0.012 m)
- **Flow**: Incompressible, turbulent, steady
- **Inlet velocity**: 1 m/s
- **Turbulence model**: k-omega SST
- **Turbulence intensity**: 5%

### 2.1 Key Physics

The backward-facing step generates a **recirculation zone** downstream
of the step due to flow separation. A **reattachment point** forms where
the reversed flow re-attaches to the downstream wall.

### 2.2 Dimensionless Parameters

$$
Re_H = \frac{U H}{\nu} = \frac{1 \times 0.012}{1.5 \times 10^{-5}} \approx 800
$$

The recirculation bubble length for turbulent flow at this Re:

$$
L_r \approx 6.5 H \approx 0.078 \text{ m}
$$

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
    velocity=(ValueWithUnit(1, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
    turbulence_intensity=0.05,
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

### 4.1 Recirculation Zone Analysis

The recirculation zone is identified by negative axial velocity:

$$
u_x < 0 \quad \text{in the recirculation region}
$$

The reattachment length is found at the wall location where $u_x = 0$
downstream of the step.

### 4.2 Report Generation

```python
from foampilot.report.report_generator import CFDReportGenerator

report = CFDReportGenerator(
    case_path=case_path,
    title="Backward-Facing Step Report",
    author="FoamPilot",
)

report.add_statistic("Re_H", 800, "", "Hydraulic Reynolds number")
report.add_statistic("L_r_expected", 6.5, "H", "Expected reattachment length ratio")

report.save_html_report(filename="step_report.html")
```

### 4.3 LaTeX/Typst Reports

```python
from foampilot.report.latex_pdf import LatexDocument
from foampilot.report.typst_pdf import ScientificDocument, TypstRenderer

# LaTeX
doc = LatexDocument("Backward-Facing Step", "FoamPilot",
                    output_dir=case_path)
doc.add_title()
doc.add_section("Recirculation Zone", "Length and velocity analysis.")
doc.generate_document(output_format="tex")

# Typst
tdoc = ScientificDocument("BFS Analysis", "FoamPilot")
tdoc.add_equation(r"L_r = 6.5 H", caption="Reattachment length", label="eq:reattachment")
r = TypstRenderer()
r.render(tdoc)
```

---

## 5. Expected Results

| Quantity | Expected |
|----------|----------|
| Recirculation length (L_r/H) | 6.0–7.0 |
| Reattachment point x/H | 6.5 |
| Velocity recovery | By x/H ≈ 20 |

---

## 6. Execution

```bash
cd foampilot/tutorials/03_pitzDaily_step
python run.py
python report_generator.py
```
