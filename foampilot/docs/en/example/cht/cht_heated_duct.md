# CHT Heat Exchanger – FoamPilot Tutorial

## Overview

This tutorial demonstrates a **complete conjugate heat transfer (CHT) workflow** using
**FoamPilot** and **OpenFOAM 13** (`chtMultiRegionFoam`). It models a steady-state,
laminar, compressible flow of air in a heated duct coupled to a solid copper wall.

FoamPilot's **CHT module** (`foampilot.cht`) provides dedicated classes:

- `ChtSolver` — multi-region conjugate heat transfer solver
- `FluidRegion` — fluid domain (heRhoThermo, compressible)
- `SolidRegion` — solid domain (heSolidThermo)
- `CoupledInterface` — fluid-solid thermal coupling interface
- `FixedTemperatureBC`, `CoupledTemperatureBC`, etc. — boundary condition helpers
- `calc_nusselt_number`, `calc_heat_transfer_coefficient`, `calc_thermal_resistance` — post-processing

FoamPilot report engine

The `CFDReportGenerator` integrates with both:

- `LatexDocument` — LaTeX/PDF reports via PyLaTeX
- `ScientificDocument` / `TypstRenderer` — Typst-based scientific documents

The `CFDReportGenerator` provides:

- `add_statistic()` — register scalar statistics (Re, Nu, h, etc.)
- `add_figure()` — register images
- `add_table()` — register data tables
- `collect_time_series()` — collect field statistics across time steps
- `collect_region_statistics()` — per-region field statistics
- `save_html_report()` — interactive HTML report with Plotly
- `save_latex_report()` — LaTeX report with tables and figures
- `save_typst_report()` — Typst scientific document

📁 **Location**: `foampilot/tutorials/09_CHT_heatedDuct/`

---

## 1. Prerequisites

- OpenFOAM 13 installed and accessible
- FoamPilot installed (`pip install -e .`)
- Python dependencies: `pyvista`, `numpy`, `pandas`, `OpenFOAM` runtime

---

## 2. Case Physics

- **Geometry**: 2D duct (0.1 m × 0.02 m × 0.0005 m)
- **Fluid**: Air (compressible, `heRhoThermo`, perfect gas, hConst)
- **Solid**: Copper wall (380 W/m·K, 8960 kg/m³, 385 J/kg·K)
- **Flow**: Laminar, steady-state (transient via `chtMultiRegionFoam`)
- **Boundary conditions**:
  - Inlet: 300 K, 1.0 m/s
  - Outlet: 350 K, 1e5 Pa
  - Bottom wall (solid): 350 K
  - Top wall: 350 K

---

## 3. Workflow

### 3.1 Case Setup with CHT Solver

```python
from foampilot.cht import ChtSolver, FluidRegion, SolidRegion, CoupledInterface

fluid_region = FluidRegion(
    name="fluid",
    temperature=300.0,
    velocity=(1.0, 0.0, 0.0),
    turbulence_model="kOmegaSST",
    thermophysical_model="heRhoThermo",
    equation_of_state="perfectGas",
)

solid_region = SolidRegion(
    name="solid",
    temperature=350.0,
    thermal_conductivity=380.0,   # W/(m·K) — copper
    density=8960.0,               # kg/m³
    specific_heat=385.0,          # J/(kg·K)
)

solver = ChtSolver(
    case_path=case_path,
    solver_name="chtMultiRegionFoam",
    regions=[fluid_region, solid_region],
    interfaces=[CoupledInterface(...)],
)

solver.system.controlDict.start_time = 0
solver.system.controlDict.end_time = 1.0
solver.system.controlDict.delta_t = 5e-4
solver.system.controlDict.application = "chtMultiRegionFoam"

solver.setup_case()
solver.write_case()
```

### 3.2 Mesh Generation

The mesh is generated via a JSON configuration:

```python
from foampilot import Meshing

mesh = Meshing(case_path, mesher="blockMesh")
mesh.mesher.load_from_json(case_path / "block_mesh.json")
mesh.mesher.write(file_path=case_path / "system" / "blockMeshDict")
solver.run_command(["blockMesh"], log_filename="log.blockMesh")
```

### 3.3 Multi-Region Setup

```python
solver.run_command(["createZones"], log_filename="log.createZones")
solver.run_command(["splitMeshRegions", "-cellZones", "-defaultRegionName", "fluid"],
                   log_filename="log.splitMeshRegions")
solver.run_command(["foamSetupCHT"], log_filename="log.foamSetupCHT")
```

### 3.4 Simulation Execution

```python
solver.run_simulation(nb_proc=1)
```

### 3.5 VTK Conversion

```python
solver.run_command(["foamToVTK", "-region", "fluid", "-latestTime",
                    "-fields", "(T U p k omega)"],
                   log_filename="log.foamToVTK_fluid")
solver.run_command(["foamToVTK", "-region", "solid", "-latestTime",
                    "-fields", "(T)"],
                   log_filename="log.foamToVTK_solid")
```

---

## 4. Post-Processing

The post-processing script (`run_post.py`) uses foampilot CHT analysis functions:

```python
from foampilot.cht import (
    calc_nusselt_number,
    calc_heat_transfer_coefficient,
    calc_thermal_resistance,
    calc_total_heat_balance,
    calc_temperature_contour,
)
```

### 4.1 Key Results

| Metric | Value | Reference |
|--------|-------|-----------|
| Interface T (fluid side) | 350.00 K | 350.00 K |
| Interface T (solid side) | 350.00 K | 350.00 K |
| Heat transfer coefficient h | 3.38 W/(m²·K) | 3.38 |
| Nusselt number Nu | 0.2597 | 0.2597 |
| Thermal resistance R_th | 0.2963 K/W | 0.2963 |

### 4.2 Temperature Statistics

| Region | T_min (K) | T_max (K) | T_mean (K) |
|--------|-----------|-----------|------------|
| Fluid | 300.00 | 350.00 | 304.48 |
| Solid | 349.98 | 350.00 | 350.00 |

---

## 5. Report Generation

FoamPilot's `CFDReportGenerator` automates report creation. The CHT tutorial
generates a comprehensive report with:

```python
from foampilot.report.report_generator import CFDReportGenerator

report = CFDReportGenerator(
    case_path=case_path,
    title="CHT Heated Duct Report",
    author="FoamPilot",
)

# Add key statistics
report.add_statistic("Nu", 0.2597, "", "Nusselt number")
report.add_statistic("h", 3.38, "W/(m²·K)", "Heat transfer coefficient")
report.add_statistic("R_th", 0.2963, "K/W", "Thermal resistance")
report.add_statistic("T_interface", 350.0, "K", "Interface temperature")

# Add figures
report.add_figure("postProcessing/fluid_temperature_contour.png",
                  "Temperature contour (fluid)")
report.add_figure("postProcessing/solid_temperature_contour.png",
                  "Temperature contour (solid)")
report.add_figure("postProcessing/cht_temperature_contour.png",
                  "CHT temperature overlay")

# Generate LaTeX report
report.save_latex_report(compile_pdf=True)

# Generate interactive HTML report
report.save_html_report()
```

### 5.1 Report Types

| Method | Output | Features |
|--------|--------|----------|
| `save_latex_report()` | `.tex` / `.pdf` | LaTeX via PyLaTeX, tables, figures |
| `save_typst_report()` | `.typ` | Typst scientific document |
| `save_html_report()` | `.html` | Interactive Plotly figures, embedded tables |

---

## 6. Files

| File | Description |
|------|-------------|
| `run.py` | Main tutorial script (CHT setup, mesh, simulation) |
| `run_post.py` | Post-processing and analysis |
| `block_mesh.json` | Geometry configuration for `BlockMesher` |
| `README.md` | This documentation |

---

## 7. Execution

```bash
cd foampilot/tutorials/09_CHT_heatedDuct
python run.py
python run_post.py
```

---

## 8. Expected Outputs

```
postProcessing/
├── temperature_statistics.csv
├── temperature_profile.csv
├── temperature_profile_combined.csv
├── fluid_temperature_contour.png
├── solid_temperature_contour.png
├── cht_temperature_contour.png
└── CHT_Report.md
```
