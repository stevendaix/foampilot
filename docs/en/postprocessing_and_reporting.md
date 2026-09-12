# Post-processing and reporting

FoamPilot supports two complementary post-processing paths. The traditional path converts a case with `foamToVTK` and then loads the VTK output with PyVista. The direct path reads native OpenFOAM meshes and fields into PyVista without creating an intermediate VTK tree.

## Native OpenFOAM readers

Use the direct reader when the case already contains a valid `constant/polyMesh` and you want to avoid an external conversion step.

```python
from foampilot.postprocess import OpenFOAMDirectReader

reader = OpenFOAMDirectReader("/path/to/case")
mesh = reader.to_pyvista(fields=["U", "p"], time_step="latest")
print(mesh.n_points, mesh.n_cells)
```

The reader detects point and cell fields from OpenFOAM field headers, supports lazy loading, caches fields, and can read compressed field files. The convenience function is useful for small scripts:

```python
from foampilot.postprocess import read_openfoam

mesh = read_openfoam(
    "/path/to/case",
    fields=["U", "p"],
    time_step="latest",
)
```

For conjugate heat-transfer cases, use `CHTDirectReader`. It discovers the fluid and solid regions and returns a PyVista `MultiBlock` structure.

```python
import pyvista as pv
from foampilot.postprocess import CHTDirectReader

reader = CHTDirectReader("/path/to/cht-case")
print(reader.region_names)
blocks = reader.get_all_meshes(fields=["T"], time_step="latest")

plotter = pv.Plotter(off_screen=True)
for region_name, region_mesh in blocks.items():
    plotter.add_mesh(region_mesh, scalars="T", name=region_name)
plotter.screenshot("temperature.png")
plotter.close()
```

Interface temperatures can be inspected directly when a named region interface is available:

```python
interface = reader.get_interface_temperatures(
    "fluid_to_solid", time_step="latest"
)
print(interface["fluid_T"])
print(interface["solid_T"])
print(interface["T_interface"])
```

## PyVista post-processing

`FoamPostProcessing` remains useful when the existing workflow relies on `foamToVTK`, time-step discovery, or higher-level plotting helpers.

```python
from foampilot.postprocess import FoamPostProcessing

post = FoamPostProcessing(case_path="/path/to/case")
post.foamToVTK()
time = post.get_all_time_steps()[-1]
mesh = post.load_time_step(time)["cell"]
```

Typical operations include slices, contours, vector plots, vortex analysis, mesh statistics, image export, and animation export. In headless environments, create plots with `off_screen=True` or use FoamPilot’s rendering helpers to detect a usable off-screen backend.

## Interactive web presentations

The `foampilot.postprocess.web_presentation` module provides Plotly builders for velocity, pressure, and temperature fields and a `CFDDashboard` for interactive exploration. A minimal pattern is:

```python
from foampilot.postprocess.web_presentation import (
    plotly_velocity_magnitude,
    plotly_pressure_contour,
    CFDDashboard,
)

velocity_figure = plotly_velocity_magnitude(mesh)
pressure_figure = plotly_pressure_contour(mesh)
# Pass the figures to the dashboard or to a Plotly/Streamlit application.
```

The dashboard is intended for exploration and communication. For a reproducible engineering record, save the input script, generated dictionaries, solver log, figures, and report together.

## Simulation and mesh reports

The `foampilot.report` package includes structured reports for mesh quality, convergence, and solver studies. The reporting API is designed to run after the simulation so that failed or incomplete runs can be recorded rather than silently ignored.

The LaTeX API is suitable when a PDF calculation note is required:

```python
from foampilot.report import latex_pdf

document = latex_pdf.LatexDocument(
    title="OpenFOAM simulation report",
    author="FoamPilot",
    filename="simulation_report",
    output_dir="postProcessing/report",
)
document.add_section("Purpose", "Summary of the simulated case.")
document.add_figure("postProcessing/velocity.png", caption="Velocity magnitude")
document.generate_document(output_format="pdf")
```

For document generation without a LaTeX toolchain, the Typst renderer exposes structured building blocks such as sections, equations, figures, tables, code blocks, and bibliographies. Prefer Typst when the project already uses `.typ` templates or when deterministic layout is important.

## Parallel studies

`ParallelStudy` automates a comparison of processor decompositions. It can write `decomposeParDict`, run baseline and parallel cases, parse logs, collect timing and mesh metrics, and export processor-boundary visualisations. OpenFOAM and an MPI runtime must be available on `PATH`.

Before launching a study, make a copy of the case or use a disposable output directory. Parallel runs modify the case by creating processor directories and reconstruction outputs.

## Recommended result layout

A repeatable project can use the following layout:

```text
case_project/
├── run.py
├── case/
│   ├── 0/
│   ├── constant/
│   └── system/
├── logs/
├── postProcessing/
│   ├── figures/
│   ├── tables/
│   └── reports/
└── README.md
```

Keep generated output separate from source geometry and CSV inputs. This makes it possible to remove a case directory and rebuild it from the script without losing the scientific provenance of the run.

## Field types and derived quantities

Post-processing should distinguish between **point data**, **cell data**, and **surface data**. A velocity vector stored at cell centres is not interchangeable with a value interpolated to vertices. Surface pressure and wall shear stress must be integrated on the actual wall patch, while volume averages require cell volumes.

Common derived quantities include:

| Quantity | Typical definition or use |
| --- | --- |
| Velocity magnitude | $|\mathbf{U}|$ for speed maps and threshold zones. |
| Vorticity | $\nabla\times\mathbf{U}$ for rotational structures. |
| Q-criterion | Identifies regions where rotation dominates strain. |
| Wall shear stress | Tangential traction at a wall; sensitive to near-wall mesh. |
| Pressure coefficient | $C_p=(p-p_\infty)/(\tfrac12\rho U_\infty^2)$ for external flows. |
| Heat flux | Normal conductive or total thermal flux at a surface. |
| Nusselt number | Dimensionless heat transfer based on a stated characteristic length. |
| Phase fraction | Interface location and liquid-volume diagnostics in VOF. |
| Scalar mixing index | Uniformity or variance of a transported concentration. |

The definition, reference state, sign convention, and averaging operation must accompany every exported quantity.

## Residuals and convergence

The residual is an algebraic measure of how well a discretised equation is satisfied during an iteration. It is not automatically an error estimate for the physical quantity of interest. A case can have small residuals and an incorrect drag coefficient, heat balance, or outlet flow split.

A robust post-processing report should therefore contain:

1. solver residual histories for every region and field;
2. monitored forces, fluxes, temperatures, or scalar averages;
3. continuity errors and volume conservation;
4. final mesh statistics;
5. the final time, timestep, Courant number, and iteration counts;
6. the convergence criterion used for the engineering output.

`ResidualsPost` can transform solver logs into CSV, JSON, PNG, or HTML artefacts. Keep the original log file because parsed summaries can hide warnings, floating-point exceptions, or solver restarts.

## Boundary and patch analysis

Patch-level analysis is essential for external aerodynamics, biomedical flow, and CHT. A reliable patch report identifies the patch name, patch type, area, number of faces, min/max/mean field values, and integrated flux or force where applicable.

For a vehicle, report forces by patch and by direction. For a vascular model, report flow rate and pressure at every inlet/outlet and verify conservation. For CHT, report heat flux independently on the fluid side and solid side of the interface, with the normal convention made explicit.

## Wind ensembles and multiple cases

The wind-analysis module provides objects such as `WindRose`, `WindCaseResult`, `WindEnsemble`, `LawsonProcessor`, and `LawsonVisualizer`. These can organise several wind directions or atmospheric cases and combine their results into directional summaries. They do not replace the physical definition of the inlet profile or the selection of a comfort criterion.

A wind ensemble should record for every case:

| Metadata | Example |
| --- | --- |
| Direction | Meteorological or Cartesian convention, stated explicitly. |
| Reference speed | Height and averaging period. |
| Atmospheric profile | Log law, power law, measured profile, or precursor field. |
| Stability | Neutral, stable, unstable, or unknown. |
| Solver/model | RANS closure, wall functions, timestep, and discretisation. |
| Weight | Frequency or probability assigned to the case. |

## CHT post-processing

For a CHT case, load all regions at the same physical time. Comparing a fluid field at one time to a solid field at another time can create a false interface mismatch. The direct `CHTDirectReader` can load temperature fields as a `MultiBlock` object; the CHT utilities can calculate interface temperatures, heat fluxes, thermal resistance, heat-transfer coefficients, and Nusselt numbers.

A minimum CHT report should include:

- fluid and solid region names;
- material properties and temperature dependence;
- interface patch pairs;
- temperature continuity at the interface;
- heat-flux continuity at the interface;
- total heat entering, leaving, and stored;
- local and integrated Nusselt numbers;
- mesh resolution normal to the interface;
- the convergence history.

## Data export and provenance

When exporting a field to CSV, JSON, VTK, or images, write a metadata file containing:

```text
case identifier
OpenFOAM version
FoamPilot commit
mesh cell count
physical time
field names and locations
units
coordinate system
filter/interpolation operation
reference values
```

This is especially important for biomedical and urban cases, where a visualisation can be detached from the original geometry, coordinate reference system, or patient/environmental input.

## Report types

FoamPilot supports several report levels:

| Report | Best use |
| --- | --- |
| Residual CSV/PNG/HTML | Fast numerical-health check during development. |
| Mesh-quality report | Geometry and discretisation review before solving. |
| Simulation report | Reproducible case summary with figures and tables. |
| Parallel-study report | Processor-count comparison and decomposition diagnostics. |
| LaTeX PDF | Formal calculation note or publication-style report. |
| Typst document | Structured scientific document without requiring a LaTeX workflow. |
| Streamlit/Plotly dashboard | Interactive exploration for engineers and collaborators. |

Do not use a dashboard as the only archive. Interactive state can be lost; the case script, dictionaries, solver log, raw data, and static summary remain the reproducible record.
