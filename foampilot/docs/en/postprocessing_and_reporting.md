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
