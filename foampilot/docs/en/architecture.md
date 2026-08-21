# Architecture and workflow

FoamPilot is a **Python orchestration layer around OpenFOAM**. It does not replace OpenFOAM’s solvers, meshing utilities, or file formats. Instead, it provides Python objects that create, inspect, execute, and post-process an OpenFOAM case.

> A FoamPilot case should be treated as a reproducible build artifact: the Python script is the source of truth, while `0/`, `constant/`, and `system/` are generated inputs and simulation outputs.

## End-to-end workflow

A typical workflow has six stages:

| Stage | FoamPilot responsibility | Main outputs |
| --- | --- | --- |
| Define | Create a `Solver`, physical properties, and boundary-condition objects. | Python configuration |
| Mesh | Generate or import a mesh using `blockMesh`, Gmsh, snappyHexMesh, or a direct OpenFOAM mesh. | `constant/polyMesh` and mesh dictionaries |
| Configure | Write `controlDict`, discretisation schemes, linear solvers, transport properties, turbulence, gravity, and optional function objects. | `system/` and `constant/` |
| Run | Launch a serial or parallel OpenFOAM solver and keep the log in the case directory. | Time directories and log files |
| Inspect | Read native OpenFOAM results directly or convert them to VTK for PyVista. | PyVista meshes and derived fields |
| Report | Generate plots, dashboards, CSV summaries, LaTeX PDFs, or Typst documents. | Figures, tables, and reports |

The stages are deliberately explicit. A script can stop after mesh generation, modify a generated dictionary, or rerun only post-processing without rebuilding the case.

## Package map

The public package is organised by responsibility rather than by OpenFOAM executable:

| Package | Purpose |
| --- | --- |
| `foampilot.base` | Case paths, file abstractions, and meshing orchestration. |
| `foampilot.solver` | Solver selection, case setup, execution, decomposition, and reconstruction. |
| `foampilot.boundaries` | Patch assignment, standard boundary conditions, raw dictionaries, and CSV-driven conditions. |
| `foampilot.constant` | Fluid, turbulence, gravity, phase, radiation, and material dictionaries. |
| `foampilot.system` | `controlDict`, `fvSchemes`, `fvSolution`, function objects, constraints, models, and decomposition. |
| `foampilot.cht` | Multi-region conjugate heat transfer with fluid/solid regions and interface conditions. |
| `foampilot.mesh` and `foampilot.openfoam` | Mesh generation, direct mesh export, Gmsh, and snappyHexMesh helpers. |
| `foampilot.postprocess` | PyVista post-processing, native OpenFOAM readers, wind analysis, and web presentations. |
| `foampilot.report` | Mesh reports, convergence reports, parallel studies, LaTeX, and Typst rendering. |
| `foampilot.urban` | Experimental urban CFD data models, simplification, geometry, meshing, patches, validation, and OSM readers. |
| `foampilot.utilities` | Units, fluid properties, residuals, weather files, geometry conversion, and coupling utilities. |

## Generated files and validation

FoamPilot writes OpenFOAM dictionaries through Python file objects. After every generation step, inspect the resulting files rather than relying only on the in-memory attributes. In particular, verify that `system/controlDict`, all initial fields in `0/`, the mesh under `constant/polyMesh`, and the relevant material dictionaries were written.

For incompressible cases, `constant/transportProperties` must contain the dynamic values used by the solver, including the kinematic viscosity `nu`. If a value is assigned dynamically but does not appear in the generated dictionary, treat the case as invalid and check the corresponding constant-directory writer before launching OpenFOAM.

A minimal validation sequence is:

```bash
checkMesh -case path/to/case
foamDictionary path/to/case/constant/transportProperties -entry nu
foamDictionary path/to/case/system/controlDict -entry application
```

The exact validation commands depend on the OpenFOAM distribution. FoamPilot can generate files, but OpenFOAM remains the authority for dictionary syntax, mesh validity, and solver compatibility.

## Optional dependencies

The base installation and optional extras are separated in `pyproject.toml`. The `docs` extra installs MkDocs, the `dev` extra installs the test and lint tools, `gnn` contains graph-learning dependencies, and `urban` contains geospatial readers such as OSMnx, GeoPandas, Rasterio, and LAS/LAZ support.

```bash
pip install -e ".[dev,docs]"
# Optional urban workflows
pip install -e ".[urban]"
```

Some specialised utilities also require system applications or external datasets. Check the relevant example before running a workflow in a clean environment.
