# FoamPilot documentation

FoamPilot is a Python orchestration layer for **OpenFOAM**. It helps engineers and researchers generate cases, configure physics, run solvers, inspect native results, and produce reproducible reports without manually maintaining every dictionary file.

> FoamPilot does not replace OpenFOAM. It makes OpenFOAM workflows programmable, inspectable, and repeatable.

## Choose a starting point

| If you want to… | Start here |
| --- | --- |
| Install the simulation environment | [OpenFOAM installation](install_openfoam.md) |
| Understand the project’s design | [Architecture and workflow](architecture.md) |
| Build a first case | [User guide](user_documentation.md) and the [complete examples catalogue](examples_catalog.md) |
| Choose and validate a mesh | [Meshing strategy and cases](meshing_cases.md) |
| Read results without `foamToVTK` | [Post-processing and reporting](postprocessing_and_reporting.md) |
| Set up a fluid-solid heat-transfer case | [CHT workflow and data setup](cht_workflow.md) |
| Study biomedical flow, outdoor wind, or thermoregulation | [Applied theory](theory_applied.md) |
| Configure CHT, urban CFD, weather, or physiological utilities | [Advanced workflows](advanced_workflows.md) |
| Extend FoamPilot or contribute code | [Developer guide](dev/dev_index.md) |

## Main capabilities

FoamPilot covers case generation, boundary conditions, `blockMesh`, Gmsh and snappyHexMesh workflows, serial and parallel execution, CSV-driven conditions, CHT multi-region setup, direct OpenFOAM readers, PyVista/Plotly visualisation, mesh and convergence reporting, and specialised geometry or atmospheric utilities. The documentation now also explains the governing laws, data requirements, model-selection reasons, validation targets, and limitations behind the examples.

The English documentation is the primary maintained reference. Tutorial pages are executable guides where available; always inspect the generated files and validate the case with the OpenFOAM tools installed on your system.
