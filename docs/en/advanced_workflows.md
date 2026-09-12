# Advanced workflows

This page gathers workflows that are present in the package but were not previously described in the English documentation. They are more specialised than the basic cavity and external-aerodynamics examples and should be considered experimental unless the corresponding tutorial has been validated with the target OpenFOAM release.

## Conjugate heat transfer

The `foampilot.cht` package builds multi-region cases for `chtMultiRegionFoam` and related solvers. The main objects are `ChtSolver`, `FluidRegion`, `SolidRegion`, and `CoupledInterface`. Boundary-condition factories cover fixed temperature, heat flux, inlet/outlet temperature, symmetry, total temperature, radiation-coupled, and coupled interface conditions.

A CHT case is organised by region:

```text
case/
├── 0/
│   ├── fluid/
│   └── solid/
├── constant/
│   ├── fluid/
│   ├── solid/
│   └── regionInterfaces/
└── system/
```

The generated `controlDict` contains the region solver mapping. A serial run launches the standalone CHT executable; a parallel run decomposes all regions, launches MPI, and reconstructs all regions.

```python
from foampilot.cht import ChtSolver, FluidRegion, SolidRegion

fluid = FluidRegion(name="fluid", temperature=300.0)
solid = SolidRegion(name="solid", temperature=350.0)
solver = ChtSolver(
    case_path="case",
    solver_name="chtMultiRegionFoam",
    regions=[fluid, solid],
)
solver.setup_case()
solver.run_simulation(nb_proc=1)
```

The exact constructor and material arguments depend on the OpenFOAM version used by the case. Use the `09_CHT_heatedDuct` tutorial as the executable reference and inspect the generated region dictionaries before running.

The CHT post-processing helpers can calculate heat flux, interface heat flux, Nusselt number, thermal boundary-layer thickness, heat-transfer coefficient, total heat balance, temperature contours, and thermal resistance.

## Windkessel and physiological utilities

`WindkesselModel` is available from the top-level package for reduced-order cardiovascular boundary modelling. It should be coupled to a clearly defined pressure/flow convention and validated against the intended OpenFOAM boundary condition before production use.

The utilities package also contains vascular and medical-geometry helpers, including NIfTI-to-STL conversion, aorta surface cleaning, mesh optimisation, and a CSV foam integrator. These tools may require optional packages such as NiBabel, Trimesh, VMTK, PyFQMR, or PyACVD.

## Weather and atmospheric inputs

`WeatherFileEPW` reads EnergyPlus Weather (EPW) files. It can be used to extract outdoor temperature, wind, radiation, and other time-series inputs before converting them to FoamPilot boundary conditions or atmospheric forcing. Treat the EPW file as an input dataset and record its source, location, and time zone in the case metadata.

The `foampilot.utilities.wind_profile` and `foampilot.postprocess.wind_analysis` modules provide wind-profile and wind-ensemble helpers. These are useful for comparing multiple wind directions or atmospheric boundary-layer assumptions, but they do not replace a physical calibration of the atmospheric boundary conditions.

## Urban CFD

The `foampilot.urban` package is an experimental pipeline for urban-scale CFD. It exposes data models for buildings, terrain, roads, and CFD domains; geometry simplification and clean-up; Gmsh or surface-based quarter-domain builders; mesh sizing and wake-refinement objects; patch assignment; atmospheric-boundary-layer profiles; and geometry/mesh validation.

A high-level workflow is:

```python
from foampilot.urban import (
    UrbanModel,
    CFDSimplifier,
    MeshConfig,
    ABLProfile,
    GeometryValidator,
)

# Load or construct an UrbanModel from the supported reader.
# Simplify geometry for CFD, build a domain, size the mesh,
# assign patches, validate, then export to the OpenFOAM workflow.
```

OSM and LiDAR readers are optional because they depend on geospatial libraries and external datasets. Install the extra before importing them:

```bash
pip install -e ".[urban]"
```

Urban cases should document the coordinate reference system, metric conversion, wind frame, terrain source, building-height assumptions, simplification tolerance, mesh budget, and atmospheric profile. These details are essential for reproducibility and are not safely inferable from the generated mesh alone.

## MakeHuman and thermoregulation

The repository contains a MakeHuman-to-STL workflow for thermoregulation experiments. The workflow exports a body model, selects the main skin surface, creates JOS-3 surface zones, and writes a zone mapping for later coupling. It is an external workflow rather than a generic FoamPilot solver feature, so the English documentation should point to its README and state its external requirements explicitly.

When using this workflow, record the MakeHuman version, the model pose, the exported surface group, the JOS-3 zone mapping, and the OpenFOAM case used for coupling. Do not interpret a generated STL as a validated physiological model without checking the surface topology and the zone assignment.
