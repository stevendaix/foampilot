# Conjugate heat transfer: data setup, execution, and validation

Conjugate heat transfer (CHT) couples heat conduction in solids with heat transport in one or more fluid regions. FoamPilot provides Python objects for the region definitions, thermophysical dictionaries, temperature fields, interfaces, control dictionaries, execution, and thermal post-processing. OpenFOAM remains responsible for solving the coupled finite-volume equations.

## 1. Physical problem

A CHT case contains at least one fluid and one solid region. In the fluid, the solver resolves momentum, continuity, and energy. In the solid, it resolves heat conduction; there is no fluid velocity field in the solid.

For a compressible fluid, the energy equation is written in a thermodynamically consistent form using the selected enthalpy or internal-energy formulation. A simplified temperature form is:

$$
\rho c_p\left(\frac{\partial T}{\partial t}+\mathbf{U}\cdot\nabla T\right)
=\nabla\cdot(k_f\nabla T)+S_T,
$$

where $\rho$ is density, $c_p$ is heat capacity, $k_f$ is fluid conductivity, and $S_T$ represents sources. In a solid at rest:

$$
\rho_s c_{p,s}\frac{\partial T_s}{\partial t}
=\nabla\cdot(k_s\nabla T_s)+S_s.
$$

At a perfect fluid-solid interface:

$$
T_f=T_s,
$$

and

$$
-k_f\nabla T_f\cdot\mathbf{n}
=-k_s\nabla T_s\cdot\mathbf{n}.
$$

The first condition expresses temperature continuity. The second expresses continuity of heat flux. If contact resistance, radiation, roughness, or a thin coating is physically important, the interface model must be changed rather than hidden in a mesh adjustment.

## 2. Data model in FoamPilot

The CHT API is organised around four concepts:

| Object | Role |
| --- | --- |
| `FluidRegion` | Initial temperature and velocity, turbulence model, thermo model, equation of state, mixture, and transport model for a fluid region. |
| `SolidRegion` | Initial temperature and solid material properties for a conducting region. |
| `CoupledInterface` | Pairing of fluid and solid patches and the interface dictionary. |
| `ChtSolver` | Case path, solver executable, regions, interfaces, region solver mapping, setup, and serial/parallel execution. |

The minimal setup pattern is:

```python
from foampilot.cht import (
    ChtSolver,
    FluidRegion,
    SolidRegion,
    CoupledInterface,
)

fluid = FluidRegion(
    name="fluid",
    temperature=300.0,
    velocity=(1.0, 0.0, 0.0),
    turbulence_model="kOmegaSST",
)

solid = SolidRegion(
    name="solid",
    temperature=350.0,
)

interface = CoupledInterface(
    name="fluid_to_solid",
    fluid_region="fluid",
    solid_region="solid",
)

solver = ChtSolver(
    case_path="case",
    solver_name="chtMultiRegionFoam",
    regions=[fluid, solid],
    interfaces=[interface],
)
solver.setup_case()
solver.run_simulation(nb_proc=1)
```

The concrete material parameters must be filled in for the target problem. Do not use the defaults as validated material data.

## 3. Case directory and region data

A multi-region case is not a single-region case with an extra temperature field. The directory layout carries physical meaning:

```text
case/
├── 0/
│   ├── fluid/
│   │   ├── T
│   │   ├── U
│   │   ├── p
│   │   ├── p_rgh
│   │   └── turbulence fields
│   └── solid/
│       └── T
├── constant/
│   ├── fluid/
│   │   ├── thermophysicalProperties
│   │   └── turbulenceProperties
│   ├── solid/
│   │   ├── thermophysicalProperties
│   │   └── transportProperties
│   └── regionInterfaces/
│       └── fluid_to_solid.dict
└── system/
    ├── controlDict
    ├── regionProperties or region solver mapping
    ├── fvSchemes
    ├── fvSolution
    └── createZonesDict
```

The precise dictionary names vary with OpenFOAM release and solver family. The repository’s `09_CHT_heatedDuct` tutorial is the reference for the version it targets. Always inspect generated dictionaries because `foamSetupCHT`, `splitMeshRegions`, and region-specific conventions differ between releases.

## 4. Geometry and region definition

The heated-duct tutorial uses a background mesh and a cell-zone definition to distinguish the fluid from the heated solid. The conceptual sequence is:

```text
blockMesh
→ createZones
→ splitMeshRegions -cellZones
→ create region dictionaries and fields
→ run chtMultiRegionFoam
```

The cell-zone definition must cover the intended solid volume without gaps or overlaps. A failed region split can create a case that appears to contain a solid directory but does not represent the intended physical domain.

Validate the region topology with:

```bash
checkMesh -case case -region fluid
checkMesh -case case -region solid
splitMeshRegions -cellZones -overwrite -case case
```

Use the command appropriate to the installed OpenFOAM distribution and do not run a destructive `-overwrite` operation on the only copy of a source case.

## 5. Fluid-region data

The fluid data must specify enough information to close the momentum and energy equations:

| Data | Meaning | Typical choices |
| --- | --- | --- |
| Density law | Relation between density, pressure, and temperature. | Perfect gas, constant density, Boussinesq where supported. |
| Heat capacity | Energy storage. | Constant or temperature-dependent $c_p$. |
| Conductivity | Molecular heat diffusion. | Constant or temperature-dependent $k_f$. |
| Viscosity | Momentum diffusion. | Constant or temperature-dependent dynamic viscosity. |
| Equation of state | Pressure-density relation. | Ideal gas or incompressible approximation. |
| Turbulence | Closure for unresolved flow. | Laminar, $k$–$\epsilon$, $k$–$\omega$ SST, or version-specific RAS model. |
| Initial velocity | Starting flow field. | Zero or prescribed bulk velocity. |
| Initial temperature | Starting thermal field. | Inlet temperature, wall temperature, or a uniform estimate. |

For an air duct, the fluid region is commonly treated as compressible in CHT because temperature affects density and energy. If the temperature range is small and the intended solver supports it, an incompressible thermal formulation may be more appropriate, but it must be physically justified.

## 6. Solid-region data

The solid region requires at least:

- density $\rho_s$;
- heat capacity $c_{p,s}$;
- thermal conductivity $k_s$;
- initial temperature;
- external and interface thermal boundary conditions.

For a copper wall, high conductivity produces a relatively small temperature gradient through the solid compared with a low-conductivity insulation layer. This does not mean the solid can be removed: its thickness and thermal resistance still determine the heat flux.

The solid model can be extended with temperature-dependent properties, radiation, heat generation, or anisotropic conductivity if the solver and generated dictionaries support them.

## 7. Boundary-condition data

The boundary conditions must be defined separately for each region. Common fluid conditions are:

| Boundary | Fluid fields |
| --- | --- |
| Inlet | Velocity, temperature, pressure, turbulence quantities. |
| Outlet | Pressure and compatible velocity/temperature outflow conditions. |
| Fluid-solid interface | Coupled temperature and heat-flux condition. |
| Symmetry | `symmetryPlane` for all compatible fields. |
| Adiabatic wall | Zero heat flux, with no-slip or the selected wall velocity treatment. |

Common solid conditions are:

| Boundary | Solid field |
| --- | --- |
| Coupled interface | Temperature and heat flux coupled to the fluid region. |
| Fixed-temperature wall | Prescribed `T`. |
| Heat-flux wall | Prescribed normal heat flux. |
| Adiabatic wall | Zero gradient or solver-specific insulated condition. |
| Radiation wall | Radiation-coupled condition when radiation is included. |

FoamPilot exposes factories such as `get_coupled_temperature_bc`, `get_fixed_temperature_bc`, `get_heat_flux_bc`, `get_inlet_outlet_bc`, and `get_symmetry_bc`. The generated condition must still be checked against the installed OpenFOAM version.

## 8. Solver execution

The CHT solver is a standalone multi-region executable in the common OpenFOAM workflow. FoamPilot’s serial path launches the solver with the case path. Its parallel path decomposes all regions, runs MPI, and reconstructs all regions.

```python
solver.run_simulation(
    nb_proc=1,
    log_filename="log.chtMultiRegionFoam",
)
```

For parallel execution:

```python
solver.run_simulation(nb_proc=8)
```

Before a parallel run, confirm that MPI, `decomposePar`, and `reconstructPar` are compatible with the OpenFOAM build. Save the serial baseline because it is needed to distinguish a decomposition issue from a physical or numerical issue.

## 9. Convergence and conservation

A CHT run should be considered converged only when several conditions are satisfied:

1. residuals decrease to the selected tolerances;
2. region temperatures and heat fluxes reach stable trends;
3. interface temperature continuity is acceptable;
4. the total heat entering and leaving the coupled system is balanced;
5. the result is insensitive to a reasonable reduction of timestep or increase in iterations;
6. the engineering quantity of interest is stable.

The `foampilot.cht.postprocess` helpers include calculations for region heat flux, interface heat flux, Nusselt number, thermal boundary-layer thickness, heat-transfer coefficient, total heat balance, temperature contours, and thermal resistance.

## 10. Heated-duct example data

The repository tutorial uses a fluid region initially near 300 K and a heated solid region near 350 K. Its input data includes a `block_mesh.json`, a Python case builder, a post-processing script, region-specific field files, and material dictionaries. The expected outputs include region temperature statistics, fluid temperature profiles, combined profiles, and fluid/solid temperature contours.

The values in the tutorial are demonstration data. For a real duct, replace them with measured or design values for:

- mass flow or inlet velocity;
- inlet temperature and pressure;
- wall thickness and material;
- external heat source or wall temperature;
- fluid composition and property correlations;
- contact resistance, if applicable;
- domain length sufficient for the thermal development region.

## 11. Heat-transfer interpretation

The local convective coefficient is often defined by:

$$
 h=\frac{q''}{T_w-T_b},
$$

where $q''$ is wall heat flux, $T_w$ is wall temperature, and $T_b$ is a suitable bulk-fluid temperature. The Nusselt number is:

$$
 Nu=\frac{hL}{k_f}.
$$

The characteristic length $L$ must be stated: hydraulic diameter, duct height, local distance from the inlet, or another physically meaningful scale. A Nusselt number without its length scale and boundary condition is incomplete.

## 12. Common CHT failure modes

| Symptom | Likely cause |
| --- | --- |
| Missing region fields | Region directories were not created or the setup was interrupted. |
| Solver cannot find a material property | Region-specific `thermophysicalProperties` is incomplete or named differently for the OpenFOAM release. |
| Interface temperature jumps | Wrong patch pairing, non-conformal mapping, or incompatible interface conditions. |
| Heat balance is not closed | Insufficient convergence, incorrect flux sign, missing boundary heat loss, or an unintended source. |
| Fluid temperature is unstable | Time step, energy relaxation, thermo model, or boundary condition is inconsistent. |
| Solid temperature is uniform when it should not be | Conductivity, thickness, source, or region geometry is wrong; check the generated solid mesh. |
| Parallel case differs from serial case | Decomposition, reconstruction, processor boundary treatment, or insufficient convergence. |

## References

[1]: https://doc.openfoam.com/2306/tools/processing/solvers/rtm/heat-transfer/chtMultiRegionFoam/ "OpenFOAM documentation: chtMultiRegionFoam"

[2]: https://openfoamwiki.net/index.php/Getting_started_with_chtMultiRegionSimpleFoam_-_planeWall2D "OpenFOAM Wiki: getting started with multi-region heat transfer"
