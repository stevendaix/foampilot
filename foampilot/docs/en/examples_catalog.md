# Complete examples and tutorials catalogue

This page is the map of the executable examples currently distributed with FoamPilot. Each tutorial is intentionally small enough to inspect, but each one also follows the same reproducible pattern: define the case in Python, generate the OpenFOAM dictionaries, create or import the mesh, apply boundary conditions, run the solver, post-process the result, and archive the generated artefacts.

## How to read a tutorial

Every tutorial should be read at three levels. The **physical level** explains the conservation laws, constitutive assumptions, and dimensionless numbers. The **OpenFOAM level** identifies the solver, fields, dictionaries, boundary conditions, and function objects. The **FoamPilot level** shows which Python objects generate those files and how the workflow can be reproduced or parameterised.

Before running a tutorial, check the OpenFOAM version named in its README, the external geometry or tutorial data required by the script, and the availability of optional Python dependencies. Generated cases must be inspected and validated with `checkMesh` and the solver log before interpreting the plots.

## Summary matrix

| Tutorial | Main physics | Solver family | Mesh strategy | Main outputs |
| --- | --- | --- | --- | --- |
| `01_cavity_laminar` | Incompressible laminar recirculation | `icoFoam` / incompressible transient flow | `blockMesh` | Velocity, pressure, residuals, figures, report |
| `02_simpleCar_turbulent` | Steady external turbulent aerodynamics | `simpleFoam` / incompressible RANS | Gmsh or imported geometry with boundary patches | Velocity, pressure, wall forces, report |
| `03_pitzDaily_step` | Backward-facing-step separation and reattachment | `simpleFoam` | Gmsh or structured channel geometry | Recirculation length, residuals, profiles |
| `04_damBreak_multiphase` | Transient water-air free surface | `interFoam` | Gmsh/block-style 2-D domain | Interface evolution, phase fraction, animation |
| `05_scalarTransport` | Passive scalar or temperature-like transport | `scalarTransportFoam` function object | Channel mesh | Scalar contours, time histories, CSV data |
| `06_buildingAero` | Urban external wind and wakes | `simpleFoam` | `blockMesh` background plus `snappyHexMesh` | Wind field, turbulence, wake and building statistics |
| `07_motorBike` | External vehicle aerodynamics | `simpleFoam` or OpenFOAM-13 `incompressibleFluid` path | `blockMesh` plus `snappyHexMesh` | Drag, pressure, wake, animation, report |
| `08_thermalBuoyancy` | Natural convection with buoyancy | Boussinesq/compressible thermal flow | `blockMesh` | Temperature, `U`, `p_rgh`, residuals, thermal report |
| `09_CHT_heatedDuct` | Conjugate heat transfer in fluid and solid regions | `chtMultiRegionFoam` | `blockMesh`, zones, region splitting | Region temperatures, heat flux, Nusselt number, balance |
| Muffler case study | Internal flow, pressure loss, acoustic/fluidic analysis | Case-specific OpenFOAM workflow | JSON/geometry-driven setup | Pressure, velocity, acoustic or flow report |
| SimpleCar case study | Scripted external-flow case with JSON mesh configuration | Case-specific incompressible workflow | JSON-based mesh generation | Case dictionaries, fields, figures, and report |

## 1. Lid-driven cavity: laminar transient flow

### Purpose

The cavity is the canonical verification case for a viscous incompressible flow. A square cavity contains fluid, the upper wall moves at a prescribed velocity, and the remaining walls are stationary. The case isolates viscous diffusion, pressure-velocity coupling, wall no-slip conditions, and the development of recirculation cells.

### Mathematical model

The tutorial solves the incompressible Navier–Stokes equations:

$$
\nabla\cdot\mathbf{U}=0,
$$

$$
\frac{\partial\mathbf{U}}{\partial t}+\nabla\cdot(\mathbf{U}\mathbf{U})
=-\nabla p+\nu\nabla^2\mathbf{U}.
$$

The Reynolds number is usually kept low enough for a laminar reference solution. The problem is transient because the flow starts from rest and approaches a steady recirculating state.

### FoamPilot workflow

The script creates a two-dimensional `blockMesh` case, writes `controlDict`, `fvSchemes`, `fvSolution`, and `transportProperties`, then applies a moving-lid velocity and no-slip walls. It runs the transient solver, extracts residuals, and generates plots or a report.

### What to verify

The primary verification quantities are the centreline velocity profiles, the number and position of recirculation cells, the residual decay, and the sensitivity to mesh refinement and time step. A visually plausible vortex is not sufficient: compare the centreline profiles with a published benchmark or the OpenFOAM reference case.

## 2. SimpleCar: steady turbulent external aerodynamics

### Purpose

This case introduces an external flow around a simplified vehicle. It demonstrates how a body geometry is placed in a wind-tunnel-like domain, how the inlet turbulence is prescribed, and how pressure and shear forces are extracted from the body surface.

### Model and assumptions

The flow is incompressible and turbulent. A RANS closure, commonly `kOmegaSST` in the case family, replaces the unresolved turbulent stresses by an eddy-viscosity model. The SST model is chosen because it blends near-wall sensitivity from the $k$–$\omega$ family with a more free-stream-tolerant behaviour away from the wall. It is a practical compromise for separation around bluff or streamlined bodies; it is not a substitute for model validation.

The drag coefficient is obtained from:

$$
C_D=\frac{F_D}{\tfrac12\rho U_\infty^2 A_\mathrm{ref}},
$$

where $F_D$ is the streamwise force, $\rho$ is the density, $U_\infty$ is the reference wind speed, and $A_\mathrm{ref}$ is the selected reference area.

### Mesh and boundary conditions

The background domain must be long enough upstream and downstream to avoid contaminating the vehicle pressure field. The car surface requires a named wall patch. Inlet velocity, turbulent kinetic energy, and turbulence dissipation or specific dissipation are prescribed consistently; the outlet should avoid artificial reflection; the ground treatment must match whether the vehicle is stationary, moving, or represented in a moving-ground frame.

### Post-processing

Extract surface pressure, wall shear stress, integrated force coefficients, separation regions, and wake velocity profiles. Always report the reference area, reference velocity, density, turbulence model, wall treatment, and mesh statistics together with $C_D$.

## 3. PitzDaily: backward-facing step

### Purpose

The backward-facing step is a separated internal flow used to study shear-layer development, recirculation, reattachment, and turbulence-model sensitivity.

### Physics

The inlet flow crosses a sudden expansion. A separation bubble forms behind the step, and the reattachment length depends on Reynolds number, inlet profile, turbulence model, wall resolution, and numerical schemes. The case is steady in its nominal solver configuration, but the separated flow may exhibit unsteady behaviour if the mesh, timestep, or model allows it.

### Main diagnostics

The most important output is the reattachment length, normally expressed relative to the step height. Complementary diagnostics are wall pressure, wall shear stress, centreline velocity, recirculation-zone length, and residual histories. The result should not be judged from residuals alone because a steady solver can converge to a numerically stable but physically biased solution.

## 4. DamBreak: transient multiphase VOF

### Purpose

The DamBreak case demonstrates a transient free-surface problem. A water column collapses under gravity and displaces air. The interface is represented by a phase fraction field, normally `alpha.water`.

### Governing model

The VOF approach solves a transport equation for the phase fraction:

$$
\frac{\partial\alpha}{\partial t}+\nabla\cdot(\alpha\mathbf{U})=0,
$$

with interface-compression and boundedness controls. The mixture density and viscosity are reconstructed from the phase fractions. Gravity drives the collapse and pressure must be interpreted consistently with the hydrostatic contribution.

### Numerical priorities

The Courant number, interface compression, bounded phase fraction, and time-step adaptation are more important than simply increasing the number of iterations. Inspect `alpha.water` at several times, check that $0\leq\alpha\leq1$, and verify that the liquid volume is conserved within the expected numerical tolerance.

### Outputs

The tutorial is suitable for exporting interface snapshots, animations, free-surface height histories, pressure fields, and residuals. The same pattern can be reused for sloshing, filling, draining, or wave-impact cases, but each application requires a separate validation of surface tension, wetting, and contact-line assumptions.

## 5. Scalar transport

### Purpose

This case transports a passive scalar through a channel. The scalar can represent a concentration, tracer, pollutant, or a temperature-like quantity when the energy equation is deliberately simplified.

### Equation

For a constant diffusivity $D$:

$$
\frac{\partial C}{\partial t}+\nabla\cdot(\mathbf{U}C)
=\nabla\cdot(D\nabla C)+S_C.
$$

The scalar does not modify the flow unless a coupled buoyancy, density, reaction, or source model is added. This separation makes the case useful for testing advection-diffusion numerics and CSV-driven boundary conditions.

### Diagnostics

Compare the scalar profile with the expected convective-diffusive length scale. Report the scalar Peclet number, inlet profile, diffusivity, outlet treatment, boundedness, and numerical scheme. If the scalar represents temperature, state clearly whether it is a passive field or a fully coupled thermal model.

## 6. Building aerodynamics: external urban wind

### Purpose

The building case introduces a group of obstacles in an atmospheric or wind-tunnel-like domain. It illustrates the difference between a background hexahedral mesh and local surface-based refinement.

### Physical model

The flow is usually incompressible and turbulent. For a first engineering model, a steady RANS closure such as $k$–$\epsilon$ or $k$–$\omega$ SST is often chosen because it gives a manageable cost for mean wind and wake predictions. The model does not resolve all transient eddies; it predicts their averaged effect through turbulent viscosity.

For atmospheric applications, a uniform inlet is only acceptable when the physical problem is a controlled wind tunnel. For a real atmospheric boundary layer, the inlet velocity and turbulence fields must be height-dependent and mutually consistent. See [Outdoor wind theory](theory_applied.md#outdoor-wind-and-atmospheric-boundary-layers).

### Mesh workflow

The typical sequence is:

```text
background blockMesh
→ surfaceFeatureExtract
→ snappyHexMesh castellated mesh
→ snap to building surfaces
→ optional boundary layers
→ checkMesh and patch validation
```

The buildings, ground, inlet, outlet, side boundaries, and top boundary must have stable names. Refinement should be concentrated around building edges, roof lines, canyon passages, and wake regions rather than applied uniformly.

### Outputs

Useful outputs include pedestrian-level wind speed, velocity vectors, pressure, turbulent kinetic energy, roof and street-canyon recirculation, and statistics on selected building patches. Report the inlet profile, roughness assumptions, wall functions, domain extents, refinement levels, and cell count.

## 7. MotorBike: complex external geometry

### Purpose

The MotorBike example is a more demanding surface-based external-aerodynamics case. It tests geometry import, feature extraction, surface snapping, local refinement, wall patches, force integration, and animation.

### Model choice

The repository scripts and README contain version-dependent references. Inspect the actual script before running: some configurations use a `simpleFoam`/`incompressibleFluid` pathway, while the script documentation also refers to a Spalart–Allmaras RAS model. The selected solver, turbulence model, wall treatment, and geometry source must be recorded in the generated case.

Spalart–Allmaras is attractive for attached or moderately separated external aerodynamic flows because it is relatively inexpensive and solves one transported turbulence variable. $k$–$\omega$ SST may be preferred when separation behaviour and robustness across adverse pressure gradients are more important. Neither choice is universally superior; the mesh and validation data often dominate the uncertainty.

### Mesh and validation

Use a coarse mesh to validate geometry orientation and patch names, then refine the leading edges, wheels, fairings, ground contact, and wake. Verify that the surface has no leaks or inverted normals and that the local cell size supports the intended wall treatment. Compare drag and pressure distributions only after the force reference quantities are fixed.

## 8. Thermal buoyancy: natural convection

### Purpose

The thermal-buoyancy example models a heated room or cavity with gravity, temperature differences, and buoyancy-driven flow.

### Boussinesq approximation

For moderate temperature differences, density variations can be neglected in the continuity and inertia terms and retained only in the buoyancy force. A typical relation is:

$$
\rho\approx\rho_0[1-\beta(T-T_0)],
$$

and the buoyancy contribution is proportional to $\rho_0\beta(T-T_0)\mathbf{g}$. This is computationally cheaper than a fully compressible ideal-gas treatment but should not be used when density changes are large, compressibility matters, or the temperature range invalidates the linear approximation.

### Boundary conditions and diagnostics

The case prescribes hot and cold walls, adiabatic or insulated surfaces, gravity, and a thermal turbulence model. Monitor $T$, $U$, `p_rgh`, $k$, $\epsilon$ or $\omega$, and `alphat` as applicable. The principal dimensionless groups are the Rayleigh number, Prandtl number, and Nusselt number. Validate temperature differences, circulation cells, and heat-transfer rates against a benchmark where possible.

## 9. Heated duct: conjugate heat transfer

The heated-duct case is documented in detail in [CHT case setup](cht_workflow.md). It is the reference example for fluid-solid region creation, region-specific fields, material properties, coupled interfaces, `chtMultiRegionFoam`, direct or VTK post-processing, and thermal-balance reporting.

## 10. Muffler case study

The muffler case is a larger, more application-oriented example. It demonstrates how FoamPilot can combine geometry handling, internal-flow modelling, pressure-loss analysis, acoustic or fluidic post-processing, and report generation. The relevant page is [Detailed muffler example](example/muffler/detailled_example_muffler.md).

The important modelling decisions are the internal volume and perforated or connected passages, inlet and outlet pressure/flow data, wall roughness assumptions, compressibility or incompressibility, and the frequency range if acoustic quantities are interpreted. A pressure field alone is not an acoustic prediction; the acoustic assumptions and sampling strategy must be documented.

## 11. SimpleCar case study

The detailed SimpleCar page complements the executable turbulent tutorial. It focuses on a JSON-driven case setup, mesh configuration, OpenFOAM dictionary manipulation, boundary conditions, and automated reporting. Use it when learning how a project-level script can generate a complete case rather than only reproducing a small benchmark.

## 12. Thermal and geometry add-on examples

The repository also contains specialised examples and utilities around geometry conversion, aorta surface processing, weather/EPW inputs, wind profiles, human geometry, MakeHuman/JOS-3 thermoregulation, and CSV coupling. These are not all equivalent to solver tutorials: some are preprocessing or data-exchange workflows. Their documentation must therefore state the input-data format, coordinate system, external software, generated artefacts, and validation checks.

## Tutorial artefacts and reproducibility

The tutorial directories may contain run scripts, geometry files, residual exports, images, animations, and generated reports. Keep generated results separate from source inputs when adapting a tutorial. Record the OpenFOAM release, Python environment, mesh count, solver settings, convergence criteria, and any manual changes to generated dictionaries.

## What each tutorial does not prove

A tutorial demonstrates a workflow; it does not establish industrial accuracy. Accuracy requires mesh convergence, time-step or Courant-number studies, model sensitivity, conservation checks, comparison with analytical or experimental data, and an uncertainty statement. The more complex the geometry or physiology, the more important these checks become.
