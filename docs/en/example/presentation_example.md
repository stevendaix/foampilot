# FOAMPilot Examples

FoamPilot is a Python library designed to simplify the creation, configuration, and execution of OpenFOAM simulations. It provides a modular and intuitive approach to manage CFD cases, mesh generation, boundary conditions, function objects, and post-processing of results.

This section presents different examples illustrating the advantages and flexibility of FoamPilot for automating CFD workflows and facilitating learning OpenFOAM with Python.

## Objectives of the Examples

The examples aim to:

- Show how to initialize an OpenFOAM case from Python.
- Demonstrate mesh generation and modification using JSON files.
- Illustrate the definition of fluid properties and boundary conditions.
- Set up `functionObjects` to monitor physical quantities (forces, pressure, field averages, etc.).
- Create and manage OpenFOAM-specific dictionaries (`topoSetDict`, `createPatchDict`, etc.).
- Run the simulation and automate post-processing.
- Provide reproducible examples for learning and prototyping.

## List of Examples

This section contains tutorials demonstrating various CFD physics and solver
configurations using FoamPilot.

### Incompressible Solvers

- [Cavity Laminar](cavity_laminar/tutorial.md): Laminar lid-driven cavity flow using `icoFoam`.
- [SimpleCar Turbulent](simplecar_turbulent/tutorial.md): Steady RANS turbulent flow around a simplified car using `simpleFoam` with k-omega SST.
- [PitzDaily Step](pitzdaily_step/tutorial.md): Backward-facing step turbulent flow with recirculation zone analysis.
- [MotorBike](motorBike/tutorial.md): High-speed external aerodynamics around a motorcycle.
- [Building Aero](buildingAero/tutorial.md): Urban boundary layer flow around buildings.

### Multiphase & Scalar

- [DamBreak VOF](dambreak_vof/tutorial.md): Two-phase water-air flow using VOF model with `interFoam`.
- [Scalar Transport](scalarTransport/tutorial.md): Passive scalar (temperature) transport in a channel.

### Buoyancy & Conjugate Heat Transfer

- [Thermal Buoyancy](thermalBuoyancy/tutorial.md): Natural convection in a heated room using Boussinesq approximation.
- [CHT Heated Duct](cht/cht_heated_duct.md): Conjugate heat transfer between air and a copper wall using `chtMultiRegionFoam`.

### Muffler Studies

- [Muffler](muffler/detailled_example_muffler.md): Detailed example of a car muffler with acoustic and fluidic analysis.
- [SimpleCar](simplecar/detailled_example.md): OpenFOAM simpleCar tutorial with JSON-based mesh generation.

## Notes

Each example comes with a standalone Python script that:

1. Defines the case path (`current_path`).
2. Initializes fluid properties (density, viscosity, pressure, temperature, etc.).
3. Initializes the FoamPilot solver and system/constant folders.
4. Generates the mesh from a JSON file.
5. Adds the necessary `functionObjects` (field average, reference pressure, run-time control, etc.).
6. Manipulates OpenFOAM dictionaries for patch creation and zone definition.
7. Applies boundary conditions using the modern API.
8. Runs the simulation.
9. Automatically post-processes results and exports CSV, JSON, PNG, and HTML files.

These examples are designed to be modular and easily adaptable to various CFD case studies.
