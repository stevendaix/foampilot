# FOAMPilot Examples

FOAMPilot is a Python library designed to simplify the creation, configuration, and execution of OpenFOAM simulations. It provides a modular and intuitive approach to manage CFD cases, mesh generation, boundary conditions, function objects, and post-processing of results.

This section presents different examples illustrating the advantages and flexibility of FOAMPilot for automating CFD workflows and facilitating learning OpenFOAM with Python.

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

This section will be updated as more tests are developed.

- [Muffler](muffler/detailled_example_muffler.md): A detailed example of a car muffler, showing complex mesh generation, boundary conditions, and analysis of acoustic and fluidic results.  
- [SimpleCar](simplecar/detailled_example.md): Example based on the official OpenFOAM tutorial [SimpleCar](https://develop.openfoam.com/Development/openfoam/-/tree/30d2e2d3cfd2c2f268dd987b413dbeffd63962eb/tutorials/incompressible/simpleFoam/simpleCar), illustrating the simulation of airflow around a simple car with JSON-based mesh generation, boundary condition application, and aerodynamic force monitoring.

## Notes

Each example comes with a standalone Python script that:

1. Defines the case path (`current_path`).
2. Initializes fluid properties (density, viscosity, pressure, temperature, etc.).
3. Initializes the FOAMPilot solver and system/constant folders.
4. Generates the mesh from a JSON file.
5. Adds the necessary `functionObjects` (field average, reference pressure, run-time control, etc.).
6. Manipulates OpenFOAM dictionaries for patch creation and zone definition.
7. Applies boundary conditions using the modern API.
8. Runs the simulation.
9. Automatically post-processes results and exports CSV, JSON, PNG, and HTML files.

These examples are designed to be modular and easily adaptable to various CFD case studies.
