# Conceptual and Theoretical Overview of the foampilot Module

The `foampilot` module is designed as an object-oriented Python layer for the computational fluid dynamics (CFD) platform **OpenFOAM**. Its main purpose is to abstract the complexity of OpenFOAM case configuration, which traditionally relies on text-based dictionary files, by providing an intuitive and robust Python programming interface (API).

The architecture of `foampilot` closely mirrors the structure of an OpenFOAM case, with each submodule managing a fundamental aspect of the simulation.

## 1. foampilot.solver: The Core of Physics and Numerics

The `solver` submodule is the central control of the simulation. It does not just run an OpenFOAM solver; it handles **dynamic solver selection** based on physical and numerical properties defined by the user.

| OpenFOAM Concept | Role in foampilot | Theoretical Description |
| :--- | :--- | :--- |
| **Solver** | **`Solver` Class** | Acts as an **intelligent solver manager**. By modifying boolean properties (e.g., `compressible`, `transient`, `is_vof`), it automatically selects the appropriate OpenFOAM solver corresponding to the physical equations to solve (Navier-Stokes for incompressible flow, equation of state for compressible flow, etc.). |
| **Physical Configuration** | **Solver Properties** | Defines the problem type (steady vs. transient, single-phase vs. multiphase VOF, with or without gravity/energy). This abstraction ensures that the user does not need to worry about the exact OpenFOAM solver name, only the physics of the problem. |

## 2. foampilot.constant: Definition of the Physical Medium

The `constant` submodule manages OpenFOAM's `constant` directory, which contains fluid and mesh properties.

| OpenFOAM Concept | Role in foampilot | Theoretical Description |
| :--- | :--- | :--- |
| **Fluid Properties** | **`transportProperties`, `physicalProperties` Classes** | Define essential fluid properties (kinematic viscosity $\nu$, density $\rho$, specific heat $C_p$, etc.). These are crucial for closing the Navier-Stokes equations and modeling transport phenomena. |
| **Turbulence Model** | **`turbulenceProperties` Class** | Handles the selection and configuration of turbulence models (e.g., $k-\epsilon$, $k-\omega$ SST). These models add transport equations to account for turbulent effects, closing the RANS (Reynolds-Averaged Navier-Stokes) system. |
| **Gravity** | **`gravityFile` Class** | Allows activation and definition of the gravity vector $\mathbf{g}$, essential for buoyancy (Boussinesq) or free-surface flow simulations. |

## 3. foampilot.system: Numerical and Temporal Control

The `system` submodule manages OpenFOAM's `system` directory, which dictates how equations are discretized and solved.

| OpenFOAM Concept | Role in foampilot | Theoretical Description |
| :--- | :--- | :--- |
| **Simulation Control** | **`controlDictFile` Class** | Defines temporal parameters (time step $\Delta t$, start/end time), result writing frequencies, and execution functions (e.g., `runTimeControl` for automatic stop). |
| **Numerical Schemes** | **`fvSchemesFile` Class** | Handles discretization of equation terms (time derivatives, convection, diffusion). The choice of schemes (e.g., Euler for time, `upwind` or `Gauss linear` for convection) directly affects numerical stability and accuracy. |
| **Algebraic Solvers** | **`fvSolutionFile` Class** | Configures matrix solvers for linear systems resulting from discretization (e.g., `PCG` for pressure, `BiCGStab` for velocity). It also defines convergence criteria (tolerance) and under-relaxation strategies. |

## 4. foampilot.mesh: Mesh Generation

The `mesh` submodule is responsible for mesh creation, i.e., spatial discretization of the computational domain.

| OpenFOAM Concept | Role in foampilot | Theoretical Description |
| :--- | :--- | :--- |
| **Structured Mesh** | **`BlockMeshFile` Class** | Abstracts OpenFOAM's `blockMesh` utility. Allows defining the domain geometry using hexahedral blocks, an efficient method for simple or parameterized geometries. |
| **Unstructured Mesh** | **`gmsh_mesher`, `snappymesh` Classes** | Handle integration with advanced meshing tools (`Gmsh`, `snappyHexMesh`) for complex geometries (e.g., STL files). They generate the necessary configuration files for these utilities. |

## 5. foampilot.boundaries: Physical Boundary Conditions

The `boundaries` submodule is crucial to define the interaction of the fluid with its environment.

| OpenFOAM Concept | Role in foampilot | Theoretical Description |
| :--- | :--- | :--- |
| **Boundary Conditions** | **`Boundary` Class** | Manages boundary conditions for each physical field ($\mathbf{U}$, $p$, $k$, $\epsilon$, etc.) on mesh patches. Theoretically, these are necessary to provide missing information at domain boundaries, ensuring unique PDE solutions. |
| **Condition Types** | **`Boundary` Methods** | Provides methods for common physical conditions: `set_velocity_inlet` (Dirichlet for velocity), `set_pressure_outlet` (Neumann for pressure), `set_wall` (no-slip or slip), etc. |
| **Wall Functions** | **Automatic Integration** | Integrates appropriate wall functions for turbulence models, allowing boundary layer modeling without an extremely fine mesh near walls. |

## 6. foampilot.postprocess and foampilot.report: Result Analysis

These submodules manage the post-simulation phase: data extraction, analysis, and presentation.

| Submodule | Conceptual Role | Theoretical Description |
| :--- | :--- | :--- |
| **`postprocess`** | **Visualization and Analysis** | Uses libraries like **PyVista** to load OpenFOAM results (VTK files) and perform standard post-processing operations (slices, contours, vectors, streamlines). Allows visualization of physical fields and derived quantities (e.g., Q criterion, vorticity). |
| **`report`** | **Report Generation** | Automates creation of structured simulation reports (e.g., PDF). Aggregates key data (input parameters, convergence residuals, post-processing images) for traceability and reproducibility. |

## 7. foampilot.utilities and foampilot.commons: Cross-Cutting Tools

These modules provide essential supporting functionality for the entire framework.

| Submodule | Conceptual Role | Theoretical Description |
| :--- | :--- | :--- |
| **`utilities.manageunits`** | **Unit Management** | Uses the `ValueWithUnit` class to ensure dimensional consistency of inputs. This practice is essential in physics and engineering to avoid conversion errors and make the code unit-system independent (SI, Imperial, etc.). |
| **`utilities.dictonnary`** | **OpenFOAM Dictionary Handling** | Provides tools to create and manipulate complex data structures corresponding to OpenFOAM dictionary files (e.g., `topoSetDict`, `createPatchDict`). |
| **`commons`** | **Generic Utilities** | Contains functions for class serialization, reading mesh files (`polyMesh/boundary`), and other low-level operations necessary to interface with OpenFOAM file formats. |

This document provides an overview of how `foampilot` structures and manages fundamental OpenFOAM simulation concepts, translating them into a modular and intuitive Python architecture.
