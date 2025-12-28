# Developer Quickstart Documentation: `foampilot` Module

## 1. Introduction

The `foampilot` module is designed to simplify and automate the creation, configuration, execution, and post-processing of simulation cases based on **OpenFOAM**. It provides an object-oriented Python interface to manage complex OpenFOAM configuration files, allowing developers to focus on the physics and geometry of the problem rather than dictionary syntax.

This documentation aims to give a detailed overview of the code structure to facilitate understanding and contribution to the project.

## 2. Key Concepts and Architecture

The architecture of `foampilot` closely mirrors the structure of a standard OpenFOAM case, which is divided into three main directories: `constant`, `system`, and the initial time directory (`0`).

### Central Class: `Solver`

The `Solver` class (`foampilot.solver.Solver`) is the central orchestrator of the module. It encapsulates the entire simulation case and provides access points to the `constant` and `system` directories, as well as boundary condition management.

Upon initialization, a `Solver` instance automatically creates objects needed to manage configuration files:

* `solver.constant`: An instance of `ConstantDirectory` to manage files in the `constant` directory.
* `solver.system`: An instance of `SystemDirectory` to manage files in the `system` directory.
* `solver.boundary`: An instance of `Boundary` to manage boundary conditions.

## 3. Detailed Code Structure

The core logic of `foampilot` is located in the `foampilot/foampilot/src/foampilot` directory. This directory is organized into logical submodules, each responsible for a specific aspect of OpenFOAM case management.

| Source Directory | Description | Key Classes (Examples) |
| :--- | :--- | :--- |
| `base` | Base classes and utilities for OpenFOAM file handling. | `Meshing`, `OpenFOAMFile` |
| `solver` | Contains the main `Solver` class and simulation execution logic. | `Solver`, `BaseSolver` |
| `constant` | Manages configuration files in the `constant` directory. | `ConstantDirectory`, `transportPropertiesFile`, `turbulencePropertiesFile` |
| `system` | Manages configuration files in the `system` directory. | `SystemDirectory`, `controlDictFile`, `fvSchemesFile`, `fvSolutionFile` |
| `boundaries` | Defines and applies boundary conditions. | `Boundary`, `boundaries_conditions_config` |
| `mesh` | Mesh tools, including integration with `classy_blocks` and `snappyHexMesh`. | `BlockMeshFile`, `Meshing`, `gmsh_mesher` |
| `utilities` | Utility functions and classes not specific to OpenFOAM (units, fluid properties, etc.). | `Quantity`, `FluidMechanics`, `manageunits` |
| `postprocess` | Classes for result analysis, visualization (via `pyvista`), and data extraction. | `FoamPostProcessing`, `ResidualsPost` |

## 4. Internal Mechanisms for Developers

For effective contribution, it is essential to understand how `foampilot` translates Python objects into OpenFOAM files and manages complex configurations.

### 4.1. File Writing Mechanism (`OpenFOAMFile`)

The core of data serialization lies in the base class `OpenFOAMFile` (`foampilot/foampilot/src/foampilot/base/openFOAMFile.py`).

* **Inheritance and Attributes:** Each OpenFOAM configuration file (e.g., `transportPropertiesFile` or `controlDictFile`) inherits from `OpenFOAMFile`. Configuration parameters are stored in the instance attribute `self.attributes`.
* **Dynamic Access:** Overriding magic methods `__getattr__` and `__setattr__` allows direct access and modification of parameters as object attributes (e.g., `solver.constant.transportProperties.nu = ...`), even if they are stored in the `self.attributes` dictionary.
* **Serialization (`write_file`):** The `write_file` method recursively iterates over `self.attributes` and uses the internal `_format_value` method to convert Python data types (booleans, numbers, tuples, especially `Quantity`) into OpenFOAM-specific syntax (e.g., `true`/`false`, parenthesis-enclosed lists).

### 4.2. Unit and Dimension Management (`Quantity`)

The `Quantity` class (`foampilot/foampilot/src/foampilot/utilities/manageunits.py`) is a *wrapper* around the `pint` library and ensures physical consistency.

* **Purpose:** Stores a numerical value with its physical unit (e.g., `Quantity(10, "m/s")`).
* **Automatic Conversion:** When writing to an OpenFOAM file, `OpenFOAMFile._format_value` checks if the value is a `Quantity`. If so, it uses `get_in(target_unit)` to convert the value to the unit expected by OpenFOAM (defined in `OpenFOAMFile.DEFAULT_UNITS`), ensuring all written values are in OpenFOAM’s base units.
* **OpenFOAM Dimensions:** The `to_openfoam_dimensions()` method uses `pint` to derive OpenFOAM’s dimension vector (M, L, T, Θ, N, J, A) from the unit, which is crucial for generating field file headers (e.g., `U`, `p`).

### 4.3. Solver and Field Orchestration (`Solver` and `CaseFieldsManager`)

The `Solver` class delegates field management to the `CaseFieldsManager` class (`foampilot/foampilot/src/foampilot/base/cases_variables.py`).

* **Solver Selection:** `Solver` (`foampilot/foampilot/src/foampilot/solver/solver.py`) uses boolean properties (e.g., `self.compressible`, `self.with_gravity`, `self.is_vof`) to determine the simulation type. The internal `_update_solver()` method selects the appropriate OpenFOAM solver (e.g., `incompressibleFluid`, `compressibleVoF`) and updates the `BaseSolver` instance.
* **Field Management:** `CaseFieldsManager` uses these properties to dynamically generate the list of required physical fields (e.g., `U`, `p`, `k`, `epsilon`, `T`).
    * If `self.with_gravity` is true, the pressure field becomes `p_rgh`.
    * If a turbulence model is defined, associated fields (`k`, `epsilon`, `omega`, `nut`) are added.
    * This field list is then used by `Boundary` to initialize boundary conditions for *all* required fields.

### 4.4. Advanced Boundary Condition Management (`Boundary`)

The `Boundary` class (`foampilot/foampilot/src/foampilot/boundaries/boundaries_dict.py`) translates physical boundary conditions into OpenFOAM configurations.

* **Centralized Configuration:** Uses a configuration dictionary (`BOUNDARY_CONDITIONS_CONFIG`) mapping physical condition types (e.g., `"velocityInlet"`) to required OpenFOAM configurations for each field (U, p, k, etc.) based on the selected turbulence model.
* **Wildcard Application:** `apply_condition_with_wildcard(pattern, condition_type, **kwargs)` applies a condition to all patches matching a regular expression (`pattern`).
* **Condition Resolution:** For each field, `_resolve_field_config` determines the final OpenFOAM configuration. For example, a wall condition selects between `noSlip` or `slip` and applies appropriate wall functions (`wallFunction`) for turbulence fields using the `WALL_FUNCTIONS` dictionary.
* **File Generation:** `write_boundary_conditions()` iterates over all fields managed by `CaseFieldsManager` and uses `OpenFOAMFile.write_boundary_file` to generate boundary condition files in the `0/` directory.

## 5. Developer Workflow (Based on `muffler.py`)

This usage example illustrates a typical workflow for a developer using `foampilot`:

| Step | Description | Key Classes and Methods |
| :--- | :--- | :--- |
| **1. Initialization** | Set working directory and initialize solver. | `Solver(path)`, `FluidMechanics` |
| **2. Physical Configuration** | Determine fluid properties and apply them to configuration files. | `FluidMechanics.get_fluid_properties()`, `solver.constant.transportProperties.nu = ...` |
| **3. Mesh** | Define geometry and mesh (often via `classy_blocks` integration) and generate `blockMeshDict`. | `classy_blocks.Cylinder`, `cb.Mesh()`, `Meshing(path, mesher="blockMesh")` |
| **4. Boundary Conditions** | Initialize and apply boundary conditions to mesh-defined patches. | `solver.boundary.initialize_boundary()`, `solver.boundary.apply_condition_with_wildcard()` |
| **5. File Writing** | Generate all OpenFOAM configuration files on disk. | `solver.system.write()`, `solver.constant.write()`, `solver.boundary.write_boundary_conditions()` |
| **6. Execution** | Run the OpenFOAM simulation. | `solver.run_simulation()` |
| **7. Post-Processing** | Analyze results, generate visualizations and reports. | `FoamPostProcessing`, `ResidualsPost`, `latex_pdf.LatexDocument` |

This workflow highlights how the different `foampilot` modules interact to provide a complete abstraction of the OpenFOAM simulation process. To contribute effectively, a developer must understand how each module class interacts with the central `Solver` object and how Python commands are translated into OpenFOAM dictionary syntax.
