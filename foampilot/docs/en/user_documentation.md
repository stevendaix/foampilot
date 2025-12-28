# User Documentation for `foampilot`

## 1. Overall Working Philosophy of `foampilot`

The `foampilot` module is designed as a Python wrapper for **OpenFOAM**, aimed at simplifying and automating the computational fluid dynamics (CFD) simulation process. It abstracts the complexity of OpenFOAM’s file structure and commands, allowing the user to define, run, and post-process a simulation entirely in Python.

The philosophy of `foampilot` is based on the following principles:

1.  **Case Definition in Python:** Instead of manually editing configuration files (dictionaries) in the OpenFOAM directory structure, the user interacts with Python objects (`Solver`, `Meshing`, `Boundary`, `Constant`, `System`).
2.  **Automatic File Generation:** Python objects are responsible for automatically generating OpenFOAM configuration files (`controlDict`, `fvSchemes`, `transportProperties`, etc.) in the case directory.
3.  **Integration with the Python Ecosystem:** The module integrates with powerful Python libraries for specific tasks:
    *   **`classy_blocks`** for structured mesh generation (`blockMesh`).
    *   **`pyfluid`** (implicit in examples) for managing fluid properties and physical constants.
    *   **`pyvista`** for advanced post-processing and visualization.
    *   **`latex_pdf`** for generating structured simulation reports.

In short, `foampilot` transforms a **manual and fragmented workflow** (editing files, running shell commands) into a **scripted and reproducible workflow** (a single Python script).

## 2. Geometry and Mesh Selection

Choosing the meshing method is crucial in CFD and depends on the geometry complexity. `foampilot` supports three main scenarios:

| Meshing Method | Target Geometry | `foampilot` Tool / Library | Description & Advantages |
| :--- | :--- | :--- | :--- |
| **`blockMesh`** | Simple, extruded, or hexahedral block geometries. | `Meshing(..., mesher="blockMesh")` (via `classy_blocks`) | Ideal for simple geometries (channels, cylinders, etc.) or regular computational domains. Provides **full control** over mesh quality and cell distribution. The `run_example2.py` demonstrates creating complex geometries by combining blocks (`Cylinder`, `ExtrudedRing`, `Elbow`). |
| **`gmsh`** | Complex CAD geometries in **STEP** or IGES format. | `Meshing(..., mesher="gmsh")` | Enables meshing of complex CAD geometries with unstructured meshes (tetrahedra, prisms). Requires a geometry file (e.g., `.step`). |
| **`snappyHexMesh`** | Complex geometries in **STL** format (triangulated surface). | `Meshing(..., mesher="snappy")` | Standard for highly complex geometries (vehicles, buildings). Generates hexahedral mesh conforming to the STL surface with automatic boundary layer refinement. |

### 2.1. Structured Meshing with `blockMesh` (via `classy_blocks`)

For geometries that can be decomposed into hexahedral blocks (including extrusions), `foampilot` uses the `classy_blocks` library.

**Workflow:**
1.  Define basic geometric shapes (`cb.Cylinder`, `cb.ExtrudedRing`, `cb.Elbow`).
2.  Use chaining methods (`.chain()`, `.expand()`, `.fill()`) to build complex geometry.
3.  Set mesh on each shape using `.chop_axial()`, `.chop_radial()`, `.chop_tangential()`.
4.  Assign **patches** (surfaces) with `.set_start_patch()`, `.set_end_patch()`.
5.  Assemble everything in a `cb.Mesh()` object and write `blockMeshDict`:

```python
# Example usage
mesh = cb.Mesh()
# ... add shapes ...
mesh.set_default_patch("walls", "wall")
mesh.write(current_path / "system" / "blockMeshDict", current_path /"debug.vtk")
```

### 2.2. Unstructured Meshing with `gmsh` (for STEP)

For CAD geometries in STEP format:

1.  Ensure the STEP file is available (e.g., `geometry.step`).
2.  Initialize a `Meshing` object with `mesher="gmsh"`.
3.  Run the meshing process with the STEP file path:

```python
mesh_obj = Meshing(current_path, mesher="gmsh")
mesh_obj.mesher.run(current_path / "geometry.step")
```

### 2.3. Surface Meshing with `snappyHexMesh` (for STL)

For complex STL geometries:

1.  Create a simple `blockMeshDict` (via `classy_blocks` or manually) for the encompassing domain.
2.  Place the STL file in `constant/triSurface`.
3.  Initialize `Meshing` with `mesher="snappyHexMesh"`.
4.  Run the meshing. `foampilot` manages `snappyHexMesh` configuration and execution:

```python
mesh_obj = Meshing(current_path, mesher="snappyHexMesh")
mesh_obj.mesher.run()
```

*Note:* Detailed `snappyHexMeshDict` configuration (refinement levels, boundary layers) must be handled by the user or via advanced `foampilot` functions if available.

## 3. Solver Selection and Physics

Solver selection determines how `foampilot` handles simulation physics. The `Solver` class configures the case, and the appropriate OpenFOAM solver is selected and executed in the background.

### 3.1. Solver Selection

Implicit solver selection is done by configuring the `Solver` object:

```python
from foampilot.solver import Solver

solver = Solver(current_path)
solver.compressible = False   # Incompressible simulation
solver.with_gravity = False   # No gravity
# ... other properties: turbulence, multiphase, etc.
```

Based on these properties, `foampilot` configures `controlDict` and other dictionaries to use the most appropriate OpenFOAM solver (e.g., `simpleFoam` or `pimpleFoam` for incompressible, `rhoSimpleFoam` for compressible).

| Physics | `Solver` Property | Typical OpenFOAM Solver |
| :--- | :--- | :--- |
| **Incompressible** | `solver.compressible = False` | `incompressibleFluid` (internal `foampilot` solver) |
| **Compressible** | `solver.compressible = True` | `compressibleFluid` (internal `foampilot` solver) |
| **Transient** | `solver.transient = True` | (handles transient settings) |
| **Turbulence** | `solver.turbulence_model = "kEpsilon"` | (configures turbulence models) |
| **Multiphase (VOF)** | `solver.is_vof = True` | `incompressibleVoF` or `compressibleVoF` (internal `foampilot` solvers) |
| **Solid (Displacement)** | `solver.is_solid = True` | `solidDisplacement` (internal `foampilot` solver) |
| **Energy (Thermal)** | `solver.energy_activated = True` | (enables thermal fields) |

### 3.2. Boundary Conditions

Boundary conditions (BCs) are managed via `solver.boundary`, applied to **patches** created during meshing.

```python
solver.boundary.initialize_boundary()

# Inlet velocity
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(Quantity(10, "m/s"), Quantity(0, "m/s"), Quantity(0, "m/s")),
    turbulence_intensity=0.05
)

# Outlet pressure
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet"
)

# Wall
solver.boundary.apply_condition_with_wildcard(
    pattern="walls",
    condition_type="wall"
)

solver.boundary.write_boundary_conditions()
```

| `condition_type` | Description | Typical Fields |
| :--- | :--- | :--- |
| `fixedValue` | Imposed fixed value (e.g., temperature). | `T`, `C` |
| `zeroGradient` | Zero normal gradient (Neumann). | `p`, `T`, `U` |
| `velocityInlet` | Inlet velocity with turbulence parameters. | `U`, `k`, `epsilon` |
| `pressureOutlet` | Fixed or zero-gradient pressure. | `p` |
| `wall` | Solid wall (no-slip, zero heat flux by default). | `U`, `T` |
| `symmetryPlane` | Symmetry plane. | `U`, `p`, `T` |

### 3.3. Modifying Dictionaries or Adding a Patch

OpenFOAM dictionaries are exposed as Python objects:

```python
from foampilot.utilities.manageunits import Quantity

solver.constant.transportProperties.nu = Quantity(1e-6, "m2/s")
solver.system.controlDict.writeInterval = 100
solver.system.controlDict.endTime = 1000
```

*System files managed by `foampilot` include: `controlDict`, `fvSchemes`, `fvSolution`, `decomposeParDict`, plus custom dictionaries.*

```python
solver.constant.write()
solver.system.write()
```

Adding a patch:

*   **With `blockMesh`**:

```python
shapes[-1].set_end_patch("newPatch")
```

*   **With `gmsh` or `snappyHexMesh`**: defined in mesh configuration files. Apply BCs afterward.

## 4. `system` and `constant` Setup with `pyfluid`

`pyfluid` (or `FluidMechanics` from `foampilot.utilities.fluids_theory`) defines physical fluid properties and constants.

```python
from foampilot.utilities.fluids_theory import FluidMechanics
from foampilot.utilities.manageunits import Quantity

fluid_mech = FluidMechanics(
    FluidMechanics.get_available_fluids()['Water'],
    temperature=Quantity(293.15, "K"),
    pressure=Quantity(101325, "Pa")
)

properties = fluid_mech.get_fluid_properties()
solver.constant.transportProperties.nu = properties['kinematic_viscosity']

solver.system.write()
solver.constant.write()
```

`constant` files managed: `transportProperties`, `physicalProperties`, `turbulenceProperties`, `g`, `pRef`, `radiationProperties`, `fvModels`.

## 5. Running the Solver

```python
solver.run_simulation()
```

Parallel execution:

```python
solver.decompose_domain(cores=4)
solver.run_simulation(parallel=True)
solver.reconstruct_domain()
```

## 6. Post-Processing with `pyvista`

```python
from foampilot import postprocess
import pyvista as pv

foam_post = postprocess.FoamPostProcessing(case_path=current_path)
foam_post.foamToVTK()
latest_time_step = foam_post.get_all_time_steps()[-1]
structure = foam_post.load_time_step(latest_time_step)
cell_mesh = structure["cell"]

pl_contour = pv.Plotter(off_screen=True)
pl_contour.add_mesh(cell_mesh, scalars='p', show_scalar_bar=True)
foam_post.export_plot(pl_contour, current_path / "contour_plot.png")
```

Capabilities include slices, contours, vector plots, vortex analysis, mesh statistics, and exporting data.

## 7. LaTeX Reporting with `latex_pdf`

`latex_pdf` generates structured PDF reports from Python:

```python
doc = latex_pdf.LatexDocument(
    title="Simulation Report: Muffler Flow Case",
    author="Automated Report",
    filename="simulation_report",
    output_dir=current_path
)

doc.add_table(mesh_table_data, headers=["Statistic", "Value"], caption="Mesh Quality Statistics")

for img_name in ["slice_plot.png", "contour_plot.png"]:
    img_path = current_path / img_name
    if img_path.exists():
        doc.add_figure(str(img_path), caption=img_name.replace("_", " ").title(), width="0.7\\textwidth")

doc.generate_document(output_format="pdf")
```

This ensures full **traceability** and **reproducibility** of simulation results.

