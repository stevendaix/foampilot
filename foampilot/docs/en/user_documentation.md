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
    velocity=(ValueWithUnit(10, "m/s"), ValueWithUnit(0, "m/s"), ValueWithUnit(0, "m/s")),
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
from foampilot.utilities.manageunits import ValueWithUnit

solver.constant.transportProperties.nu = ValueWithUnit(1e-6, "m2/s")
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
from foampilot.utilities.manageunits import ValueWithUnit

fluid_mech = FluidMechanics(
    FluidMechanics.get_available_fluids()['Water'],
    temperature=ValueWithUnit(293.15, "K"),
    pressure=ValueWithUnit(101325, "Pa")
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

### 3.3. Time-Varying and Spatial Boundary Conditions from CSV Files

#### Overview

The `foampilot.boundaries.csv_boundary_condition` module provides a comprehensive API for applying boundary conditions that vary over time and/or are spatially distributed from CSV files or pandas DataFrames. This module relies on two OpenFOAM mechanisms:

1. **Function1 `table` (CSV format)** — for uniform values that vary over time (e.g., sinusoidal inlet temperature).
2. **`nonuniformList` values** — for spatially interpolated distributions on patch faces (e.g., 2D temperature profile imposed at the inlet).

Temperature/energy transport for incompressible flows is handled via a **`scalarTransport` functionObject** in `system/functions`. See section 5.4 for energy configuration details.

#### High-Level API

Two functions are exposed via `foampilot.boundaries`:

##### `set_csv_condition()` — Uniform time-varying conditions

Attaches a uniform but time-varying boundary condition to a patch using OpenFOAM's `Function1::table` with CSV format.

**Signature:**

```python
set_csv_condition(
    boundary,          # solver.boundary object
    patch_name,        # str: patch name (e.g., "inlet")
    field,             # str: field name (e.g., "T")
    data,             # str | Path | pandas.DataFrame
    time_column=0,     # str | int: column for time
    value_column=None, # str | int: column for scalar value
    value_columns=None,# list: columns for vector (3 items)
    header_lines=0,    # int: header lines to skip
    separator=",",     # str: CSV separator
    out_of_bounds="clamp",  # str: "clamp", "error", "warn", "zero", "repeat"
    interpolation_scheme="linear",  # str: "linear" or "spline"
    default_value=None,  # float | str: default value for "value" entry
    csv_filename=None,   # str: filename in constant/
)
```

**Example — Scalar temperature with sinusoidal time variation:**

```python
import pandas as pd
from foampilot.boundaries import set_csv_condition

df = pd.DataFrame({
    "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "T_K": [300, 350, 320, 380, 340, 360],
})

set_csv_condition(
    boundary=solver.boundary,
    patch_name="inlet",
    field="T",
    data=df,
    time_column="time_s",
    value_column="T_K",
    header_lines=0,
    separator=",",
    out_of_bounds="clamp",
    interpolation_scheme="linear",
    default_value=300,
)
```

The CSV is written to `constant/` without headers. The `Function1::table` reads columns by index.

**Generated `0/T` file:**

```cpp
boundaryField
{
    inlet
    {
        type            uniformFixedValue;
        uniformValue    table
        {
            type            csv;
            nHeaderLine     0;
            columns         (0 1);
            file            "constant/inlet_temperature.csv";
            separator       ",";
            mergeSeparators false;
            interpolationScheme linear;
        }
        value           uniform 300;
    }
}
```

**Example — Vector velocity with time variation:**

```python
df = pd.DataFrame({
    "time_s": [0.0, 1.0, 2.0],
    "Ux": [1.0, 2.0, 1.5],
    "Uy": [0.0, 0.5, 0.3],
    "Uz": [0.0, 0.0, 0.0],
})

set_csv_condition(
    boundary=solver.boundary,
    patch_name="inlet",
    field="U",
    data=df,
    time_column="time_s",
    value_columns=["Ux", "Uy", "Uz"],
)
```

The `columns` entry for vector fields is `(0 (1 2 3))`.

---

##### `set_spatial_csv_condition()` — Spatially interpolated distributions

Interpolates values from a point cloud CSV onto the OpenFOAM patch face centers, then writes `nonuniformList` values into time-directory field files.

**Signature:**

```python
set_spatial_csv_condition(
    boundary,              # solver.boundary object
    patch_name,            # str: patch name
    field,                 # str: field name (e.g., "T")
    data,                  # str | Path | pandas.DataFrame
    time_column=0,         # str | int: time column
    spatial_columns=None,  # list: [time, x, y, z, value] (point cloud)
    face_id_column=None,   # str | int: face ID column (long format)
    value_column=None,     # str | int: value column (long format)
    header_lines=0,        # int
    separator=",",         # str
    default_value=None,    # float | str
    interpolation_method="linear",  # "linear", "nearest", "cubic"
)
```

**Supported formats:**

1. **Point cloud** — Columns: `time, x, y, z, value`. Source points interpolated onto face centers via `scipy.interpolate.griddata`.
2. **Long format with face IDs** — Columns: `time, face_id, value`. Each row specifies a face and its value at a given time.
3. **Wide format** — One row per time, one column per spatial point.

**Example — Spatial temperature profile:**

```python
import pandas as pd
import math

rows = []
for t in [0.0, 0.5, 1.0]:
    for i in range(10):
        x = i * 0.2
        y = 0.5
        temp = 300 + 50 * math.sin(2 * math.pi * x / 2.0) + 20 * t
        rows.append({"time_s": t, "x": x, "y": y, "z": 0.05, "T_K": temp})

df = pd.DataFrame(rows)

set_spatial_csv_condition(
    boundary=solver.boundary,
    patch_name="inlet",
    field="T",
    data=df,
    time_column="time_s",
    spatial_columns=["x", "y", "z", "T_K"],
    header_lines=0,
    separator=",",
    default_value=300,
    interpolation_method="nearest",
)
```

**Generated files:**

- `0/T` — contains the non-uniform distribution at the initial time (copied from the `0/T` template).
- `<time>/T` — one file per CSV time step, with interpolated non-uniform face values.

---

#### Low-Level Helpers

##### `CsvTimeSeries`

Utility class for managing time-series CSV data:

```python
from foampilot.boundaries import CsvTimeSeries

ts = CsvTimeSeries(
    csv_file,          # Path | str | DataFrame
    time_column="time_s",
    value_column="T_K",
    header_lines=1,
    separator=",",
)
ts.get_initial_value()  # -> float: first value
ts.get_times()          # -> np.ndarray: time column
ts.get_values()         # -> np.ndarray: value column
ts.get_dataframe()      # -> pd.DataFrame
ts.write_csv_table(destination_path, header_lines=0, separator=",")
```

##### `write_csv_table()`

Writes a CSV in OpenFOAM-compatible format to `constant/<filename>` (no headers, no index):

```python
from foampilot.boundaries import write_csv_table

csv_path = write_csv_table(
    case_path=solver.case_path,
    csv_data=df_or_path,
    time_column=0,
    value_columns=[1, 2, 3],  # for vector
    header_lines=0,
    separator=",",
    filename="inlet_data.csv",
)
```

##### `make_uniform_fixed_value_bc()` / `make_uniform_fixed_value_vector_bc()`

Generate the OpenFOAM dictionary for `uniformFixedValue`:

```python
from foampilot.boundaries import make_uniform_fixed_value_bc

bc = make_uniform_fixed_value_bc(
    csv_path="constant/inlet_temperature.csv",
    time_column=0,
    value_column=1,
    header_lines=0,
    separator=",",
    out_of_bounds="clamp",
    interpolation_scheme="linear",
    default_value=300,
)
solver.boundary.set_raw_condition("inlet", "T", bc)
```

For vector fields:

```python
from foampilot.boundaries import make_uniform_fixed_value_vector_bc

bc = make_uniform_fixed_value_vector_bc(
    csv_path="constant/inlet_velocity.csv",
    time_column=0,
    value_columns=[1, 2, 3],
)
solver.boundary.set_raw_condition("inlet", "U", bc)
```

---

#### CSV File Format

The source CSV must contain a **time column** and **one or three value columns**:

| Format | Columns | Example |
| :--- | :--- | :--- |
| **Scalar** | `time, value` | `0.0, 350` |
| **Vector** | `time, vx, vy, vz` | `0.0, 1.0, 0.0, 0.0` |
| **Spatial (point cloud)** | `time, x, y, z, value` | `0.0, 0.5, 0.0, 0.05, 350` |

---

#### Energy Management (Incompressible Temperature)

For incompressible flows with temperature transport:

```python
solver = Solver(case_path)
solver.compressible = False
solver.with_gravity = False
solver.energy_activated = True
solver.turbulence_model = "laminar"

solver.constant.transportProperties.nu = ValueWithUnit(1.5e-5, "m^2/s")
solver.constant.transportProperties.Pr = 0.85
```

The solver remains `incompressibleFluid`. Temperature transport is handled by a `scalarTransport` functionObject in `system/functions`:

```cpp
#includeFunc scalarTransport(T, diffusivity=constant, D = 1.76471e-05)
```

Where `D = nu / Pr` (thermal diffusivity).

**Automatic fvSchemes entries:**

| Entry | Value | Description |
| :--- | :--- | :--- |
| `div(phi,T)` | `bounded Gauss linearUpwind grad(T)` | Convection of T |
| `laplacian(DT,T)` | `Gauss linear corrected` | Diffusion of T |

**Automatic fvSolution entries:**

| Entry | Description |
| :--- | :--- |
| `T` solver | `smoothSolver`, tolerance 1e-6, relTol 0.1 |
| `TFinal` | Inherits `$T` with `relTol 0` |
| `relaxationFactors.equations.T` | 0.7 |

**Generated `system/functions` file:**

```cpp
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      functions;
}

#includeFunc scalarTransport(T, diffusivity=constant, D = 1.76471e-05)
```

---

#### Limitations and Best Practices

1. **Critical call order**: `write_boundary_conditions()` must be called **before** `set_spatial_csv_condition()` so that the `0/<field>` template exists.
2. **Spatial CSV requires SciPy**: `pip install scipy`.
3. **CSV separator** is automatically quoted in the OpenFOAM file (e.g., `","`).
4. **Vector columns format**: OpenFOAM uses `columns (0 (1 2 3))` for vector fields.
5. **Incompressible energy**: Always use `incompressibleFluid` with `scalarTransport`. Do NOT use the `functions` solver module — it does not solve the flow.

---

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

### 6.1. Direct OpenFOAM Reading (without `foamToVTK`)

For faster post-processing workflows, `foampilot` provides direct readers that
parse OpenFOAM's native `polyMesh` and field files into PyVista objects,
bypassing the intermediate `foamToVTK` conversion step.

**Single-region cases:**

```python
from foampilot.postprocess import OpenFOAMDirectReader
import pyvista as pv

reader = OpenFOAMDirectReader("/chemin/vers/cas")
mesh = reader.to_pyvista(fields=["U", "p"], time_step="1")

print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} cells")
mesh.plot(scalars="U", cmap="viridis")
```

**Multi-region CHT cases:**

```python
from foampilot.postprocess import CHTDirectReader

reader = CHTDirectReader("/chemin/vers/cas_cht")
print("Regions:", reader.region_names)        # ["fluid", "solid"]
print("Types:", reader.regions)               # {"fluid": "fluid", "solid": "solid"}

# Load all regions with temperature field
mb = reader.get_all_meshes(fields=["T"], time_step="0.1")

# Visualize
pl = pv.Plotter(off_screen=True)
for name in mb.keys():
    pl.add_mesh(mb[name], scalars="T", cmap="coolwarm", opacity=0.8)
pl.screenshot("cht_temperature.png")
pl.clear()

# Interface temperatures
temps = reader.get_interface_temperatures("fluid_to_solid", time_step="0.1")
print(temps)  # {"fluid_T": ..., "solid_T": ..., "T_interface": ...}
```

**Key advantages:**

- No need to run `foamToVTK` (faster, no extra disk usage)
- Automatic detection of point vs cell fields based on `FoamFile` headers
- Lazy loading: mesh and fields are only read when accessed
- Built-in caching of fields across multiple requests
- Works with gzipped field files (`.gz`)

**Available classes and functions:**

| Name | Purpose |
| :--- | :--- |
| `OpenFOAMDirectReader` | Read a single-region case into `pv.UnstructuredGrid` |
| `CHTDirectReader` | Read a CHT multi-region case into `pv.MultiBlock` |
| `read_openfoam()` | Convenience function for single-region cases |
| `read_cht_openfoam()` | Convenience function for CHT cases |

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

