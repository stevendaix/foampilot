# Muffler CFD Example – FoamPilot

## Overview

This example demonstrates a **complete CFD workflow** using **FoamPilot** and **OpenFOAM** to simulate an incompressible flow through a muffler geometry. It is intended as a **reference example** showcasing FoamPilot’s philosophy:

- Explicit physical modeling (fluids, units)
- Parametric geometry and structured meshing
- Robust boundary-condition management
- Automated simulation execution
- Advanced post-processing and visualization
- Automatic PDF report generation

📁 **Location**: `examples/muffler`

---

## 1. Prerequisites

Before running this example, ensure that:

- OpenFOAM is correctly installed and accessible in your environment
- FoamPilot is installed
- The following Python dependencies are available:
  - `classy_blocks`
  - `pyvista`
  - `numpy`
  - `pandas`

---

## 2. Case Initialization

We start by defining the working directory and initializing the FoamPilot solver.

```python
from foampilot.solver import Solver
from pathlib import Path

current_path = Path.cwd() / "cas_test"
solver = Solver(current_path)

solver.compressible = False
solver.with_gravity = False
```

The `Solver` class is the **central orchestration object** in FoamPilot. It coordinates:

- OpenFOAM dictionaries
- Boundary conditions
- Simulation execution

---

## 3. Fluid Properties

FoamPilot relies on explicit fluid modeling through the `FluidMechanics` API.

```python
from foampilot import FluidMechanics, Quantity

available_fluids = FluidMechanics.get_available_fluids()

fluid = FluidMechanics(
    available_fluids["Water"],
    temperature=Quantity(293.15, "K"),
    pressure=Quantity(101325, "Pa")
)

properties = fluid.get_fluid_properties()
nu = properties["kinematic_viscosity"]
```

The kinematic viscosity is then injected directly into the OpenFOAM configuration:

```python
solver.constant.transportProperties.nu = nu
```

This ensures **unit consistency** and avoids hard-coded numerical values.

---

## 4. Writing OpenFOAM Configuration Files

```python
solver.system.write()
solver.constant.write()
solver.system.fvSchemes.to_dict()
```

FoamPilot automatically generates:

- `controlDict`
- `fvSchemes`
- `fvSolution`
- `transportProperties`

---

## 5. Geometry Definition (ClassyBlocks)

### 5.1 Geometric Parameters

```python
pipe_radius = 0.05
muffler_radius = 0.08
ref_length = 0.1
cell_size = 0.015
```

### 5.2 Geometry Construction

The geometry is built as a sequence of **parametric shapes**:

1. Inlet pipe (cylinder)
2. Expansion ring (muffler body)
3. Filled section
4. 90° elbow outlet

Example for the inlet cylinder:

```python
import classy_blocks as cb

shapes = []

shapes.append(cb.Cylinder(
    [0, 0, 0],
    [3 * ref_length, 0, 0],
    [0, pipe_radius, 0]
))

shapes[-1].chop_axial(start_size=cell_size)
shapes[-1].chop_radial(start_size=cell_size)
shapes[-1].chop_tangential(start_size=cell_size)
shapes[-1].set_start_patch("inlet")
```

Patches are defined **at geometry level**, ensuring consistency between meshing and boundary conditions.

---

## 6. Mesh Generation

```python
mesh = cb.Mesh()
for shape in shapes:
    mesh.add(shape)

mesh.set_default_patch("walls", "wall")
mesh.write(
    current_path / "system" / "blockMeshDict",
    current_path / "debug.vtk"
)
```

The mesh is then generated using OpenFOAM:

```python
from foampilot import Meshing

meshing = Meshing(current_path, mesher="blockMesh")
meshing.mesher.run()
```

---

## 7. Boundary Conditions

FoamPilot provides a **generic wildcard-based boundary API**.

```python
solver.boundary.initialize_boundary()
```

### 7.1 Velocity Inlet

```python
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(
        Quantity(10, "m/s"),
        Quantity(0, "m/s"),
        Quantity(0, "m/s")
    ),
    turbulence_intensity=0.05
)
```

### 7.2 Pressure Outlet

```python
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet"
)
```

### 7.3 Walls

```python
solver.boundary.apply_condition_with_wildcard(
    pattern="walls",
    condition_type="wall"
)
```

### 7.4 Writing Boundary Files

```python
solver.boundary.write_boundary_conditions()
```

---

## 8. Running the Simulation

```python
solver.run_simulation()
```

FoamPilot automatically manages:

- solver selection
- execution
- logging

---

## 9. Post-processing

### 9.1 Residuals

```python
from foampilot.utilities import ResidualsPost

residuals = ResidualsPost(current_path / "log.incompressibleFluid")
residuals.process(
    export_csv=True,
    export_json=True,
    export_png=True,
    export_html=True
)
```

### 9.2 Conversion to VTK

```python
from foampilot import postprocess

foam_post = postprocess.FoamPostProcessing(case_path=current_path)
foam_post.foamToVTK()
```

---

## 10. Visualization and Analysis

FoamPilot integrates tightly with **PyVista**.

Available analyses include:

- Scalar and vector visualization
- Q-criterion
- Vorticity
- Mesh statistics
- Region-based statistics
- CSV / JSON export
- Animations

```python
foam_post.calculate_q_criterion(mesh=cell_mesh, velocity_field="U")
foam_post.calculate_vorticity(mesh=cell_mesh, velocity_field="U")n
foam_post.create_animation(
    scalars="U",
    filename=current_path / "animation.gif",
    fps=5
)
```

---

## 11. Automatic PDF Report Generation

FoamPilot includes a LaTeX-based reporting engine.

```python
from foampilot import latex_pdf

doc = latex_pdf.LatexDocument(
    title="Simulation Report: Muffler Flow Case",
    author="Automated Report",
    output_dir=current_path
)

doc.add_title()
doc.add_toc()
doc.add_abstract(
    "This report summarizes the incompressible CFD simulation of a muffler."
)

doc.generate_document(output_format="pdf")
```

The generated report includes:

- Fluid properties
- Mesh statistics
- Field statistics
- Figures
- Data appendices

---

## 12. Summary

This example demonstrates a **full industrial-grade CFD workflow**:

- Parametric geometry and meshing
- Physically consistent fluid modeling
- Robust boundary-condition management
- Advanced post-processing
- Automated reporting

It is intended as a **template** for more complex CFD studies using FoamPilot.


