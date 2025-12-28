# OpenFOAM Python Example with Foampilot – Detailed Explanation

This document explains step by step the Python example that sets up and runs an OpenFOAM simulation using the **Foampilot** library. It covers the mesh generation from JSON, solver setup, function objects, boundary conditions, and post-processing.


[SimpleCar Tutorial (OpenFOAM)](https://develop.openfoam.com/Development/openfoam/-/tree/30d2e2d3cfd2c2f268dd987b413dbeffd63962eb/tutorials/incompressible/simpleFoam/simpleCar)

---

## 1. Define Case Path

```python
current_path = Path.cwd() / "cases"
```

We define the directory where the simulation case will be stored. All OpenFOAM folders like `system`, `constant`, and `0` will be generated under this path.

---

## 2. Fluid Properties (Modern API)

```python
available_fluids = FluidMechanics.get_available_fluids()
fluid = FluidMechanics(
    available_fluids["Air"],
    temperature=Quantity(293.15, "K"),
    pressure=Quantity(101325, "Pa")
)
nu = fluid.get_fluid_properties()["kinematic_viscosity"]
```

- `FluidMechanics.get_available_fluids()` returns a dictionary of predefined fluids.
- `FluidMechanics` allows setting temperature and pressure.
- `get_fluid_properties()` returns properties like `kinematic_viscosity` (`nu`) for later use in the solver.

---

## 3. Initialize Solver

```python
solver = Solver(current_path)
solver.compressible = False
solver.with_gravity = False
solver.constant.transportProperties.nu = nu
solver.system.write()
solver.constant.write()
```

- Initializes the solver object with the case path.
- Sets **incompressible** flow and **no gravity**.
- Sets the viscosity in `transportProperties`.
- Writes default OpenFOAM folders: `system` and `constant`.

---

## 4. Mesh Generation via JSON

```python
data_path = Path.cwd() / "block_mesh.json"
mesh = Meshing(current_path, mesher="blockMesh")
mesh.mesher.load_from_json(data_path)
mesh.mesher.write(file_path=current_path / "system" / "blockMeshDict")
mesh.mesher.run()
```

- **JSON-based mesh**: the mesh configuration (blocks, vertices, boundaries) is stored in `block_mesh.json`.
- `load_from_json()` reads the JSON and converts it into an internal Python structure for `blockMesh`.
- `write()` generates the OpenFOAM `blockMeshDict` file.
- `run()` executes `blockMesh` to create the mesh.

This approach allows **dynamic modification** of meshes without manually editing OpenFOAM dictionaries.

---

## 5. Function Objects

### 5.1 Field Average

```python
name_field, field_average_dict = utilities.Functions.field_average("fieldAverage")
utilities.Functions.write_function_field_average(name_field, field_average_dict, base_path=current_path, folder='system')
```

- Creates a **fieldAverage** function object in `system/controlDict`.
- `field_average_dict` contains configuration: which field to average and over which patches.
- `write_function_field_average()` writes it to OpenFOAM format.

### 5.2 Reference Pressure

```python
name_field_ref, reference_dict = utilities.Functions.reference_pressure("referencePressure")
utilities.Functions.write_function_reference_pressure(name_field_ref, reference_dict, base_path=current_path, folder='system')
```

- Sets a **reference pressure** to avoid singularities in incompressible solvers.
- Automatically writes the function object in `controlDict`.

### 5.3 Run-Time Control

```python
conditions1 = {
    "condition1": {
        "type": "average",
        "functionObject": "forceCoeffs1",
        "fields": "(Cd)",
        "tolerance": "1e-3",
        "window": "20",
        "windowType": "exact"
    }
}
name_field_rt1, rt1_dict = utilities.Functions.run_time_control("runTimeControl", conditions=conditions1)
utilities.Functions.write_function_run_time_control(
    name_field=name_field_rt1,
    name_condition="runTimeControl1",
    function_dict=rt1_dict,
    base_path=current_path,
    folder='system'
)
solver.system.write_functions_file(includes=["fieldAverage", "runTimeControl"])
```

- Allows **stopping criteria** based on conditions like maximum duration or convergence of averages.
- `run_time_control` returns a dictionary describing the control logic.
- `write_function_run_time_control()` saves it to `system/controlDict`.
- `write_functions_file()` updates the functions list.

---

## 6. Patches & topoSetDict

### 6.1 createPatchDict

```python
patch_names = ["airIntake"]
patches_dict = utilities.dictonnary.dict_tools.create_patches_dict(patch_names)
create_patch_dict_file = utilities.dictonnary.OpenFOAMDictAddFile(object_name='createPatchDict', **patches_dict)
create_patch_dict_file.write("createPatchDict", current_path)
```

- `create_patches_dict()` defines new patches dynamically.
- `OpenFOAMDictAddFile` writes the OpenFOAM dictionary to disk.

### 6.2 topoSetDict

```python
actions = [
    utilities.dictonnary.dict_tools.create_action(...),
    ...
]
actions_dict = utilities.dictonnary.dict_tools.create_actions_dict(actions)
create_topo_set_dict_file = utilities.dictonnary.OpenFOAMDictAddFile(object_name='topoSetDict', **actions_dict)
create_topo_set_dict_file.write("topoSetDict", current_path)
solver.system.run_topoSet()
solver.system.run_createPatch()
```

- `topoSetDict` is used to define cell/face/zone sets programmatically.
- `create_action()` defines operations like creating a `cellSet` or `faceSet`.
- `run_topoSet()` executes the `topoSet` utility.
- `run_createPatch()` creates the defined patches in OpenFOAM.

---

## 7. Boundary Conditions

```python
solver.boundary.initialize_boundary()
solver.boundary.apply_condition_with_wildcard(pattern="inlet", condition_type="velocityInlet", velocity=(Quantity(10, "m/s"), Quantity(0, "m/s"), Quantity(0, "m/s")), turbulence_intensity=0.05)
solver.boundary.apply_condition_with_wildcard(pattern="airIntake", condition_type="velocityInlet", velocity=(Quantity(1.2, "m/s"), Quantity(0, "m/s"), Quantity(0, "m/s")), turbulence_intensity=0.05)
solver.boundary.apply_condition_with_wildcard(pattern="outlet", condition_type="pressureOutlet")
solver.boundary.apply_condition_with_wildcard(pattern="walls", condition_type="wall")
solver.boundary.write_boundary_conditions()
```

- `initialize_boundary()` prepares the 0 folder structure.
- `apply_condition_with_wildcard()` applies conditions to all patches matching a pattern.
- `write_boundary_conditions()` writes all OpenFOAM boundary files.

---

## 8. Run Simulation

```python
solver.run_simulation()
```

- Executes the OpenFOAM solver (`simpleFoam`) with all configured settings.

---

## 9. Post-Processing

```python
residuals_post = utilities.ResidualsPost(current_path / "log.incompressibleFluid")
residuals_post.process(export_csv=True, export_json=True, export_png=True, export_html=True)
```

- Extracts residuals from the log file.
- Exports results in multiple formats: CSV, JSON, PNG, and HTML.

---

### ✅ Summary

This example demonstrates:

1. **Dynamic mesh generation** from JSON.
2. **Automatic function objects** for averages, references, and run-time control.
3. **Patch and cell/face set management** via `createPatchDict` and `topoSetDict`.
4. **Flexible boundary condition setup** using wildcards.
5. **Seamless simulation and post-processing** using Python wrappers.

