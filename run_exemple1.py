from foampilot import incompressibleFluid, Meshing, commons, utilities
from pathlib import Path
from foampilot.utilities.manageunits import Quantity

# Define the base case directory
current_path = Path.cwd() / 'exemple1'

# --- 1. Initialize the solver object ---
# This creates an OpenFOAM-like solver environment for an incompressible fluid simulation
solver = incompressibleFluid(path_case=current_path)

# Generate and write the system and constant directories
# These contain the basic OpenFOAM configuration files
system_dir = solver.system.write()
system_dir = solver.constant.write()

# Convert fvSchemes into a Python dictionary for inspection
solver.system.fvSchemes.to_dict()

# --- 2. Meshing setup ---
# Load blockMesh configuration from a JSON file
data_path = Path.cwd() / 'data' / "block_mesh.json"
meshing = Meshing(path_case=current_path)
meshing.load_from_json(data_path)

# Write the blockMeshDict file and run the meshing process
meshing.write()
meshing.run_blockMesh()

# --- 3. Boundary file management ---
# Handle boundary conditions stored inside the "0/" directory
test = commons.BoundaryFileHandler(current_path)
test.data

# --- 4. Adding functionObjects (fieldAverage, referencePressure, runTimeControl) ---
# Example: add a fieldAverage function to system/controlDict
name_field, field_average_dict = utilities.Functions.field_average("fieldAverage")
utilities.Functions.write_function_field_average(name_field, field_average_dict, base_path=current_path, folder='system')

# Example: add a reference pressure function
name_field_ref, reference_dict = utilities.Functions.reference_pressure("referencePressure")
utilities.Functions.write_function_reference_pressure(name_field_ref, reference_dict, base_path=current_path, folder='system')

# Example: runTimeControl conditions (stopping criteria)
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

conditions2 = {
    "condition1": {
        "type": "maxDuration",
        "duration": "100"
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

# Create the functions file in system
solver.system.write_functions_file(includes=["fieldAverage", "runTimeControl"])

# --- 5. Dictionary manipulation (patches & topoSet) ---
# Define patch names for createPatchDict
patch_names = ["airIntake"]
patches_dict = utilities.dictonnary.dict_tools.create_patches_dict(patch_names)

# Write createPatchDict file
create_patch_dict_file = utilities.dictonnary.OpenFOAMDictAddFile(object_name='createPatchDict', **patches_dict)
create_patch_dict_file.write("createPatchDict", current_path)

# Define topoSet actions (to create cellSets, faceSets, etc.)
actions = [
    utilities.dictonnary.dict_tools.create_action(
        name="porousCells",
        action_type="cellSet",
        action="new",
        source="boxToCell",
        box=[(2.05, 0.4, -1), (2.1, 0.85, 1)]
    ),
    utilities.dictonnary.dict_tools.create_action(
        name="porousZone",
        action_type="cellZoneSet",
        action="new",
        source="setToCellZone",
        set="porousCells"
    ),
    utilities.dictonnary.dict_tools.create_action(
        name="airIntake",
        action_type="faceSet",
        action="new",
        source="patchToFace",
        patch="body"
    ),
    utilities.dictonnary.dict_tools.create_action(
        name="airIntake",
        action_type="faceSet",
        action="subset",
        source="boxToFace",
        box=[(2.6, 0.75, 0), (2.64, 0.8, 0.1)]
    )
]

# Create topoSetDict and write it
actions_dict = utilities.dictonnary.dict_tools.create_actions_dict(actions)
create_topo_set_dict_file = utilities.dictonnary.OpenFOAMDictAddFile(object_name='topoSetDict', **actions_dict)
create_topo_set_dict_file.write("topoSetDict", current_path)

# Run topoSet and createPatch commands
solver.system.run_topoSet()
solver.system.run_createPatch()

# --- 6. Boundary condition setup ---
# Initialize boundary handler
solver.boundary.initialize_boundary()

# Example: set uniform inlet boundary for "airIntake"
solver.boundary.set_uniform_normal_fixed_value_all_fields("airIntake", mode="intakeType1", ref_value=1.2)

# Velocity inlet condition
solver.boundary.set_velocity_inlet(
    pattern="inlet",
    velocity=(Quantity(10, "m/s"), Quantity(0, "m/s"), Quantity(0, "m/s")),
    turbulence_intensity=0.05  # 5% turbulence intensity
)

# Pressure outlet condition
solver.boundary.set_pressure_outlet(
    pattern="outlet",
    velocity=(Quantity(10, "m/s"), Quantity(0, "m/s"), Quantity(0, "m/s")),
)

# Wall boundary conditions (no-slip)
solver.boundary.set_wall(pattern="lowerWall", velocity=(Quantity(0, "m/s"), Quantity(0, "m/s"), Quantity(0, "m/s")))
solver.boundary.set_wall(pattern="body", velocity=(Quantity(0, "m/s"), Quantity(0, "m/s"), Quantity(0, "m/s")))
solver.boundary.set_wall(pattern="upperWall", velocity=(Quantity(0, "m/s"), Quantity(0, "m/s"), Quantity(0, "m/s")))

# Write boundary condition files for all fields
fields = ["U", "p", "k", "epsilon", "nut"]
for field in fields:
    solver.boundary.write_boundary_file(field)

print("Boundary condition files have been generated.")

# --- 7. Run the simulation ---
solver.run_simulation()

# --- 8. Post-processing: residuals ---
# Parse residuals from the solver log and export in multiple formats
residuals_post = utilities.ResidualsPost(current_path / "log.incompressibleFluid")
residuals_post.process(export_csv=True, export_json=True, export_png=True, export_html=True)
