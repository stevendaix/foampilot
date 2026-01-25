#!/usr/bin/env python

from pathlib import Path
from foampilot.solver import Solver
from foampilot import Meshing, commons, utilities, postprocess,latex_pdf, FluidMechanics , ValueWithUnit
import classy_blocks as cb
import numpy as np
import json

# exemple base : https://develop.openfoam.com/Development/openfoam/-/tree/30d2e2d3cfd2c2f268dd987b413dbeffd63962eb/tutorials/incompressible/simpleFoam/simpleCar

# ------------------------------
# 1. DEFINE CASE PATH
# ------------------------------
current_path = Path.cwd() / "cases"

# ------------------------------
# 2. FLUID PROPERTIES (modern API)
# ------------------------------
available_fluids = FluidMechanics.get_available_fluids()
fluid = FluidMechanics(
    available_fluids["Air"],
    temperature=ValueWithUnit(293.15, "K"),
    pressure=ValueWithUnit(101325, "Pa")
)
nu = fluid.get_fluid_properties()["kinematic_viscosity"]

# ------------------------------
# 3. INITIALIZE SOLVER
# ------------------------------
solver = Solver(current_path)
solver.compressible = False
solver.with_gravity = False

# Set viscosity
solver.constant.transportProperties.nu = nu

# Generate OF folders
solver.system.write()
solver.constant.write()

# ------------------------------
# 4. MESH GENERATION (blockMesh JSON)
# ------------------------------
# Load blockMesh configuration from a JSON file
data_path = Path.cwd() / "block_mesh.json"
mesh = Meshing(current_path,mesher="blockMesh")

mesh.mesher.load_from_json(data_path)

# Write the blockMeshDict file and run the meshing process
mesh.mesher.write(file_path = current_path / "system" / "blockMeshDict")
mesh.mesher.run()

# ------------------------------
# 6. FUNCTION OBJECTS 
# ------------------------------

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



# ------------------------------
# 5. BOUNDARY CONDITIONS (new API)
# ------------------------------
solver.boundary.initialize_boundary()

# Velocity inlet
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(
        ValueWithUnit(10, "m/s"),
        ValueWithUnit(0, "m/s"),
        ValueWithUnit(0, "m/s")
    ),
    turbulence_intensity=0.05
)

# Pressure outlet
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet"
)


# Velocity inlet
solver.boundary.apply_condition_with_wildcard(
    pattern="airIntake",
    condition_type="velocityInlet",
    velocity=(
        ValueWithUnit(1.2, "m/s"),
        ValueWithUnit(0, "m/s"),
        ValueWithUnit(0, "m/s")
    ),
    turbulence_intensity=0.05
)

# Walls
solver.boundary.apply_condition_with_wildcard(
    pattern="walls",
    condition_type="wall"
)

# Write all boundary files
solver.boundary.write_boundary_conditions()



# ------------------------------
# 8. RUN SIMULATION
# ------------------------------
solver.run_simulation()

# ------------------------------
# 9. POST-PROCESS
# ------------------------------
residuals_post = utilities.ResidualsPost(current_path / "log.incompressibleFluid")
residuals_post.process(export_csv=True, export_json=True, export_png=True, export_html=True)

print("Simulation and post-processing completed.")