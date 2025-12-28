#!/usr/bin/env python

from pathlib import Path
from foampilot.solver import Solver
from foampilot import Meshing, commons, utilities, postprocess
from foampilot.utilities.fluids_theory import FluidMechanics
from foampilot.utilities.manageunits import Quantity
import classy_blocks as cb
import numpy as np
import json

# ------------------------------
# 1. DEFINE CASE PATH
# ------------------------------
current_path = Path.cwd() 

# ------------------------------
# 2. FLUID PROPERTIES (modern API)
# ------------------------------
available_fluids = FluidMechanics.get_available_fluids()
fluid = FluidMechanics(
    available_fluids["Air"],
    temperature=Quantity(293.15, "K"),
    pressure=Quantity(101325, "Pa")
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
block_mesh_json = Path.cwd() / "data" / "block_mesh.json"
with open(block_mesh_json) as f:
    mesh_dict = json.load(f)

# Create ClassyBlocks Mesh
mesh = cb.Mesh.from_dict(mesh_dict)

# Default patch
mesh.set_default_patch("walls", "wall")

# Write blockMeshDict
mesh.write(
    current_path / "system" / "blockMeshDict",
    current_path / "debug.vtk"
)

# Run blockMesh
mesher = Meshing(current_path, mesher="blockMesh")
mesher.mesher.run()

# ------------------------------
# 5. BOUNDARY CONDITIONS (new API)
# ------------------------------
solver.boundary.initialize_boundary()

# Velocity inlet
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

# Pressure outlet
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet"
)

# Walls
solver.boundary.apply_condition_with_wildcard(
    pattern="walls",
    condition_type="wall"
)

# Write all boundary files
solver.boundary.write_boundary_conditions()

# ------------------------------
# 6. FUNCTION OBJECTS (modern API)
# ------------------------------

# fieldAverage
solver.system.add_function_object(
    name="fieldAverage",
    function_type="fieldAverage",
    fields=["U", "p", "k"],
    operation="average",
    writeControl="timeStep",
    writeInterval=10
)

# runTimeControl
solver.system.add_function_object(
    name="runTimeControl",
    function_type="runTimeControl",
    conditions=[
        {
            "type": "average",
            "functionObject": "forceCoeffs",
            "fields": "(Cd)",
            "tolerance": 1e-3,
            "window": 20
        }
    ]
)

solver.system.write_functions_file()

# ------------------------------
# 7. TOPOSET / CREATEPATCH (modern)
# ------------------------------

commons.OpenFOAM.run_tool(
    "topoSet",
    case_path=current_path
)

commons.OpenFOAM.run_tool(
    "createPatch",
    case_path=current_path
)

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