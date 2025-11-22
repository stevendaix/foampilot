#!/usr/bin/env python

# Import required libraries
from foampilot import incompressibleFluid, Meshing, commons, utilities, postprocess
from foampilot.utilities.fluids_theory import FluidMechanics
import pyvista as pv
from pathlib import Path
from foampilot.utilities.manageunits import Quantity
import numpy as np
import classy_blocks as cb

# Define the working directory for the simulation case
current_path = Path.cwd() / 'exemple2'

# List available fluids
print("Available fluids:")
available_fluids = FluidMechanics.get_available_fluids()
for name in available_fluids:
    print(f"- {name}")

# Create a FluidMechanics instance for water at room temperature and atmospheric pressure
fluid_mech = FluidMechanics(
    available_fluids['Water'],
    temperature=Quantity(293.15, "K"),
    pressure=Quantity(101325, "Pa")
)

# Get fluid properties including kinematic viscosity
properties = fluid_mech.get_fluid_properties()
kinematic_viscosity = properties['kinematic_viscosity']
print(f"\nUsing fluid: Water")
print(f"Kinematic viscosity: {kinematic_viscosity.get_in('m^2/s')} m²/s")

# Initialize the solver for incompressible fluid simulation
solver = incompressibleFluid(path_case=current_path)

# Set the kinematic viscosity in the solver's constant directory

solver.constant.transportProperties.nu=kinematic_viscosity

system_dir = solver.system.write()
system_dir = solver.constant.write()
