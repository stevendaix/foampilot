from build123d import *
from build123d import exporters3d

import gmsh
import random
#!/usr/bin/env python
from pathlib import Path

# Import required libraries
from foampilot import Meshing, commons, postprocess,latex_pdf,ValueWithUnit,FluidMechanics,Solver

current_path = Path.cwd() / "exemple_gmsh"



# List available fluids
print("Available fluids:")
available_fluids = FluidMechanics.get_available_fluids()
for name in available_fluids:
    print(f"- {name}")

# Create a FluidMechanics instance for water at room temperature and atmospheric pressure
fluid_mech = FluidMechanics(
    available_fluids['Air'],
    temperature=ValueWithUnit(293.15, "K"),
    pressure=ValueWithUnit(101325, "Pa")
)

# Get fluid properties including kinematic viscosity
properties = fluid_mech.get_fluid_properties()
kinematic_viscosity = properties['kinematic_viscosity']
print(f"\nUsing fluid: Water")
print(f"Kinematic viscosity: {kinematic_viscosity}")

# Initialize the solver 
solver = Solver(current_path)
solver.compressible = False
solver.with_gravity = False

# Set the kinematic viscosity in the solver's constant directory

solver.constant.transportProperties.nu=kinematic_viscosity

# Generate system and constant directories with updated OpenFOAM configuration
system_dir = solver.system.write()
system_dir = solver.constant.write()



# -----------------------------
# Paramètres du quartier
# -----------------------------
lot_width = 150.0
lot_length = 300.0
street_width = 20.0
min_h = 15.0
max_h = 40.0
building_depth = 12.0
gap = 5.0
n_buildings_side = 5

# Domaine fluide
mx_in, mx_out = 1.5, 3.0
my, mz = 1.5, 2.0

# Taille des éléments
lc_min, lc_max = 3.0, 5.0

# -----------------------------
# 1. Sol
# -----------------------------
with BuildPart() as ground:
    Box(lot_length, lot_width, 1)
ground_part = ground.part
ground_part.label = "GROUND"

# -----------------------------
# 2. Fonction pour créer un bâtiment
# -----------------------------
def make_building(x, y, w, idx):
    h = random.uniform(min_h, max_h)
    with BuildPart() as b:
        Box(w, building_depth, h)
    b.part.label = f"BUILDING_{idx}"
    b.part = b.part.translate((x, y, 0))
    return b.part

# -----------------------------
# 3. Assemblage bâtiments
# -----------------------------
city = ground_part
space = lot_length - 20
bw = space/n_buildings_side - gap
x = -space/2 + bw/2
y_front = lot_width/2 - street_width - building_depth/2

for i in range(n_buildings_side):
    city += make_building(x, y_front, bw, i+1)
    city += make_building(x, -y_front, bw, i+1+n_buildings_side)
    x += bw + gap

# -----------------------------
# 4. Domaine fluide
# -----------------------------
Dx = lot_length*(1+mx_in+mx_out)
Dy = lot_width*my
Dz = max_h*mz

offset = (mx_out-mx_in)*lot_length/2
 
with BuildPart() as dom:
    Box(Dx, Dy, Dz)
fluid_domain = dom.part.translate((offset, 0, 0))
fluid_domain.label = "FLUID"

# Volume fluide = domaine - ville
fluid_volume = fluid_domain - city


# -----------------------------
# 6. Export STEP pour Gmsh
# -----------------------------
exporters3d.export_step(fluid_volume, current_path / "city_block_cfd_domain.step")
print("STEP exporté.")

print(f"Dx: {Dx}, Dy: {Dy}, Dz: {Dz}")

mesh = Meshing(current_path,mesher="gmsh")
mesh.mesher.load_geometry(current_path / "city_block_cfd_domain.step")

mesh.mesher.assign_boundary_patches(
    xmin = offset - Dx / 2,
    xmax = offset + Dx / 2,
    ymin = -Dy / 2,
    ymax = Dy / 2,
    zmin = -Dz/2,
    zmax = Dz/2
)

mesh.mesher.mesh_volume( lc_min= 2,
                        lc_max= 5,
                       )

mesh.mesher.get_basic_mesh_stats()

mesh.mesher.analyze_mesh_quality()

mesh.mesher.get_volume_tags()

mesh.mesher.get_face_tags()

unassigned = mesh.mesher.get_unassigned_faces()
if unassigned:
    mesh.mesher.define_physical_group(dim=2, tags=unassigned, name="building")
    print(f"Renamed {len(unassigned)} unassigned faces to 'building'")
else:
    print("No unassigned faces found")

mesh.mesher.export_to_openfoam(run_gmshtofoam = True)

mesh.mesher.finalize()