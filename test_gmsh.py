from build123d import *
import gmsh
import random
#!/usr/bin/env python

# Import required libraries
from foampilot.solver import Solver
from foampilot import Meshing, commons, utilities, postprocess,latex_pdf
from foampilot.utilities.fluids_theory import FluidMechanics
import pyvista as pv
from pathlib import Path
from foampilot.utilities.manageunits import Quantity
import numpy as np
import classy_blocks as cb
import json
import pandas as pd

# Define the working directory for the simulation case
current_path = Path.cwd() / 'exemple_gmsh'

# List available fluids
print("Available fluids:")
available_fluids = FluidMechanics.get_available_fluids()
for name in available_fluids:
    print(f"- {name}")

# Create a FluidMechanics instance for water at room temperature and atmospheric pressure
fluid_mech = FluidMechanics(
    available_fluids['Air'],
    temperature=Quantity(293.15, "K"),
    pressure=Quantity(101325, "Pa")
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

# Convert numerical schemes settings to Python dictionary for inspection
solver.system.fvSchemes.to_dict()



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
from build123d import exporters3d
exporters3d.export_step(fluid_volume, current_path / "city_block_cfd_domain.step")
print("STEP exporté.")

# -----------------------------
# 7. Gmsh - maillage 3D
# -----------------------------
gmsh.initialize()
gmsh.model.add("city_block")

# Importer STEP
gmsh.merge(str(current_path / "city_block_cfd_domain.step"))
# Récupérer toutes les faces

# Définir les limites du domaine
xmin = offset - Dx / 2
xmax = offset + Dx / 2
ymin = -Dy / 2
ymax = Dy / 2
zmin = 0
zmax = Dz

# Récupérer tous les volumes
volumes = [v[1] for v in gmsh.model.getEntities(dim=3)]

# Fragmentation pour obtenir des faces séparées
for vol in volumes:
    gmsh.model.occ.fragment([(3, vol)], [])
gmsh.model.occ.synchronize()

# Récupérer toutes les faces après synchronisation
faces = gmsh.model.getEntities(dim=2)

# Tolérance pour la détection des faces
tolerance = 1e-2

# Dictionnaire pour stocker les faces par patch
patches = {
    "INLET": [],
    "OUTLET": [],
    "GROUND": [],
    "TOP": [],
    "SIDE_NORTH": [],
    "SIDE_SOUTH": [],
}

# Liste pour les faces non attribuées
unassigned_faces = []

# Associer chaque face au bon groupe physique
for face in faces:
    dim, tag = face
    com = gmsh.model.occ.getCenterOfMass(dim, tag)
    x, y, z = com
    if abs(x - xmin) < tolerance:
        patches["INLET"].append(tag)
    elif abs(x - xmax) < tolerance:
        patches["OUTLET"].append(tag)
    elif abs(z - zmin) < tolerance:
        patches["GROUND"].append(tag)
    elif abs(z - zmax) < tolerance:
        patches["TOP"].append(tag)
    elif abs(y - ymax) < tolerance:
        patches["SIDE_NORTH"].append(tag)
    elif abs(y - ymin) < tolerance:
        patches["SIDE_SOUTH"].append(tag)
    else:
        unassigned_faces.append(tag)

# Créer les groupes physiques dans Gmsh
for patch_name, face_tags in patches.items():
    if face_tags:  # Si la liste n'est pas vide
        gmsh.model.addPhysicalGroup(2, face_tags, name=patch_name)

# Créer un groupe physique pour les faces non attribuées
if unassigned_faces:
    gmsh.model.addPhysicalGroup(2, unassigned_faces, name="UNASSIGNED")

# Volume
gmsh.model.addPhysicalGroup(3, volumes, name="FLUID")
gmsh.model.occ.synchronize()

# Définir la taille des éléments
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)
gmsh.option.setNumber("Mesh.MshFileVersion", 2)

# Générer le maillage 3D
gmsh.model.mesh.generate(3)
msh_file = current_path / "city_block_cfd_domain.msh"

vtk_file = current_path / "city_block_cfd_domain.vtk"

gmsh.write(str(msh_file))
gmsh.write(str(vtk_file))
gmsh.finalize()
print(f"Maillage 3D généré : {msh_file}")

print("Maillage 3D généré : city_block_cfd_domain.msh")


import pyvista as pv

# Charger le maillage
pv_mesh = pv.read(str(vtk_file))

# Vérifier le type de maillage
print(f"Type de maillage : {type(pv_mesh)}")  # Doit afficher <class 'pyvista.core.unstructured.UnstructuredGrid'>

# Définir les plans de coupe (X=0, Y=0, Z=0)
planes = [
    {"normal": (1, 0, 0), "name": "coupe_X", "origin": pv_mesh.center},  # Coupe selon X
    {"normal": (0, 1, 0), "name": "coupe_Y", "origin": pv_mesh.center},  # Coupe selon Y
    {"normal": (0, 0, 1), "name": "coupe_Z", "origin": pv_mesh.center},  # Coupe selon Z
]

# Générer et sauvegarder les images de coupes
for plane in planes:
    sliced = pv_mesh.slice(normal=plane["normal"], origin=plane["origin"])
    pv.plot(
        sliced,
        screenshot=f"{plane['name']}.png",
        window_size=[1024, 768],
        show_edges=True,
        color="lightblue",
        off_screen=True,  # Mode hors écran pour éviter les problèmes d'affichage
    )
    print(f"Image sauvegardée : {plane['name']}.png")



# -----------------------------
# 8. Conversion Gmsh vers OpenFOAM (gmshToFoam)
# -----------------------------
print("\nConversion du maillage Gmsh vers OpenFOAM (gmshToFoam)...")
try:
    # Appel de la méthode gmshToFoam
    Meshing.gmshToFoam(msh_file, current_path)
    print("Conversion réussie. Les fichiers du maillage OpenFOAM sont dans constant/polyMesh.")
except Exception as e:
    print(f"Erreur lors de la conversion avec gmshToFoam : {e}")

