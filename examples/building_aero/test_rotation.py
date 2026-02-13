from build123d import *
from build123d import exporters3d
import gmsh
import random
import json
import math
from pathlib import Path

# Import required libraries
from foampilot import Meshing, commons, postprocess,latex_pdf,ValueWithUnit,FluidMechanics,Solver

current_path = Path.cwd() / "exemple_gmsh"
current_path.mkdir(exist_ok=True)

# -----------------------------
# Configuration complète dans un dictionnaire
# -----------------------------
config = {
    "quartier": {
        "lot_width": 150.0,
        "lot_length": 300.0,
        "street_width": 20.0,
        "min_h": 15.0,
        "max_h": 40.0,
        "building_depth": 12.0,
        "gap": 5.0,
        "n_buildings_side": 5
    },
    "domaine_fluide": {
        "mx_in": 1.5,
        "mx_out": 3.0,
        "my": 1.5,
        "mz": 2.0,
        "rotation_angle": 15.0  # Nouveau paramètre de rotation
    },
    "maillage": {
        "lc_min": 2.0,
        "lc_max": 5.0
    },
    "fluide": {
        "nom": "Air",
        "temperature": 293.15,
        "pression": 101325
    },
    "seed": 42  # Pour reproductibilité des hauteurs aléatoires
}

# -----------------------------
# Sauvegarde de la configuration
# -----------------------------
config_path = current_path / "buildings_config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"Configuration sauvegardée dans: {config_path}")

# -----------------------------
# Initialisation du fluide et solveur
# -----------------------------
print("Available fluids:")
available_fluids = FluidMechanics.get_available_fluids()
for name in available_fluids:
    print(f"- {name}")

fluid_mech = FluidMechanics(
    available_fluids[config['fluide']['nom']],
    temperature=ValueWithUnit(config['fluide']['temperature'], "K"),
    pressure=ValueWithUnit(config['fluide']['pression'], "Pa")
)

properties = fluid_mech.get_fluid_properties()
kinematic_viscosity = properties['kinematic_viscosity']
print(f"\nUsing fluid: {config['fluide']['nom']}")
print(f"Kinematic viscosity: {kinematic_viscosity}")

solver = Solver(current_path)
solver.compressible = False
solver.with_gravity = False
solver.constant.transportProperties.nu = kinematic_viscosity

solver.system.write()
solver.constant.write()

# -----------------------------
# Extraction des paramètres
# -----------------------------
q = config['quartier']
d = config['domaine_fluide']
m = config['maillage']

# -----------------------------
# 1. Sol
# -----------------------------
with BuildPart() as ground:
    Box(q['lot_length'], q['lot_width'], 1)
ground_part = ground.part
ground_part.label = "GROUND"

# -----------------------------
# 2. Fonction pour créer un bâtiment
# -----------------------------
def make_building(x, y, w, idx):
    h = random.uniform(q['min_h'], q['max_h'])
    with BuildPart() as b:
        Box(w, q['building_depth'], h)
    b.part.label = f"BUILDING_{idx}"
    b.part = b.part.translate((x, y, 0))
    return b.part, h

# -----------------------------
# 3. Assemblage bâtiments avec sauvegarde des hauteurs
# -----------------------------
random.seed(config['seed'])
city = ground_part
space = q['lot_length'] - 20
bw = space / q['n_buildings_side'] - q['gap']
x = -space/2 + bw/2
y_front = q['lot_width']/2 - q['street_width'] - q['building_depth']/2

buildings_data = []
for i in range(q['n_buildings_side']):
    # Bâtiment avant
    b1, h1 = make_building(x, y_front, bw, i+1)
    city += b1
    buildings_data.append({
        "id": i+1,
        "position": {"x": x, "y": y_front, "z": 0},
        "dimensions": {"width": bw, "depth": q['building_depth'], "height": h1},
        "row": "front"
    })
    
    # Bâtiment arrière
    b2, h2 = make_building(x, -y_front, bw, i+1+q['n_buildings_side'])
    city += b2
    buildings_data.append({
        "id": i+1+q['n_buildings_side'],
        "position": {"x": x, "y": -y_front, "z": 0},
        "dimensions": {"width": bw, "depth": q['building_depth'], "height": h2},
        "row": "back"
    })
    
    x += bw + q['gap']

# Mise à jour du JSON avec les hauteurs réelles
config['buildings'] = buildings_data
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

# -----------------------------
# 4. Domaine fluide avec rotation
# -----------------------------
Dx = q['lot_length'] * (1 + d['mx_in'] + d['mx_out'])
Dy = q['lot_width'] * d['my']
Dz = q['max_h'] * d['mz']
offset = (d['mx_out'] - d['mx_in']) * q['lot_length'] / 2

with BuildPart() as dom:
    Box(Dx, Dy, Dz)
fluid_domain = dom.part.translate((offset, 0, 0))

# Application de la rotation
if d['rotation_angle'] != 0:
    angle_rad = math.radians(d['rotation_angle'])
    rotation = Rotation(axis=(0, 0, 1), angle=angle_rad)
    fluid_domain = fluid_domain.rotate(rotation)
    print(f"Rotation appliquée: {d['rotation_angle']}°")

fluid_domain.label = "FLUID"

# Volume fluide = domaine - ville
fluid_volume = fluid_domain - city

# -----------------------------
# 5. Export STEP
# -----------------------------
step_path = current_path / "city_block_cfd_domain.step"
exporters3d.export_step(fluid_volume, step_path)
print(f"STEP exporté: {step_path}")
print(f"Dx: {Dx:.1f}, Dy: {Dy:.1f}, Dz: {Dz:.1f}")

# -----------------------------
# 6. Maillage avec Gmsh
# -----------------------------
mesh = Meshing(current_path, mesher="gmsh")
mesh.mesher.load_geometry(step_path)

# Calcul des limites du domaine pour les patches
# Important: avec rotation, les limites sont différentes!
bbox = fluid_domain.bounding_box()
mesh.mesher.assign_boundary_patches(
    xmin=bbox.min.X,
    xmax=bbox.max.X,
    ymin=bbox.min.Y,
    ymax=bbox.max.Y,
    zmin=bbox.min.Z,
    zmax=bbox.max.Z
)

mesh.mesher.mesh_volume(lc_min=m['lc_min'], lc_max=m['lc_max'])

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

mesh.mesher.export_to_openfoam(run_gmshtofoam=True)
mesh.mesher.finalize()

# -----------------------------
# 7. Fonctions de reconstruction
# -----------------------------
def rebuild_from_config(config_path, apply_rotation=True):
    """Reconstruit le modèle à partir du fichier de configuration"""
    
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    
    q = cfg['quartier']
    d = cfg['domaine_fluide']
    
    # Reconstruction du sol
    with BuildPart() as ground:
        Box(q['lot_length'], q['lot_width'], 1)
    city = ground.part
    city.label = "GROUND"
    
    # Reconstruction des bâtiments
    for b in cfg['buildings']:
        with BuildPart() as building:
            Box(b['dimensions']['width'], 
                b['dimensions']['depth'], 
                b['dimensions']['height'])
        b_part = building.part.translate((
            b['position']['x'],
            b['position']['y'],
            b['position']['z']
        ))
        b_part.label = f"BUILDING_{b['id']}"
        city += b_part
    
    # Reconstruction du domaine fluide
    Dx = q['lot_length'] * (1 + d['mx_in'] + d['mx_out'])
    Dy = q['lot_width'] * d['my']
    Dz = q['max_h'] * d['mz']
    offset = (d['mx_out'] - d['mx_in']) * q['lot_length'] / 2
    
    with BuildPart() as dom:
        Box(Dx, Dy, Dz)
    fluid_domain = dom.part.translate((offset, 0, 0))
    
    # Application de la rotation si demandée
    if apply_rotation and d['rotation_angle'] != 0:
        angle_rad = math.radians(d['rotation_angle'])
        rotation = Rotation(axis=(0, 0, 1), angle=angle_rad)
        fluid_domain = fluid_domain.rotate(rotation)
    
    fluid_domain.label = "FLUID"
    
    # Volume fluide
    fluid_volume = fluid_domain - city
    
    return fluid_volume, city, fluid_domain, cfg

def remesh_from_config(config_path, output_path=None):
    """Re-maille à partir de la configuration"""
    
    if output_path is None:
        output_path = current_path / "remeshed"
        output_path.mkdir(exist_ok=True)
    
    # Reconstruction du modèle
    fluid_volume, _, _, cfg = rebuild_from_config(config_path)
    
    # Export STEP
    step_path = output_path / "reconstructed_domain.step"
    exporters3d.export_step(fluid_volume, step_path)
    
    # Maillage
    mesh = Meshing(output_path, mesher="gmsh")
    mesh.mesher.load_geometry(step_path)
    
    bbox = fluid_volume.bounding_box()
    mesh.mesher.assign_boundary_patches(
        xmin=bbox.min.X,
        xmax=bbox.max.X,
        ymin=bbox.min.Y,
        ymax=bbox.max.Y,
        zmin=bbox.min.Z,
        zmax=bbox.max.Z
    )
    
    mesh.mesher.mesh_volume(
        lc_min=cfg['maillage']['lc_min'],
        lc_max=cfg['maillage']['lc_max']
    )
    
    mesh.mesher.get_basic_mesh_stats()
    mesh.mesher.analyze_mesh_quality()
    mesh.mesher.get_volume_tags()
    mesh.mesher.get_face_tags()
    
    unassigned = mesh.mesher.get_unassigned_faces()
    if unassigned:
        mesh.mesher.define_physical_group(dim=2, tags=unassigned, name="building")
    
    mesh.mesher.export_to_openfoam(run_gmshtofoam=True)
    mesh.mesher.finalize()
    
    return mesh

# -----------------------------
# 8. Test de reconstruction
# -----------------------------
print("\n" + "="*50)
print("Test de reconstruction à partir du JSON")
print("="*50)

# Reconstruction sans rotation pour vérification
reconstructed_volume, _, _, _ = rebuild_from_config(config_path, apply_rotation=False)
reconstructed_path = current_path / "reconstructed_no_rotation.step"
exporters3d.export_step(reconstructed_volume, reconstructed_path)
print(f"Modèle sans rotation exporté: {reconstructed_path}")

# Reconstruction avec rotation
reconstructed_rotated, _, _, _ = rebuild_from_config(config_path, apply_rotation=True)
rotated_path = current_path / "reconstructed_with_rotation.step"
exporters3d.export_step(reconstructed_rotated, rotated_path)
print(f"Modèle avec rotation exporté: {rotated_path}")

# Test de remaillage
print("\nTest de remaillage...")
remesh_from_config(config_path)

print("\n" + "="*50)
print("Résumé de la configuration")
print("="*50)
print(f"Angle de rotation: {config['domaine_fluide']['rotation_angle']}°")
print(f"Nombre de bâtiments: {len(config['buildings'])}")
print(f"Hauteurs: min={q['min_h']:.1f}, max={q['max_h']:.1f}")
print(f"Taille maillage: lc_min={m['lc_min']}, lc_max={m['lc_max']}")
print(f"Fluide: {config['fluide']['nom']}")