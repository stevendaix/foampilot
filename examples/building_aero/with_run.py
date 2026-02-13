#!/usr/bin/env python

# Import required libraries
from build123d import *
from build123d import exporters3d
import gmsh
import random
import json
import math
from pathlib import Path

from foampilot import Meshing, commons, postprocess, latex_pdf, ValueWithUnit, FluidMechanics, Solver
import pyvista as pv
import numpy as np

# Define the working directory for the simulation case
current_path = Path.cwd() / 'quartier_gmsh'
current_path.mkdir(exist_ok=True)

# -----------------------------
# Configuration complète
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
        "n_buildings_side": 5,
        "rotation_angle": 15.0  # Rotation du quartier par rapport au vent
    },
    "domaine_fluide": {
        "mx_in": 1.5,      # Multiplicateur amont
        "mx_out": 3.0,     # Multiplicateur aval
        "my": 1.5,         # Multiplicateur latéral
        "mz": 2.0          # Multiplicateur vertical
        # Plus de rotation ici - le domaine est fixe
    },
    "maillage": {
        "lc_min": 2.0,      # Taille minimale des éléments
        "lc_max": 5.0,      # Taille maximale des éléments
        "raffinement_batiments": 1.5  # Facteur de raffinement près des bâtiments
    },
    "fluide": {
        "nom": "Air",
        "temperature": 293.15,  # K
        "pression": 101325       # Pa
    },
    "simulation": {
        "inlet_velocity": 5.0,   # m/s, vitesse du vent
        "turbulence_intensity": 0.05,
        "direction_vent": [1, 0, 0]  # Vent selon x (fixe)
    },
    "seed": 42  # Pour reproductibilité
}

# Sauvegarde de la configuration
config_path = current_path / "buildings_config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"Configuration sauvegardée dans: {config_path}")

# -----------------------------
# Initialisation du fluide
# -----------------------------
print("\n" + "="*50)
print("Initialisation du fluide")
print("="*50)

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

# -----------------------------
# Initialisation du solveur
# -----------------------------
print("\n" + "="*50)
print("Initialisation du solveur")
print("="*50)

solver = Solver(current_path)
solver.compressible = False
solver.with_gravity = False

# Configuration de la viscosité
solver.constant.transportProperties.nu = kinematic_viscosity

# Configuration des schémas numériques
solver.system.fvSchemes = {
    'ddtSchemes': {'default': 'Euler'},
    'gradSchemes': {'default': 'Gauss linear'},
    'divSchemes': {
        'default': 'none',
        'div(phi,U)': 'Gauss linearUpwind grad(U)',
        'div(phi,k)': 'Gauss upwind',
        'div(phi,epsilon)': 'Gauss upwind',
        'div((nuEff*dev2(T(grad(U)))))': 'Gauss linear'
    },
    'laplacianSchemes': {'default': 'Gauss linear corrected'},
    'interpolationSchemes': {'default': 'linear'},
    'snGradSchemes': {'default': 'corrected'}
}

solver.system.fvSolution = {
    'solvers': {
        'p': {'solver': 'PCG', 'preconditioner': 'DIC', 'tolerance': 1e-6, 'relTol': 0.05},
        'U': {'solver': 'PBiCGStab', 'preconditioner': 'DILU', 'tolerance': 1e-6, 'relTol': 0.1},
        'k': {'solver': 'PBiCGStab', 'preconditioner': 'DILU', 'tolerance': 1e-6, 'relTol': 0.1},
        'epsilon': {'solver': 'PBiCGStab', 'preconditioner': 'DILU', 'tolerance': 1e-6, 'relTol': 0.1}
    },
    'SIMPLE': {'nNonOrthogonalCorrectors': 0, 'pRefCell': 0, 'pRefValue': 0},
    'relaxationFactors': {
        'fields': {'p': 0.3},
        'equations': {'U': 0.7, 'k': 0.7, 'epsilon': 0.7}
    }
}

# Écriture des fichiers de configuration
solver.system.write()
solver.constant.write()

# -----------------------------
# Paramètres
# -----------------------------
q = config['quartier']
d = config['domaine_fluide']
m = config['maillage']
sim = config['simulation']

# -----------------------------
# Construction géométrique avec build123d
# -----------------------------
print("\n" + "="*50)
print("Construction de la géométrie avec build123d")
print("="*50)

# Sol (non rotaté pour l'instant - on va tout construire et tout faire tourner)
with BuildPart() as ground:
    Box(q['lot_length'], q['lot_width'], 1)
ground_part = ground.part
ground_part.label = "GROUND"

# Fonction pour créer un bâtiment
def make_building(x, y, w, idx):
    h = random.uniform(q['min_h'], q['max_h'])
    with BuildPart() as b:
        Box(w, q['building_depth'], h)
    b.part.label = f"BUILDING_{idx}"
    b.part = b.part.translate((x, y, 0))
    return b.part, h

# Placement des bâtiments (dans le repère local du quartier)
random.seed(config['seed'])
space = q['lot_length'] - 20
bw = space / q['n_buildings_side'] - q['gap']
x = -space/2 + bw/2
y_front = q['lot_width']/2 - q['street_width'] - q['building_depth']/2

buildings_data = []
city_parts = [ground_part]

for i in range(q['n_buildings_side']):
    # Bâtiment avant
    b1, h1 = make_building(x, y_front, bw, i+1)
    city_parts.append(b1)
    buildings_data.append({
        "id": i+1,
        "position": {"x": x, "y": y_front, "z": 0},
        "dimensions": {"width": bw, "depth": q['building_depth'], "height": h1},
        "row": "front"
    })
    
    # Bâtiment arrière
    b2, h2 = make_building(x, -y_front, bw, i+1+q['n_buildings_side'])
    city_parts.append(b2)
    buildings_data.append({
        "id": i+1+q['n_buildings_side'],
        "position": {"x": x, "y": -y_front, "z": 0},
        "dimensions": {"width": bw, "depth": q['building_depth'], "height": h2},
        "row": "back"
    })
    
    x += bw + q['gap']

# Assemblage de la ville (sans rotation encore)
city = city_parts[0]
for part in city_parts[1:]:
    city += part

# Mise à jour du JSON avec les hauteurs réelles
config['buildings'] = buildings_data
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

# -----------------------------
# Application de la rotation au quartier
# -----------------------------
print(f"\nRotation du quartier: {q['rotation_angle']}°")

if q['rotation_angle'] != 0:
    angle_rad = math.radians(q['rotation_angle'])
    rotation = Rotation(axis=(0, 0, 1), angle=angle_rad)
    city = city.rotate(rotation)
    print(f"Quartier tourné de {q['rotation_angle']}°")

# Export du modèle des bâtiments (optionnel)
buildings_step = current_path / "buildings_rotated.step"
exporters3d.export_step(city, buildings_step)
print(f"Bâtiments exportés: {buildings_step}")

# -----------------------------
# Domaine fluide FIXE (aligné avec le vent)
# -----------------------------
print("\n" + "="*50)
print("Création du domaine fluide (fixe)")
print("="*50)

# Dimensions du domaine - calculées pour englober le quartier tourné
# On prend une boîte assez grande pour contenir le quartier après rotation
max_dim = max(q['lot_length'], q['lot_width']) * 1.5  # Sécurité pour la rotation
Dx = max_dim * (1 + d['mx_in'] + d['mx_out'])
Dy = max_dim * d['my']
Dz = q['max_h'] * d['mz']
offset = (d['mx_out'] - d['mx_in']) * max_dim / 2

with BuildPart() as dom:
    Box(Dx, Dy, Dz)
fluid_domain = dom.part.translate((offset, 0, 0))
fluid_domain.label = "FLUID"

# Calcul des limites du domaine fixe
bbox_dom = fluid_domain.bounding_box()
xmin_dom, xmax_dom = bbox_dom.min.X, bbox_dom.max.X
ymin_dom, ymax_dom = bbox_dom.min.Y, bbox_dom.max.Y
zmin_dom, zmax_dom = bbox_dom.min.Z, bbox_dom.max.Z

print(f"Domaine fixe:")
print(f"  x: [{xmin_dom:.1f}, {xmax_dom:.1f}] (Δx={xmax_dom-xmin_dom:.1f})")
print(f"  y: [{ymin_dom:.1f}, {ymax_dom:.1f}] (Δy={ymax_dom-ymin_dom:.1f})")
print(f"  z: [{zmin_dom:.1f}, {zmax_dom:.1f}] (Δz={zmax_dom-zmin_dom:.1f})")

# Volume fluide = domaine fixe - quartier tourné
fluid_volume = fluid_domain - city

# Export STEP pour Gmsh
step_path = current_path / "city_block_cfd_domain.step"
exporters3d.export_step(fluid_volume, step_path)
print(f"STEP exporté: {step_path}")

# -----------------------------
# Maillage avec Gmsh
# -----------------------------
print("\n" + "="*50)
print("Maillage avec Gmsh")
print("="*50)

mesh = Meshing(current_path, mesher="gmsh")
mesh.mesher.load_geometry(step_path)

# Attribution des patches sur le domaine FIXE
# C'est beaucoup plus simple car le domaine est aligné avec les axes !
mesh.mesher.assign_boundary_patches(
    xmin=xmin_dom,
    xmax=xmax_dom,
    ymin=ymin_dom,
    ymax=ymax_dom,
    zmin=zmin_dom,
    zmax=zmax_dom
)

# Création du maillage volume
mesh.mesher.mesh_volume(
    lc_min=m['lc_min'],
    lc_max=m['lc_max']
)

# Statistiques du maillage
mesh.mesher.get_basic_mesh_stats()
mesh.mesher.analyze_mesh_quality()
mesh.mesher.get_volume_tags()
mesh.mesher.get_face_tags()

# Identification des faces non assignées (bâtiments)
unassigned = mesh.mesher.get_unassigned_faces()
if unassigned:
    mesh.mesher.define_physical_group(dim=2, tags=unassigned, name="buildings")
    print(f"Faces des bâtiments identifiées: {len(unassigned)}")
else:
    print("Attention: aucune face de bâtiment identifiée")

# Export pour OpenFOAM
mesh.mesher.export_to_openfoam(run_gmshtofoam=True)
print("Maillage converti pour OpenFOAM")

mesh.mesher.finalize()

# -----------------------------
# Configuration des conditions aux limites
# -----------------------------
print("\n" + "="*50)
print("Configuration des conditions aux limites")
print("="*50)

solver.boundary.initialize_boundary()

# Inlet (face xmin - vent qui entre)
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(
        ValueWithUnit(sim['inlet_velocity'], "m/s"),  # Vent selon x
        ValueWithUnit(0, "m/s"),
        ValueWithUnit(0, "m/s")
    ),
    turbulence_intensity=sim['turbulence_intensity']
)

# Outlet (face xmax - pression imposée)
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet",
    pressure=ValueWithUnit(0, "Pa")
)

# Faces latérales (ymin et ymax) - symétrie
solver.boundary.apply_condition_with_wildcard(
    pattern="ymin",
    condition_type="symmetry"
)
solver.boundary.apply_condition_with_wildcard(
    pattern="ymax",
    condition_type="symmetry"
)

# Top (ciel) - symétrie
solver.boundary.apply_condition_with_wildcard(
    pattern="zmax",
    condition_type="symmetry"
)

# Bottom (sol) - wall
solver.boundary.apply_condition_with_wildcard(
    pattern="zmin",
    condition_type="wall"
)

# Buildings - wall
solver.boundary.apply_condition_with_wildcard(
    pattern="buildings",
    condition_type="wall"
)

# Écriture des conditions aux limites
solver.boundary.write_boundary_conditions()
print("Conditions aux limites écrites dans 0/")

# -----------------------------
# Exécution de la simulation
# -----------------------------
print("\n" + "="*50)
print("Exécution de la simulation")
print("="*50)

solver.run_simulation()

# -----------------------------
# Post-traitement
# -----------------------------
print("\n" + "="*50)
print("Post-traitement")
print("="*50)

# Analyse des résidus
if (current_path / "log.simpleFoam").exists():
    residuals_post = utilities.ResidualsPost(current_path / "log.simpleFoam")
    residuals_post.process(export_csv=True, export_json=True, export_png=True, export_html=True)
    print("Résidus analysés")

# Conversion en VTK pour visualisation
foam_post = postprocess.FoamPostProcessing(case_path=current_path)
foam_post.foamToVTK()

# Chargement des résultats
time_steps = foam_post.get_all_time_steps()
print(f"Time steps disponibles: {time_steps}")

if time_steps:
    latest_time_step = time_steps[-1]
    structure = foam_post.load_time_step(latest_time_step)
    cell_mesh = structure["cell"]
    boundaries = structure["boundaries"]
    
    print(f"Maillage chargé: {cell_mesh.n_cells} cellules")
    print(f"Frontières: {list(boundaries.keys())}")
    
    # Création du dossier pour les visualisations
    viz_dir = current_path / "visualisations"
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Coupe horizontale
    foam_post.plot_slice(
        structure=structure,
        plane="z",
        origin=(0, 0, 10),  # Hauteur piétonne
        scalars="U",
        opacity=0.5,
        path_filename=viz_dir / "slice_horizontale.png"
    )
    
    # 2. Coupe verticale
    foam_post.plot_slice(
        structure=structure,
        plane="x",
        origin=(0, 0, 0),
        scalars="p",
        opacity=0.5,
        path_filename=viz_dir / "slice_verticale.png"
    )
    
    # 3. Visualisation 3D
    pl_3d = pv.Plotter(off_screen=True)
    pl_3d.add_mesh(cell_mesh, scalars='U', show_scalar_bar=True, opacity=0.5)
    if 'buildings' in boundaries:
        pl_3d.add_mesh(boundaries['buildings'], color='red', opacity=0.8)
    foam_post.export_plot(pl_3d, viz_dir / "vue_3d.png")
    
    # 4. Lignes de courant
    seeds = pv.Line(
        pointa=(xmin_dom + 10, -50, 10),
        pointb=(xmin_dom + 10, 50, 10),
        resolution=30
    )
    streamlines = cell_mesh.streamlines(
        vectors='U',
        source_radius=5.0,
        integration_direction='forward',
        max_time=200.0
    )
    
    pl_stream = pv.Plotter(off_screen=True)
    pl_stream.add_mesh(cell_mesh, style='wireframe', opacity=0.1)
    pl_stream.add_mesh(streamlines, line_width=2, color='blue')
    foam_post.export_plot(pl_stream, viz_dir / "streamlines.png")
    
    print(f"Visualisations sauvegardées dans: {viz_dir}")

# -----------------------------
# Résumé final
# -----------------------------
print("\n" + "="*50)
print("RÉSUMÉ DE LA SIMULATION")
print("="*50)
print(f"Nombre de bâtiments: {len(buildings_data)}")
print(f"Rotation du quartier: {q['rotation_angle']}°")
print(f"Domaine fixe: {xmax_dom-xmin_dom:.1f} x {ymax_dom-ymin_dom:.1f} x {zmax_dom-zmin_dom:.1f} m")
print(f"Vitesse du vent: {sim['inlet_velocity']} m/s (selon x)")
print(f"Conditions aux limites:")
print(f"  - Inlet: x = {xmin_dom:.1f}")
print(f"  - Outlet: x = {xmax_dom:.1f}")
print(f"  - Côtés: y = {ymin_dom:.1f} et {ymax_dom:.1f} (symétrie)")
print(f"  - Sol: z = {zmin_dom:.1f} (wall)")
print(f"  - Ciel: z = {zmax_dom:.1f} (symétrie)")
print(f"  - Bâtiments: wall")
print(f"\nDossier de travail: {current_path}")
print("Simulation terminée avec succès!")

# -----------------------------
# Fonction pour reprise avec différents angles
# -----------------------------
def simuler_angle_rotation(json_path, nouvel_angle):
    """Relance une simulation avec un angle de rotation différent"""
    
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    
    cfg['quartier']['rotation_angle'] = nouvel_angle
    new_path = current_path.parent / f"quartier_angle_{nouvel_angle}"
    new_path.mkdir(exist_ok=True)
    
    new_config = new_path / "config.json"
    with open(new_config, 'w') as f:
        json.dump(cfg, f, indent=2)
    
    print(f"Configuration pour angle {nouvel_angle}° sauvegardée dans {new_config}")
    print(f"Pour lancer: copier ce script dans {new_path} et exécuter")
    
    return new_config