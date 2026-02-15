#!/usr/bin/env python

"""
Simulation CFD d'un quartier urbain avec rotation du domaine pour rose des vents.
Le domaine tourne pour aligner le vent avec l'axe X → meilleure stabilité numérique.
"""

# Import required libraries
from build123d import *
from build123d import exporters3d
import gmsh
import random
import json
import math
from pathlib import Path
import numpy as np

from foampilot import Meshing, commons, postprocess, latex_pdf, ValueWithUnit, FluidMechanics, Solver, utilities
import pyvista as pv

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
        "n_buildings_side": 5
    },
    "domaine_fluide": {
        "mx_in": 1.5,      # Multiplicateur amont
        "mx_out": 3.0,     # Multiplicateur aval
        "my": 1.5,         # Multiplicateur latéral
        "mz": 2.0,         # Multiplicateur vertical
        "rotation_angle": 15.0  # Angle de rotation du domaine (degrés)
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
        "turbulence_intensity": 0.05
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

# 🔑 Modèle de turbulence k-epsilon
solver.constant.turbulenceProperties = {
    'simulationType': 'RAS',
    'RAS': {
        'RASModel': 'kEpsilon',
        'turbulence': 'on',
        'printCoeffs': 'on'
    }
}

# Configuration des schémas numériques
solver.system.fvSchemes = {
    'ddtSchemes': {'default': 'steadyState'},
    'gradSchemes': {'default': 'Gauss linear'},
    'divSchemes': {
        'default': 'none',
        'div(phi,U)': 'bounded Gauss linearUpwind grad(U)',
        'div(phi,k)': 'bounded Gauss upwind',
        'div(phi,epsilon)': 'bounded Gauss upwind',
        'div((nuEff*dev2(T(grad(U)))))': 'Gauss linear'
    },
    'laplacianSchemes': {'default': 'Gauss linear corrected'},
    'interpolationSchemes': {'default': 'linear'},
    'snGradSchemes': {'default': 'corrected'}
}

solver.system.fvSolution = {
    'solvers': {
        'p': {
            'solver': 'GAMG',
            'tolerance': 1e-7,
            'relTol': 0.01,
            'smoother': 'GaussSeidel',
            'nPreSweeps': 0,
            'nPostSweeps': 2,
            'cacheAgglomeration': 'on',
            'agglomerator': 'faceAreaPair',
            'nCellsInCoarsestLevel': 10,
            'mergeLevels': 1
        },
        'U': {
            'solver': 'smoothSolver',
            'smoother': 'symGaussSeidel',
            'tolerance': 1e-7,
            'relTol': 0.1
        },
        'k': {
            'solver': 'smoothSolver',
            'smoother': 'symGaussSeidel',
            'tolerance': 1e-7,
            'relTol': 0.1
        },
        'epsilon': {
            'solver': 'smoothSolver',
            'smoother': 'symGaussSeidel',
            'tolerance': 1e-7,
            'relTol': 0.1
        }
    },
    'SIMPLE': {
        'nNonOrthogonalCorrectors': 0,
        'consistent': 'yes',
        'residualControl': {
            'p': 1e-4,
            'U': 1e-4,
            'k': 1e-4,
            'epsilon': 1e-4
        }
    },
    'relaxationFactors': {
        'fields': {'p': 0.3},
        'equations': {
            'U': 0.7,
            'k': 0.7,
            'epsilon': 0.7
        }
    }
}

# Configuration du controlDict
solver.system.controlDict = {
    'application': 'simpleFoam',
    'startFrom': 'startTime',
    'startTime': 0,
    'stopAt': 'endTime',
    'endTime': 1000,
    'deltaT': 1,
    'writeControl': 'timeStep',
    'writeInterval': 100,
    'purgeWrite': 2,
    'writeFormat': 'ascii',
    'writePrecision': 6,
    'writeCompression': 'off',
    'timeFormat': 'general',
    'timePrecision': 6,
    'runTimeModifiable': True
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

# Sol
with BuildPart() as ground:
    Box(q['lot_length'], q['lot_width'], 1)
city = ground.part
city.label = "GROUND"

# Fonction pour créer un bâtiment
def make_building(x, y, w, idx):
    h = random.uniform(q['min_h'], q['max_h'])
    with BuildPart() as b:
        Box(w, q['building_depth'], h)
    b.part.label = f"BUILDING_{idx}"
    b.part = b.part.translate((x, y, 0))
    return b.part, h

# Placement des bâtiments
random.seed(config['seed'])
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
# Domaine fluide avec rotation
# -----------------------------
print("\n" + "="*50)
print("Construction du domaine fluide")
print("="*50)

# Dimensions du domaine de base
Dx = q['lot_length'] * (1 + d['mx_in'] + d['mx_out'])
Dy = q['lot_width'] * d['my']
Dz = q['max_h'] * d['mz']
offset = (d['mx_out'] - d['mx_in']) * q['lot_length'] / 2

with BuildPart() as dom:
    Box(Dx, Dy, Dz)
fluid_domain_base = dom.part.translate((offset, 0, 0))
fluid_domain_base.label = "FLUID_BASE"

# 🔑 Calcul des limites AVANT rotation
bbox_before = fluid_domain_base.bounding_box()
xmin_before = bbox_before.min.X
xmax_before = bbox_before.max.X
ymin_before = bbox_before.min.Y
ymax_before = bbox_before.max.Y
zmin_before = bbox_before.min.Z
zmax_before = bbox_before.max.Z

print(f"\nDomaine AVANT rotation:")
print(f"  X: [{xmin_before:.2f}, {xmax_before:.2f}] (Δx={xmax_before-xmin_before:.2f})")
print(f"  Y: [{ymin_before:.2f}, {ymax_before:.2f}] (Δy={ymax_before-ymin_before:.2f})")
print(f"  Z: [{zmin_before:.2f}, {zmax_before:.2f}] (Δz={zmax_before-zmin_before:.2f})")

# 🔑 Application de la rotation autour de l'axe Z
if d['rotation_angle'] != 0:
    angle_rad = math.radians(d['rotation_angle'])
    rotation = Rotation(axis=(0, 0, 1), angle=angle_rad)
    fluid_domain_rotated = fluid_domain_base.rotate(rotation)
    fluid_domain_rotated.label = "FLUID_ROTATED"
    print(f"\n✅ Rotation appliquée: {d['rotation_angle']}°")
else:
    fluid_domain_rotated = fluid_domain_base
    print(f"\n⚠️ Pas de rotation (angle = 0°)")

# 🔑 Calcul des NOUVELLES limites APRÈS rotation
bbox_after = fluid_domain_rotated.bounding_box()
xmin_after = bbox_after.min.X
xmax_after = bbox_after.max.X
ymin_after = bbox_after.min.Y
ymax_after = bbox_after.max.Y
zmin_after = bbox_after.min.Z
zmax_after = bbox_after.max.Z

print(f"\nDomaine APRÈS rotation:")
print(f"  X: [{xmin_after:.2f}, {xmax_after:.2f}] (Δx={xmax_after-xmin_after:.2f})")
print(f"  Y: [{ymin_after:.2f}, {ymax_after:.2f}] (Δy={ymax_after-ymin_after:.2f})")
print(f"  Z: [{zmin_after:.2f}, {zmax_after:.2f}] (Δz={zmax_after-zmin_after:.2f})")

# Volume fluide = domaine - ville
fluid_volume = fluid_domain_rotated - city

# Export STEP pour Gmsh
step_path = current_path / "city_block_cfd_domain.step"
exporters3d.export_step(fluid_volume, step_path)
print(f"\n✅ STEP exporté: {step_path}")

# -----------------------------
# Maillage avec Gmsh
# -----------------------------
print("\n" + "="*50)
print("Maillage avec Gmsh")
print("="*50)

mesh = Meshing(current_path, mesher="gmsh")
mesh.mesher.load_geometry(step_path)

# 🔑 Attribution des patches basée sur les limites APRÈS rotation
mesh.mesher.assign_boundary_patches(
    xmin=xmin_after,  # ⚠️ CRUCIAL: Utiliser les limites APRÈS rotation
    xmax=xmax_after,
    ymin=ymin_after,
    ymax=ymax_after,
    zmin=zmin_after,
    zmax=zmax_after
)

# Création du maillage volume
mesh.mesher.mesh_volume(
    lc_min=m['lc_min'],
    lc_max=m['lc_max']
)

# Statistiques du maillage
print("\nStatistiques du maillage:")
mesh.mesher.get_basic_mesh_stats()
mesh.mesher.analyze_mesh_quality()
mesh.mesher.get_volume_tags()
mesh.mesher.get_face_tags()

# Identification des faces non assignées (bâtiments)
unassigned = mesh.mesher.get_unassigned_faces()
if unassigned:
    mesh.mesher.define_physical_group(dim=2, tags=unassigned, name="buildings")
    print(f"✅ Faces des bâtiments identifiées: {len(unassigned)}")
else:
    print("⚠️ Attention: aucune face de bâtiment identifiée")

# Export pour OpenFOAM
mesh.mesher.export_to_openfoam(run_gmshtofoam=True)
print("✅ Maillage converti pour OpenFOAM")

mesh.mesher.finalize()

# -----------------------------
# 🔑 Lecture et classification des patches
# -----------------------------
print("\n" + "="*50)
print("Lecture et classification des patches")
print("="*50)

def read_openfoam_boundary(case_path):
    """Lit le fichier boundary et retourne les patches."""
    boundary_file = case_path / "constant" / "polyMesh" / "boundary"
    
    if not boundary_file.exists():
        print(f"⚠️ Fichier boundary non trouvé: {boundary_file}")
        return {}
    
    patches = {}
    current_patch = None
    in_patch = False
    
    with open(boundary_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Détecter un nouveau patch
        if line and not line.startswith('//') and not line.startswith('/*'):
            if not any(c in line for c in ['{', '}', '(', ')', ';']) or 'type' in line or 'nFaces' in line:
                if i+1 < len(lines) and '{' in lines[i+1] and 'type' not in line:
                    current_patch = line.replace('"', '').strip()
                    if current_patch:
                        patches[current_patch] = {'type': None, 'nFaces': None}
                        in_patch = True
            
            # Extraire les infos
            if in_patch and current_patch:
                if 'type' in line and ';' in line:
                    patch_type = line.split('type')[1].replace(';', '').strip()
                    patches[current_patch]['type'] = patch_type
                
                if 'nFaces' in line and ';' in line:
                    try:
                        nFaces = int(line.split('nFaces')[1].replace(';', '').strip())
                        patches[current_patch]['nFaces'] = nFaces
                    except:
                        pass
                
                if line == '}':
                    in_patch = False
                    current_patch = None
    
    return patches

def identify_patch_type(patch_name, xmin, xmax, ymin, ymax, zmin, zmax, tolerance=1.0):
    """Identifie le type de patch selon son nom/position."""
    patch_lower = patch_name.lower()
    
    # Vérifier si le nom contient des indices de position
    if 'xmin' in patch_lower or f'{xmin:.0f}' in patch_name or f'{xmin:.1f}' in patch_name:
        return 'inlet'
    elif 'xmax' in patch_lower or f'{xmax:.0f}' in patch_name or f'{xmax:.1f}' in patch_name:
        return 'outlet'
    elif 'ymin' in patch_lower or f'{ymin:.0f}' in patch_name or f'{ymin:.1f}' in patch_name:
        return 'side_left'
    elif 'ymax' in patch_lower or f'{ymax:.0f}' in patch_name or f'{ymax:.1f}' in patch_name:
        return 'side_right'
    elif 'zmin' in patch_lower or f'{zmin:.0f}' in patch_name or f'{zmin:.1f}' in patch_name:
        return 'ground'
    elif 'zmax' in patch_lower or f'{zmax:.0f}' in patch_name or f'{zmax:.1f}' in patch_name:
        return 'top'
    elif 'building' in patch_lower:
        return 'buildings'
    else:
        return 'unknown'

# Lecture des patches
patches_info = read_openfoam_boundary(current_path)
print("\nPatches créés par Gmsh:")
for patch_name, info in patches_info.items():
    print(f"  {patch_name:35s} | Type: {info['type']:15s} | Faces: {info['nFaces']}")

# Classification
print("\nClassification des patches:")
patch_classification = {}
for patch_name in patches_info.keys():
    patch_type = identify_patch_type(
        patch_name, 
        xmin_after, xmax_after, 
        ymin_after, ymax_after, 
        zmin_after, zmax_after
    )
    patch_classification[patch_name] = patch_type
    print(f"  {patch_name:35s} → {patch_type}")

# -----------------------------
# Conditions aux limites
# -----------------------------
print("\n" + "="*50)
print("Configuration des conditions aux limites")
print("="*50)

# Calcul des valeurs de turbulence
U_inlet = sim['inlet_velocity']
I = sim['turbulence_intensity']
k_inlet = 1.5 * (U_inlet * I) ** 2
# Longueur de turbulence estimée
L = 0.07 * min(Dx, Dy)
epsilon_inlet = 0.09**(0.75) * k_inlet**(1.5) / L

print(f"\nValeurs de turbulence:")
print(f"  k = {k_inlet:.6f} m²/s²")
print(f"  epsilon = {epsilon_inlet:.6f} m²/s³")

solver.boundary.initialize_boundary()

# 🔑 Application des BC selon la classification
for patch_name, patch_type in patch_classification.items():
    
    if patch_type == 'inlet':
        print(f"\n✅ {patch_name:35s} → INLET (velocityInlet)")
        # 🔑 Vitesse selon X car domaine tourné pour aligner vent avec X
        solver.boundary.apply_condition_with_wildcard(
            pattern=patch_name,
            condition_type="velocityInlet",
            velocity=(
                ValueWithUnit(U_inlet, "m/s"),  # Vent selon X
                ValueWithUnit(0.0, "m/s"),
                ValueWithUnit(0.0, "m/s")
            ),
            turbulence_intensity=I
        )
    
    elif patch_type == 'outlet':
        print(f"✅ {patch_name:35s} → OUTLET (pressureOutlet)")
        solver.boundary.apply_condition_with_wildcard(
            pattern=patch_name,
            condition_type="pressureOutlet",
            pressure=ValueWithUnit(0, "Pa")
        )
    
    elif patch_type in ['side_left', 'side_right']:
        print(f"✅ {patch_name:35s} → SIDE (symmetry)")
        solver.boundary.apply_condition_with_wildcard(
            pattern=patch_name,
            condition_type="symmetry"
        )
    
    elif patch_type == 'top':
        print(f"✅ {patch_name:35s} → TOP (symmetry)")
        solver.boundary.apply_condition_with_wildcard(
            pattern=patch_name,
            condition_type="symmetry"
        )
    
    elif patch_type == 'ground':
        print(f"✅ {patch_name:35s} → GROUND (wall)")
        solver.boundary.apply_condition_with_wildcard(
            pattern=patch_name,
            condition_type="wall"
        )
    
    elif patch_type == 'buildings':
        print(f"✅ {patch_name:35s} → BUILDINGS (wall)")
        solver.boundary.apply_condition_with_wildcard(
            pattern=patch_name,
            condition_type="wall"
        )
    
    else:
        print(f"⚠️ {patch_name:35s} → NON CLASSIFIÉ")

# Écriture des conditions aux limites
solver.boundary.write_boundary_conditions()
print("\n✅ Conditions aux limites écrites dans 0/")

# -----------------------------
# Validation du cas
# -----------------------------
print("\n" + "="*50)
print("VALIDATION DU CAS")
print("="*50)

validation_ok = True

# Vérifier les fichiers BC
bc_dir = current_path / "0"
required_fields = ['U', 'p', 'k', 'epsilon', 'nut']

for field in required_fields:
    field_file = bc_dir / field
    if field_file.exists():
        print(f"✅ {field:10s} : OK")
    else:
        print(f"❌ {field:10s} : MANQUANT")
        validation_ok = False

if validation_ok:
    print("\n✅ CAS VALIDÉ - Prêt pour simulation")
else:
    print("\n❌ CAS NON VALIDÉ - Vérifier les erreurs")

# -----------------------------
# Exécution de la simulation
# -----------------------------
if validation_ok:
    print("\n" + "="*50)
    print("Exécution de la simulation")
    print("="*50)
    
    try:
        solver.run_simulation()
        print("\n✅ Simulation terminée avec succès")
    except Exception as e:
        print(f"\n❌ Erreur lors de la simulation: {e}")
        import traceback
        traceback.print_exc()

# -----------------------------
# Post-traitement
# -----------------------------
print("\n" + "="*50)
print("Post-traitement")
print("="*50)

# Analyse des résidus
log_file = current_path / "log.simpleFoam"
if log_file.exists():
    print("\nAnalyse des résidus...")
    residuals_post = utilities.ResidualsPost(log_file)
    residuals_post.process(export_csv=True, export_json=True, export_png=True, export_html=True)
    print("✅ Résidus analysés")
else:
    print(f"⚠️ Fichier log non trouvé: {log_file}")

# Conversion en VTK
foam_post = postprocess.FoamPostProcessing(case_path=current_path)
foam_post.foamToVTK()

# Chargement des résultats
time_steps = foam_post.get_all_time_steps()
print(f"\nTime steps disponibles: {time_steps}")

if time_steps:
    latest_time_step = time_steps[-1]
    structure = foam_post.load_time_step(latest_time_step)
    cell_mesh = structure["cell"]
    boundaries = structure["boundaries"]
    
    print(f"\nMaillage chargé: {cell_mesh.n_cells} cellules")
    print(f"Frontières: {list(boundaries.keys())}")
    
    # Création du dossier pour les visualisations
    viz_dir = current_path / "visualisations"
    viz_dir.mkdir(exist_ok=True)
    
    # Visualisations
    print("\n" + "="*50)
    print("Génération des visualisations")
    print("="*50)
    
    # 1. Coupe horizontale
    print("→ Coupe horizontale...")
    foam_post.plot_slice(
        structure=structure,
        plane="z",
        origin=(0, 0, q['max_h']/2),
        scalars="U",
        opacity=0.5,
        path_filename=viz_dir / "slice_horizontale.png"
    )
    
    # 2. Coupe verticale
    print("→ Coupe verticale...")
    foam_post.plot_slice(
        structure=structure,
        plane="y",
        origin=(0, 0, 0),
        scalars="p",
        opacity=0.5,
        path_filename=viz_dir / "slice_verticale.png"
    )
    
    # 3. Contours de vitesse
    print("→ Contours de vitesse...")
    pl_contour = pv.Plotter(off_screen=True)
    pl_contour.add_mesh(cell_mesh, scalars='U', show_scalar_bar=True, opacity=0.7)
    if 'buildings' in boundaries:
        pl_contour.add_mesh(boundaries['buildings'], color='red', opacity=0.5)
    foam_post.export_plot(pl_contour, viz_dir / "contour_vitesse.png")
    
    # 4. Vue 3D
    print("→ Vue 3D...")
    pl_3d = pv.Plotter(off_screen=True)
    pl_3d.add_mesh(cell_mesh, scalars='U', show_scalar_bar=True, opacity=0.5)
    if 'buildings' in boundaries:
        pl_3d.add_mesh(boundaries['buildings'], color='red', opacity=0.8)
    pl_3d.camera_position = 'iso'
    foam_post.export_plot(pl_3d, viz_dir / "vue_3d.png")
    
    print(f"\n✅ Visualisations sauvegardées dans: {viz_dir}")
    
    # Statistiques
    print("\n" + "="*50)
    print("Statistiques")
    print("="*50)
    
    # Vitesse moyenne dans la rue
    rue_mask = (
        (cell_mesh.points[:, 1] > -q['street_width']/2) & 
        (cell_mesh.points[:, 1] < q['street_width']/2) &
        (cell_mesh.points[:, 2] < 5)
    )
    
    if any(rue_mask):
        U_street = cell_mesh.point_data['U'][rue_mask]
        U_mag_street = np.linalg.norm(U_street, axis=1)
        
        print(f"\nVitesse dans la rue (hauteur piétonne, z<5m):")
        print(f"  Moyenne: {np.mean(U_mag_street):.2f} m/s")
        print(f"  Maximum: {np.max(U_mag_street):.2f} m/s")
        print(f"  Minimum: {np.min(U_mag_street):.2f} m/s")
    
    # Export CSV
    foam_post.export_region_data_to_csv(
        structure, "cell", ["U", "p", "k", "epsilon"], 
        viz_dir / "field_data.csv"
    )
    print(f"\n✅ Données exportées dans: {viz_dir / 'field_data.csv'}")

else:
    print("⚠️ Aucun time step trouvé - post-traitement impossible")

# -----------------------------
# Résumé final
# -----------------------------
print("\n" + "="*70)
print("RÉSUMÉ DE LA SIMULATION")
print("="*70)
print(f"Nombre de bâtiments: {len(buildings_data)}")
print(f"Angle de rotation: {d['rotation_angle']}°")
print(f"Dimensions domaine (après rotation):")
print(f"  X: [{xmin_after:.1f}, {xmax_after:.1f}] m")
print(f"  Y: [{ymin_after:.1f}, {ymax_after:.1f}] m")
print(f"  Z: [{zmin_after:.1f}, {zmax_after:.1f}] m")
print(f"Vitesse d'entrée: {sim['inlet_velocity']} m/s (selon X)")
print(f"Taille de maille: {m['lc_min']} - {m['lc_max']} m")
print(f"\nConditions aux limites appliquées:")
for patch_name, patch_type in patch_classification.items():
    print(f"  - {patch_name[:30]:30s} : {patch_type}")
print(f"\nDossier de travail: {current_path}")
print(f"Configuration: {config_path}")
print("="*70)
print("\n✅ Script terminé avec succès!\n")
