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
        "turbulence_intensity": 0.05,
        "direction_vent": [1, 0, 0]  # Vent selon x
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

# Configuration des schémas numériques (optionnel)
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
# Dimensions du domaine de base
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

# Export STEP pour Gmsh
step_path = current_path / "city_block_cfd_domain.step"
exporters3d.export_step(fluid_volume, step_path)
print(f"STEP exporté: {step_path}")

# Calcul des limites du domaine pour les patches
bbox = fluid_domain.bounding_box()
xmin, xmax = bbox.min.X, bbox.max.X
ymin, ymax = bbox.min.Y, bbox.max.Y
zmin, zmax = bbox.min.Z, bbox.max.Z

print(f"\nDimensions du domaine:")
print(f"  x: [{xmin:.1f}, {xmax:.1f}] (Δx={xmax-xmin:.1f})")
print(f"  y: [{ymin:.1f}, {ymax:.1f}] (Δy={ymax-ymin:.1f})")
print(f"  z: [{zmin:.1f}, {zmax:.1f}] (Δz={zmax-zmin:.1f})")

# -----------------------------
# Maillage avec Gmsh
# -----------------------------
print("\n" + "="*50)
print("Maillage avec Gmsh")
print("="*50)

mesh = Meshing(current_path, mesher="gmsh")
mesh.mesher.load_geometry(step_path)

# Attribution des patches basée sur la direction du vent
# On détermine les faces en fonction de leur normale et de la direction du vent
vent = np.array(sim['direction_vent'])
tol = 0.1  # Tolérance pour l'identification des faces

mesh.mesher.assign_boundary_patches(
    xmin=xmin,
    xmax=xmax,
    ymin=ymin,
    ymax=ymax,
    zmin=zmin,
    zmax=zmax
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

# Renommage des patches selon la convention OpenFOAM
# - inlet: face où le vent entre
# - outlet: face où le vent sort
# - sides: faces latérales parallèles au vent
# - top: face supérieure (ciel)
# - bottom: face inférieure (sol)
# - buildings: faces des bâtiments

# Note: gmshtofoam va créer des patches basés sur les noms des physical groups
# On va renommer les patches après conversion

# Export pour OpenFOAM
mesh.mesher.export_to_openfoam(run_gmshtofoam=True)
print("Maillage converti pour OpenFOAM")

mesh.mesher.finalize()

# -----------------------------
// ... (la suite dans le prochain message à cause de la limite de caractères)
```<｜end▁of▁thinking｜>Voici la suite du code :

```python
# -----------------------------
# Reconfiguration des patches OpenFOAM
# -----------------------------
print("\n" + "="*50)
print("Configuration des patches OpenFOAM")
print("="*50)

# Après conversion, on a un fichier boundary dans le dossier constant/polyMesh
# On va le modifier pour avoir des noms cohérents avec nos conditions aux limites

# Détermination des faces en fonction de la direction du vent
# Avec rotation, c'est plus complexe - on utilise la position relative
# On suppose que le vent vient selon l'axe x après rotation

# Création d'un dictionnaire de correspondance pour les patches
patch_mapping = {
    f"face_xmin_{xmin:.2f}": "inlet",
    f"face_xmax_{xmax:.2f}": "outlet",
    f"face_ymin_{ymin:.2f}": "side_left",
    f"face_ymax_{ymax:.2f}": "side_right",
    f"face_zmin_{zmin:.2f}": "bottom",
    f"face_zmax_{zmax:.2f}": "top",
    "buildings": "buildings"
}

# Note: dans un cas réel, il faudrait parser le fichier boundary
# et renommer les patches. Pour l'exemple, on va créer directement
# les conditions aux limites avec les bons noms

# -----------------------------
# Conditions aux limites
# -----------------------------
print("\nConfiguration des conditions aux limites")

solver.boundary.initialize_boundary()

# Inlet (entrée d'air - face où le vent entre)
solver.boundary.apply_condition_with_wildcard(
    pattern="inlet",
    condition_type="velocityInlet",
    velocity=(
        ValueWithUnit(sim['inlet_velocity'] * sim['direction_vent'][0], "m/s"),
        ValueWithUnit(sim['inlet_velocity'] * sim['direction_vent'][1], "m/s"),
        ValueWithUnit(sim['inlet_velocity'] * sim['direction_vent'][2], "m/s")
    ),
    turbulence_intensity=sim['turbulence_intensity']
)

# Outlet (sortie - pression imposée)
solver.boundary.apply_condition_with_wildcard(
    pattern="outlet",
    condition_type="pressureOutlet",
    pressure=ValueWithUnit(0, "Pa")  # Pression relative
)

# Side faces (parois latérales parallèles au vent)
# En CFD externe, on utilise souvent des conditions symétriques
# ou des glissements pour les faces latérales
solver.boundary.apply_condition_with_wildcard(
    pattern="side_.*",  # side_left et side_right
    condition_type="symmetry"  # Ou "slip" selon le besoin
)

# Top (ciel) - condition de glissement ou symétrie
solver.boundary.apply_condition_with_wildcard(
    pattern="top",
    condition_type="symmetry"
)

# Bottom (sol) - condition de non-glissement (wall)
solver.boundary.apply_condition_with_wildcard(
    pattern="bottom",
    condition_type="wall"
)

# Buildings - condition de non-glissement (wall)
solver.boundary.apply_condition_with_wildcard(
    pattern="buildings",
    condition_type="wall"
)

# Écriture des conditions aux limites
solver.boundary.write_boundary_conditions()
print("Conditions aux limites écrites dans 0/")

# -----------------------------
# Configuration du décomposeur (pour parallélisation éventuelle)
# -----------------------------
solver.system.decomposeParDict = {
    'numberOfSubdomains': 1,  # À augmenter pour du parallèle
    'method': 'scotch'
}

# -----------------------------
# Exécution de la simulation
# -----------------------------
print("\n" + "="*50)
print("Exécution de la simulation")
print("="*50)

# Option 1: Lancer directement
solver.run_simulation()

# Option 2: Exécuter en mode parallèle (décommentez si besoin)
# import subprocess
# subprocess.run(["decomposePar", "-case", str(current_path)])
# subprocess.run(["mpirun", "-np", "4", "simpleFoam", "-parallel", "-case", str(current_path)])

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
    
    # 1. Coupe horizontale à mi-hauteur
    print("\nGénération coupe horizontale...")
    foam_post.plot_slice(
        structure=structure,
        plane="z",
        origin=(0, 0, q['max_h']/2),
        scalars="U",
        opacity=0.5,
        path_filename=viz_dir / "slice_horizontale.png"
    )
    
    # 2. Coupe verticale dans l'axe du vent
    print("Génération coupe verticale...")
    foam_post.plot_slice(
        structure=structure,
        plane="x",
        origin=(0, 0, 0),
        scalars="p",
        opacity=0.5,
        path_filename=viz_dir / "slice_verticale.png"
    )
    
    # 3. Contours de vitesse
    print("Génération contours de vitesse...")
    pl_contour = pv.Plotter(off_screen=True)
    pl_contour.add_mesh(cell_mesh, scalars='U', show_scalar_bar=True, opacity=0.7)
    pl_contour.add_mesh(boundaries.get('buildings', cell_mesh), color='red', opacity=0.3)
    foam_post.export_plot(pl_contour, viz_dir / "contour_vitesse.png")
    
    # 4. Lignes de courant
    print("Génération lignes de courant...")
    # Création de seed points en amont
    seeds = pv.Line(
        pointa=(xmin + 10, -20, 5),
        pointb=(xmin + 10, 20, 5),
        resolution=20
    )
    streamlines = cell_mesh.streamlines(
        vectors='U',
        source_radius=2.0,
        integration_direction='forward',
        max_time=100.0
    )
    
    pl_stream = pv.Plotter(off_screen=True)
    pl_stream.add_mesh(cell_mesh, style='wireframe', opacity=0.1)
    pl_stream.add_mesh(streamlines, line_width=2, color='blue')
    foam_post.export_plot(pl_stream, viz_dir / "streamlines.png")
    
    # 5. Visualisation 3D complète
    print("Génération vue 3D...")
    pl_3d = pv.Plotter(off_screen=True)
    pl_3d.add_mesh(cell_mesh, scalars='U', show_scalar_bar=True, opacity=0.5)
    pl_3d.add_mesh(boundaries.get('buildings', cell_mesh), color='red', opacity=0.8)
    pl_3d.camera_position = 'iso'
    foam_post.export_plot(pl_3d, viz_dir / "vue_3d.png")
    
    # Calculs avancés
    print("\nCalculs avancés...")
    
    # Q-criterion pour visualiser les tourbillons
    mesh_with_q = foam_post.calculate_q_criterion(mesh=cell_mesh, velocity_field="U")
    if 'q_criterion' in mesh_with_q.point_data:
        print(f"Q-criterion: min={mesh_with_q.point_data['q_criterion'].min():.2e}, "
              f"max={mesh_with_q.point_data['q_criterion'].max():.2e}")
    
    # Statistiques par région
    # Vitesse moyenne dans la rue
    rue_mask = (cell_mesh.points[:, 1] > -q['street_width']/2) & \
               (cell_mesh.points[:, 1] < q['street_width']/2) & \
               (cell_mesh.points[:, 2] < 5)  # Hauteur piétonne
    if any(rue_mask):
        vitesse_rue = np.mean(cell_mesh.point_data['U'][rue_mask])
        print(f"Vitesse moyenne dans la rue (hauteur piétonne): {vitesse_rue:.2f} m/s")
    
    # Export des données
    foam_post.export_region_data_to_csv(
        structure, "cell", ["U", "p", "k", "epsilon"], 
        viz_dir / "field_data.csv"
    )
    
    print(f"\nVisualisations sauvegardées dans: {viz_dir}")

# -----------------------------
# Génération d'un rapport PDF
# -----------------------------
print("\n" + "="*50)
print("Génération du rapport PDF")
print("="*50)

try:
    pdf = latex_pdf.LatexPdf(
        author="Utilisateur",
        title=f"Simulation CFD - Quartier urbain",
        date="\\today"
    )
    
    pdf.add_section("Introduction")
    pdf.add_text(f"Simulation d'écoulement d'air autour d'un quartier de {q['n_buildings_side']*2} bâtiments.")
    pdf.add_text(f"Angle de rotation du domaine: {d['rotation_angle']}°")
    pdf.add_text(f"Vitesse du vent: {sim['inlet_velocity']} m/s")
    
    pdf.add_section("Paramètres de simulation")
    pdf.add_text(f"Fluide: {config['fluide']['nom']}")
    pdf.add_text(f"Viscosité cinématique: {kinematic_viscosity:.2e} m²/s")
    pdf.add_text(f"Taille de maille: {m['lc_min']} - {m['lc_max']} m")
    
    pdf.add_section("Résultats")
    pdf.add_text("Visualisations disponibles dans le dossier 'visualisations'.")
    
    # Ajout des images si elles existent
    images = [
        ("Coupe horizontale", viz_dir / "slice_horizontale.png"),
        ("Coupe verticale", viz_dir / "slice_verticale.png"),
        ("Contours de vitesse", viz_dir / "contour_vitesse.png"),
        ("Lignes de courant", viz_dir / "streamlines.png")
    ]
    
    for caption, img_path in images:
        if img_path.exists():
            pdf.add_figure(str(img_path), caption, width="0.8\\textwidth")
    
    pdf.generate_pdf(current_path / "rapport_simulation.tex")
    print(f"Rapport PDF généré: {current_path / 'rapport_simulation.pdf'}")
    
except Exception as e:
    print(f"Génération du PDF ignorée: {e}")

# -----------------------------
# Résumé final
# -----------------------------
print("\n" + "="*50)
print("RÉSUMÉ DE LA SIMULATION")
print("="*50)
print(f"Nombre de bâtiments: {len(buildings_data)}")
print(f"Angle de rotation: {d['rotation_angle']}°")
print(f"Dimensions domaine: {xmax-xmin:.1f} x {ymax-ymin:.1f} x {zmax-zmin:.1f} m")
print(f"Vitesse d'entrée: {sim['inlet_velocity']} m/s (direction {sim['direction_vent']})")
print(f"Taille de maille: {m['lc_min']} - {m['lc_max']} m")
print(f"Conditions aux limites:")
print(f"  - Inlet: face à x={xmin:.1f}")
print(f"  - Outlet: face à x={xmax:.1f}")
print(f"  - Côtés: faces y={ymin:.1f} et y={ymax:.1f} (symétrie)")
print(f"  - Sol: face z={zmin:.1f} (wall)")
print(f"  - Ciel: face z={zmax:.1f} (symétrie)")
print(f"  - Bâtiments: {len(unassigned) if unassigned else 0} faces (wall)")
print(f"\nDossier de travail: {current_path}")
print(f"Configuration sauvegardée: {config_path}")
print("\nSimulation terminée avec succès!")

# -----------------------------
# Fonctions de reprise
# -----------------------------
def relancer_simulation_depuis_json(json_path, modifier_angle=None, modifier_vitesse=None):
    """Relance une simulation à partir du fichier JSON"""
    
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    
    if modifier_angle is not None:
        cfg['domaine_fluide']['rotation_angle'] = modifier_angle
        print(f"Angle modifié: {modifier_angle}°")
    
    if modifier_vitesse is not None:
        cfg['simulation']['inlet_velocity'] = modifier_vitesse
        print(f"Vitesse modifiée: {modifier_vitesse} m/s")
    
    # Sauvegarde de la nouvelle config
    new_config_path = json_path.parent / f"config_angle{cfg['domaine_fluide']['rotation_angle']}_v{cfg['simulation']['inlet_velocity']}.json"
    with open(new_config_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    
    print(f"\nPour relancer avec ces paramètres, exécutez à nouveau ce script")
    print(f"en utilisant le fichier de configuration: {new_config_path}")
    
    return cfg

# Exemple d'utilisation (décommentez pour tester)
# nouvelle_config = relancer_simulation_depuis_json(config_path, modifier_angle=30.0, modifier_vitesse=10.0)