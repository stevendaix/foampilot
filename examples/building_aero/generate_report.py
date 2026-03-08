#!/usr/bin/env python
"""
Génère un rapport PDF de la simulation d'aérauliqueurbaine.
Utilise typst pour créer un document scientifique avec:
- Configuration de la simulation
- Présentation du maillage
- Résultats numériques (résidus)
- Visualisations
"""

from pathlib import Path
import json

# Add foampilot to path
import sys
sys.path.insert(0, '/home/steven/foampilot')

from foampilot.report.typst_pdf import ScientificDocument, TypstRenderer
from foampilot.utilities import ResidualsPost

# Base path
base_path = Path(__file__).parent

# Path to simulation case
current_path = base_path / 'quartier_gmsh'
viz_dir = current_path / 'visualisations'

# Configuration from JSON
config_path = current_path / 'buildings_config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# Create scientific document
doc = ScientificDocument(
    title="Simulation CFD - Aéraulique Urbaine",
    author="Foampilot Automated Report"
)

# =========================================
# 1. TITRE ET RÉSUMÉ
# =========================================
doc.add_section("Simulation d'Aéraulique Urbaine", "", level=1)

# Abstract
abstract = f"""
Cette étude présente une simulation CFD de l'écoulement de l'air autour d'un quartier urbain
composé de {config['quartier']['n_buildings_side']*2} bâtiments. Le domaine fluide estincliné à {config['domaine_fluide']['rotation_angle']}° 
pour simuler un vent provenant d'une direction non-parallèle aux rues. La turbulence 
est modélisée avec le modèle k-epsilon.
"""
doc.add_section("Résumé", abstract)

# =========================================
# 2. CONDITIONS DE SIMULATION
# =========================================
doc.add_section("Conditions de Simulation", "", level=1)

# Geometry parameters
q = config['quartier']
d = config['domaine_fluide']
m = config['maillage']
sim = config['simulation']

geometry_data = [
    ["Paramètre", "Valeur"],
    ["Largeur du lot", f"{q['lot_width']} m"],
    ["Longueur du lot", f"{q['lot_length']} m"],
    ["Largeur de rue", f"{q['street_width']} m"],
    ["Hauteur min bâtiments", f"{q['min_h']} m"],
    ["Hauteur max bâtiments", f"{q['max_h']} m"],
    ["Profondeur bâtiments", f"{q['building_depth']} m"],
    ["Écart entre bâtiments", f"{q['gap']} m"],
    ["Nb bâtiments par côté", str(q['n_buildings_side'])],
]
doc.add_table(
    geometry_data,
    headers=["Paramètre", "Valeur"],
    caption="Paramètres géométriques du quartier",
    label="tab:geometry"
)

# Fluid domain
doc.add_section("Domaine Fluide", "", level=2)
domain_data = [
    ["Paramètre", "Valeur"],
    ["Multiplicateur amont", str(d['mx_in'])],
    ["Multiplicateur aval", str(d['mx_out'])],
    ["Multiplicateur latéral", str(d['my'])],
    ["Multiplicateur vertical", str(d['mz'])],
    ["Angle de rotation", f"{d['rotation_angle']}°"],
]
doc.add_table(
    domain_data,
    headers=["Paramètre", "Valeur"],
    caption="Paramètres du domaine fluide",
    label="tab:domain"
)

# Mesh parameters
doc.add_section("Paramètres du Maillage", "", level=2)
mesh_data = [
    ["Paramètre", "Valeur"],
    ["Taille min éléments", f"{m['lc_min']} m"],
    ["Taille max éléments", f"{m['lc_max']} m"],
    ["Facteur raffinement", str(m['raffinement_batiments'])],
]
doc.add_table(
    mesh_data,
    headers=["Paramètre", "Valeur"],
    caption="Paramètres de maillage",
    label="tab:mesh"
)

# Simulation parameters
doc.add_section("Conditions aux Limites et Paramètres de Simulation", "", level=2)
sim_data = [
    ["Paramètre", "Valeur"],
    ["Fluide", config['fluide']['nom']],
    ["Température", f"{config['fluide']['temperature']} K"],
    ["Pression", f"{config['fluide']['pression']} Pa"],
    ["Vitesse inlet", f"{sim['inlet_velocity']} m/s"],
    ["Intensité turbulence", f"{sim['turbulence_intensity']*100}%"],
    ["Direction vent", f"({sim['direction_vent'][0]}, {sim['direction_vent'][1]}, {sim['direction_vent'][2]})"],
]
doc.add_table(
    sim_data,
    headers=["Paramètre", "Valeur"],
    caption="Paramètres de simulation",
    label="tab:sim"
)

# =========================================
# 3. MAILLAGE
# =========================================
doc.add_section("Analyse du Maillage", "", level=1)

# Try to add mesh quality stats if available
try:
    from foampilot import postprocess
    foam_post = postprocess.FoamPostProcessing(case_path=current_path)
    # Get mesh from last time step
    time_steps = foam_post.get_all_time_steps()
    if time_steps:
        structure = foam_post.load_time_step(time_steps[-1])
        cell_mesh = structure["cell"]
        
        mesh_stats_data = [
            ["Métrique", "Valeur"],
            ["Nombre de cellules", str(cell_mesh.n_cells)],
            ["Nombre de points", str(cell_mesh.n_points)],
        ]
        
        # Get boundaries
        boundaries = structure["boundaries"]
        mesh_stats_data.append(["Nombre de frontières", str(len(boundaries))])
        
        doc.add_table(
            mesh_stats_data,
            headers=["Métrique", "Valeur"],
            caption="Statistiques du maillage",
            label="tab:mesh_stats"
        )
except Exception as e:
    print(f"Note: impossible d'obtenir les stats du maillage: {e}")

# =========================================
# 4. RÉSULTATS NUMÉRIQUES
# =========================================
doc.add_section("Résultats Numériques", "", level=1)

# Residuals
doc.add_section("Analyse des Résidus", "", level=2)

log_file = current_path / "log.incompressibleFluid"
if log_file.exists():
    residuals_post = ResidualsPost(log_file)
    residuals_post.process(export_json=True)
    
    # Load residuals JSON
    residuals_json = current_path / "residuals" / "log_residuals.json"
    if residuals_json.exists():
        with open(residuals_json, 'r') as f:
            residuals = json.load(f)
        
        # Get final residuals
        if 'final' in residuals:
            final = residuals['final']
            residual_data = [
                ["Champ", "Résidu final"],
            ]
            for field, value in final.items():
                residual_data.append([field, f"{value:.6e}"])
            
            doc.add_table(
                residual_data,
                headers=["Champ", "Résidu final"],
                caption="Résidus finaux",
                label="tab:residuals"
            )

# =========================================
# 5. VISUALISATIONS
# =========================================
doc.add_section("Visualisations", "", level=1)

# Horizontal slices
doc.add_section("Coupes Horizontales", "", level=2)
for z_val in [0, 15, 20, 30]:
    slice_file = viz_dir / f"slice_z{z_val}.png"
    if slice_file.exists():
        doc.add_figure(
            f"slice_z{z_val}.png",
            caption=f"Coupe horizontale à z = {z_val} m - Champ de vitesse",
            label=f"fig:slice_z{z_val}",
            width="80%"
        )

# Vertical slices
doc.add_section("Coupes Verticales", "", level=2)
slice_x = viz_dir / "slice_x.png"
if slice_x.exists():
    doc.add_figure(
        "slice_x.png",
        caption="Coupe verticale dans la direction X",
        label="fig:slice_x",
        width="80%"
    )

slice_y = viz_dir / "slice_y.png"
if slice_y.exists():
    doc.add_figure(
        "slice_y.png",
        caption="Coupe verticale dans la direction Y",
        label="fig:slice_y",
        width="80%"
    )

# Contours
doc.add_section("Isocontours de Pression", "", level=2)
contour = viz_dir / "contour_pression.png"
if contour.exists():
    doc.add_figure(
        "contour_pression.png",
        caption="Isocontours de pression",
        label="fig:contour_p",
        width="80%"
    )

# Vectors
doc.add_section("Champ de Vecteurs Vitesse", "", level=2)
vectors = viz_dir / "vector_plot.png"
if vectors.exists():
    doc.add_figure(
        "vector_plot.png",
        caption="Vecteurs vitesse",
        label="fig:vectors",
        width="80%"
    )

# =========================================
# 6. COMPILATION PDF
# =========================================
print("Génération du rapport PDF...")

# Create report directory in building_aero folder
report_dir = base_path / 'report'
report_dir.mkdir(exist_ok=True)

# Copy images to report folder
import shutil
for img in ['slice_z0.png', 'slice_z15.png', 'slice_z20.png', 'slice_z30.png', 'slice_x.png', 'slice_y.png', 'contour_pression.png', 'vector_plot.png']:
    src = viz_dir / img
    if src.exists():
        shutil.copy(src, report_dir / img)

# Copy residuals image
residuals_src = current_path / 'residuals' / 'log_residuals.png'
if residuals_src.exists():
    shutil.copy(residuals_src, report_dir / 'residuals.png')

renderer = TypstRenderer()
renderer.compile_pdf(
    doc, 
    output_pdf=str(report_dir / "rapport_simulation.pdf")
)

print(f"Rapport généré: {report_dir / 'rapport_simulation.pdf'}")
