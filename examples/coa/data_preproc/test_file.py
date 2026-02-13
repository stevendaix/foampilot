#!/usr/bin/env python3
"""
TEST AVANT/APRÈS - Utilisation des fonctions de extract_and_visualize_tbad.py
==============================================================================
À MODIFIER: juste le chemin du fichier en ligne 13
"""

import sys
from pathlib import Path

import pyvista as pv
import nibabel as nib
import numpy as np
import trimesh

# Ajout du répertoire courant au path
sys.path.insert(0, str(Path(__file__).parent))

# Import depuis extract_and_visualize_tbad
from extract_and_visualize_tbad import TbadExtractor

# ============================================================================
# À MODIFIER - METTEZ VOTRE CHEMIN ICI
# ============================================================================
MON_FICHIER = "votre_fichier.nii.gz"  # ← CHANGEZ CETTE LIGNE

# ============================================================================
# EXÉCUTION
# ============================================================================

print("\n" + "="*60)
print("🧪 TEST AVANT/APRÈS AVEC extract_and_visualize_tbad.py")
print("="*60)

# Vérification fichier
if not Path(MON_FICHIER).exists():
    print(f"\n❌ Fichier non trouvé: {MON_FICHIER}")
    print("\nFichiers disponibles:")
    for f in Path(".").glob("*.nii*"):
        print(f"  - {f.name}")
    sys.exit(1)

# ========================================================================
# 1. AVANT - Visualisation du masque NIfTI
# ========================================================================
print("\n📊 AVANT - Chargement du masque NIfTI...")

# Charger le NIfTI
img = nib.load(MON_FICHIER)
data = img.get_fdata()
spacing = img.header.get_zooms()[:3]

print(f"   Dimensions: {data.shape}")
print(f"   Spacing: {spacing[0]:.2f} x {spacing[1]:.2f} x {spacing[2]:.2f} mm")

# Créer une visualisation 3D du masque (isosurface)
print("\n🖼️  Affichage AVANT - Masque 3D (isosurface)...")
print("   (Fermez la fenêtre pour continuer)")

# Créer un maillage du masque pour visualisation
from skimage.measure import marching_cubes
try:
    verts, faces, _, _ = marching_cubes(data > 0.5, level=0.5, spacing=spacing)
    mesh_avant = trimesh.Trimesh(vertices=verts, faces=faces)
    pv_avant = pv.wrap(mesh_avant)
    pv_avant.points *= 0.001  # mm → m
    
    plotter_avant = pv.Plotter(title="AVANT - Masque NIfTI (isosurface)", window_size=[1000, 800])
    plotter_avant.add_mesh(pv_avant, color='red', opacity=0.7, smooth_shading=True, label='Masque')
    plotter_avant.add_text(f"AVANT CONVERSION\n{Path(MON_FICHIER).name}", font_size=14, position='upper_edge')
    plotter_avant.add_axes()
    plotter_avant.add_legend()
    plotter_avant.show()
except:
    # Si marching cubes échoue, afficher une coupe 2D
    print("   Affichage d'une coupe 2D à la place...")
    slice_idx = data.shape[2] // 2
    slice_data = data[:, :, slice_idx]
    
    plotter_avant = pv.Plotter(title="AVANT - Coupe du masque", window_size=[1000, 800])
    grid = pv.UniformGrid()
    grid.dimensions = data.shape
    grid.spacing = spacing
    grid.origin = (0, 0, 0)
    grid.point_data["values"] = data.flatten(order="F")
    
    plotter_avant.add_mesh(grid.slice(normal=[0,0,1], origin=[0,0,slice_idx*spacing[2]]), 
                          cmap='Reds', show_scalar_bar=False)
    plotter_avant.add_text(f"AVANT - Coupe z={slice_idx}", font_size=14, position='upper_edge')
    plotter_avant.show()

# ========================================================================
# 2. EXTRACTION avec TbadExtractor
# ========================================================================
print("\n🔄 Extraction STL avec TbadExtractor...")

extractor = TbadExtractor(
    output_dir="test_output",
    refine_factor=2,
    sdf_sigma=0.25,
    target_triangles_tl=100000,
    target_triangles_fl=80000,
    extract_patches=True,
    verbose=True
)

results = extractor.extract(
    nifti_path=MON_FICHIER,
    tl_label=1,
    fl_label=2
)

# ========================================================================
# 3. APRÈS - Visualisation des STL générés
# ========================================================================
print("\n📊 APRÈS - Chargement des STL...")

# Charger les STL
mesh_tl = trimesh.load(results['tl']['walls'])
mesh_fl = trimesh.load(results['fl']['walls'])

print(f"\n   True Lumen: {len(mesh_tl.faces):,} faces")
print(f"   False Lumen: {len(mesh_fl.faces):,} faces")

# Visualisation APRÈS
print("\n🖼️  Affichage APRÈS - STL générés...")
print("   (Fermez la fenêtre pour terminer)")

pv_tl = pv.wrap(mesh_tl)
pv_fl = pv.wrap(mesh_fl)
pv_tl.points *= 0.001  # mm → m
pv_fl.points *= 0.001

plotter_apres = pv.Plotter(title="APRÈS - STL extraits", window_size=[1200, 800])

# TL en bleu
plotter_apres.add_mesh(pv_tl, color='lightblue', opacity=0.7, 
                       smooth_shading=True, label=f'True Lumen ({len(mesh_tl.faces):,} faces)')

# FL en rouge
plotter_apres.add_mesh(pv_fl, color='salmon', opacity=0.5, 
                       smooth_shading=True, label=f'False Lumen ({len(mesh_fl.faces):,} faces)')

# Patches si existent
if results['tl']['patches']:
    for i, patch_path in enumerate(results['tl']['patches']):
        patch = trimesh.load(patch_path)
        pv_patch = pv.wrap(patch)
        pv_patch.points *= 0.001
        plotter_apres.add_mesh(pv_patch, color='green', opacity=0.9, 
                               show_edges=True, label=f'TL Patch {i}')

if results['fl']['patches']:
    for i, patch_path in enumerate(results['fl']['patches']):
        patch = trimesh.load(patch_path)
        pv_patch = pv.wrap(patch)
        pv_patch.points *= 0.001
        plotter_apres.add_mesh(pv_patch, color='yellow', opacity=0.9, 
                               show_edges=True, label=f'FL Patch {i}')

plotter_apres.add_text("APRÈS CONVERSION - STL", font_size=14, position='upper_edge')
plotter_apres.add_axes()
plotter_apres.add_legend()
plotter_apres.show()

# ========================================================================
# 4. COMPARAISON CÔTE À CÔTE
# ========================================================================
print("\n🖼️  Affichage COMPARAISON avant/après...")
print("   (Fermez la fenêtre pour terminer)")

plotter_compare = pv.Plotter(shape=(1, 2), window_size=[1600, 800], 
                             title="Comparaison Avant/Après")

# Sous-plot 1: Avant (masque)
plotter_compare.subplot(0, 0)
try:
    plotter_compare.add_mesh(pv_avant, color='red', opacity=0.7, smooth_shading=True)
except:
    # Si pv_avant n'existe pas, utiliser la coupe
    pass
plotter_compare.add_text("AVANT - Masque NIfTI", font_size=12)

# Sous-plot 2: Après (STL)
plotter_compare.subplot(0, 1)
plotter_compare.add_mesh(pv_tl, color='lightblue', opacity=0.7, smooth_shading=True)
plotter_compare.add_mesh(pv_fl, color='salmon', opacity=0.5, smooth_shading=True)
plotter_compare.add_text("APRÈS - STL", font_size=12)

plotter_compare.link_views()
plotter_compare.show()

print("\n✅ Test terminé")
print(f"📁 Résultats dans: test_output/")
print(f"   - TL: {Path(results['tl']['walls']).name}")
print(f"   - FL: {Path(results['fl']['walls']).name}")