from foampilot import prepare_tbad_mesh, AortaSurfaceCleaner, create_closed_aorta_mesh
from pathlib import Path

import kagglehub

import pyvista as pv

import_data = False #set to True to download from Kaggle
#https://www.kaggle.com/datasets/xiaoweixumedicalai/imagetbad?resource=download
if import_data :
    path = kagglehub.dataset_download("xiaoweixumedicalai/imagetbad")

    print("Path to dataset files:", path)
    #il faut dezipper avec 7z dans le path

label_file = Path.cwd().parent / "imageTBAD" 
# for p in label_file.rglob("*label*.nii*"):
#     print(p)
# 1. Conversion NIfTI -> STL
label_file = label_file / "126_label.nii.gz"
stl_temp = "test_126.stl"

prepare_tbad_mesh(label_file, stl_temp)

import trimesh
mesh = trimesh.load("test_126.stl")
print(f"Nombre de sommets: {mesh.vertices.shape[0]}, Nombre de faces: {mesh.faces.shape[0]}")

# 2. Nettoyage
cleaner = AortaSurfaceCleaner(stl_temp)
# Note: Ajustez les points (ex: 50000) selon la complexité de la dissection
results = cleaner.optimize()
best_mesh = results[0]["mesh"]

cleaner.show_best_results(results)
# Activer le mode hors écran si nécessaire (d'après tes préférences)
pv.OFF_SCREEN = True  # ou False pour afficher la fenêtre

# Créer un plotter et afficher le maillage
plotter = pv.Plotter()
plotter.add_mesh(best_mesh, color="lightblue", show_edges=True)
plotter.screenshot("best_mesh_isole.png")
output_path = "aorta_cleaned.stl"
best_mesh.save(output_path)
print(f"Maillage nettoyé sauvegardé sous : {output_path}")

# 3. Fermeture (Capping)
closed_mesh, stats = create_closed_aorta_mesh(
    cleaned_mesh=best_mesh,
    original_mesh=cleaner.original_raw,
    max_planarity_deviation=0.5 # Tolérance en mm (si votre code gère la conversion)
)

# 4. Export final
closed_mesh.save("/kaggle/working/aorta_final_closed.stl")