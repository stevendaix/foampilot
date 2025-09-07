from foampilot import incompressibleFluid,Meshing , commons,utilities,postprocess

import pyvista as pv
from pathlib import Path
from foampilot.utilities.manageunits import Quantity

current_path = Path.cwd() / 'exemple2'

solver = incompressibleFluid(path_case =current_path)
system_dir = solver.system.write()
system_dir = solver.constant.write()

solver.system.fvSchemes.to_dict()

#!/usr/bin/env python
import numpy as np
import classy_blocks as cb

# Paramètres géométriques
pipe_radius = 0.05
muffler_radius = 0.08
ref_length = 0.1

# Taille de cellule constante pour cet exemple
# Cela assure une taille de maille uniforme pour la visualisation.
cell_size = 0.015

# Liste pour stocker les formes géométriques créées
# Les index dans cette liste correspondent aux formes dans le croquis de l'exemple original.
shapes = []

# 0: Création du premier cylindre (tuyau d'entrée)
# Le cylindre est défini par son point de départ, son point d'arrivée et un point définissant son rayon.
shapes.append(cb.Cylinder([0, 0, 0], [3 * ref_length, 0, 0], [0, pipe_radius, 0]))
# Définition du maillage axial (le long de l'axe du cylindre)
shapes[-1].chop_axial(start_size=cell_size)
# Définition du maillage radial (du centre vers l'extérieur)
shapes[-1].chop_radial(start_size=cell_size)
# Définition du maillage tangentiel (autour du cylindre)
shapes[-1].chop_tangential(start_size=cell_size)
# Attribution d'un patch (surface) nommé 'inlet' à la face de départ du cylindre.
shapes[-1].set_start_patch("inlet")

# 1: Chaînage d'un cylindre à la forme précédente
# La méthode 'chain' permet de créer une nouvelle forme qui prolonge la précédente.
shapes.append(cb.Cylinder.chain(shapes[-1], ref_length))
# Maillage axial pour ce nouveau segment de cylindre.
shapes[-1].chop_axial(start_size=cell_size)

# 2: Création d'un anneau extrudé (début du silencieux)
# La méthode 'expand' crée un anneau extrudé en augmentant le rayon de la forme précédente.
shapes.append(cb.ExtrudedRing.expand(shapes[-1], muffler_radius - pipe_radius))
# Maillage radial pour l'anneau extrudé.
shapes[-1].chop_radial(start_size=cell_size)

# 3: Chaînage d'un anneau extrudé (corps du silencieux)
shapes.append(cb.ExtrudedRing.chain(shapes[-1], ref_length))
# Maillage axial pour ce segment d'anneau.
shapes[-1].chop_axial(start_size=cell_size)

# 4: Chaînage d'un autre anneau extrudé (fin du silencieux)
shapes.append(cb.ExtrudedRing.chain(shapes[-1], ref_length))
# Maillage axial pour ce segment d'anneau.
shapes[-1].chop_axial(start_size=cell_size)

# 5: Remplissage de l'anneau extrudé (retour à un cylindre)
# La méthode 'fill' crée un cylindre qui remplit l'espace intérieur de l'anneau extrudé précédent.
shapes.append(cb.Cylinder.fill(shapes[-1]))
# Maillage radial pour le cylindre de remplissage.
shapes[-1].chop_radial(start_size=cell_size)

# 6: Création d'un coude
# Le centre du coude est calculé en fonction de la forme précédente.
elbow_center = shapes[-1].sketch_2.center + np.array([0, 2 * muffler_radius, 0])
# La méthode 'Elbow.chain' crée un coude qui prolonge la forme précédente.
# Les paramètres définissent l'angle de courbure, le centre de rotation, l'axe de rotation et le rayon du coude.
shapes.append(
    cb.Elbow.chain(shapes[-1], np.pi / 2, elbow_center, [0, 0, 1], pipe_radius)
)
# Maillage axial pour le coude.
shapes[-1].chop_axial(start_size=cell_size)
# Attribution d'un patch nommé 'outlet' à la face de fin du coude.
shapes[-1].set_end_patch("outlet")

# Initialisation de l'objet Mesh
# C'est l'objet principal qui va contenir toutes les formes et générer le blockMeshDict.
mesh = cb.Mesh()
# Ajout de toutes les formes créées au maillage.
for shape in shapes:
    mesh.add(shape)

# Définition d'un patch par défaut nommé 'walls' avec le type 'wall'.
# Cela s'applique à toutes les surfaces qui n'ont pas été explicitement définies avec un patch.
mesh.set_default_patch("walls", "wall")

# Écriture des fichiers de sortie
# Le premier argument est le chemin vers le fichier blockMeshDict d'OpenFOAM.
# Le second argument est le chemin vers un fichier VTK de débogage, utile pour la visualisation.
mesh.write(current_path / "system" / "blockMeshDict", current_path /"debug.vtk")

print("Fichiers blockMeshDict et debug.vtk générés avec succès dans le dossier 'case'.")




meshing = Meshing(path_case =current_path)
meshing.run_blockMesh()


# --- 3. Boundary file management ---

# Chemin de base pour le projet OpenFOAM



solver.boundary.initialize_boundary()



# Définir une condition de vitesse d'entrée
solver.boundary.set_velocity_inlet(
    pattern="inlet",
    velocity=(Quantity(10,"m/s"),Quantity(0,"m/s"),Quantity(0,"m/s")),
    turbulence_intensity=0.05  # 5% d'intensité turbulente
)

# Définir une condition de pression de sortie
solver.boundary.set_pressure_outlet(
    pattern="outlet",
    velocity=(Quantity(10,"m/s"),Quantity(0,"m/s"),Quantity(0,"m/s")),
)

# Définir une condition de paroi avec glissement (no slip wall)
solver.boundary.set_wall(
    pattern="walls",velocity=(Quantity(0,"m/s"),Quantity(0,"m/s"),Quantity(0,"m/s"))
)


# Écriture des fichiers de conditions aux limites
fields = ["U", "p", "k", "epsilon","nut"]
for field in fields:
    solver.boundary.write_boundary_file(field)


print(f"Les fichiers de conditions aux limites ont été généré")


solver.run_simulation()

residuals_post = utilities.ResidualsPost(current_path / "log.incompressibleFluid")
residuals_post.process(export_csv=True, export_json=True, export_png=True, export_html=True)



# Créer une instance de post-traitement
foam_post = postprocess.FoamPostProcessing(case_path=current_path)
foam_post.foamToVTK()


# List time steps
time_steps = foam_post.get_all_time_steps()
print(f"Available time steps: {time_steps}")

# Load the structure of the latest time step
if time_steps:
    latest_time_step = time_steps[-1]
    structure = foam_post.load_time_step(latest_time_step)
    cell_mesh = structure["cell"]
    boundaries = structure["boundaries"]
    print(f"Main mesh loaded for time step {latest_time_step}: {cell_mesh}")
    print(f"Boundaries loaded: {list(boundaries.keys())}")

    print("\n--- Testing visualization features (image generation) ---")

    # Test plot_slice
    print("Generating a slice plot...")
    pl_slice = pv.Plotter(off_screen=True)
    y_slice = cell_mesh.slice(normal='z')
    pl_slice.add_mesh(y_slice, scalars='U', lighting=False, scalar_bar_args={'title': 'U'})
    pl_slice.add_mesh(cell_mesh, color='w', opacity=0.25)
    for name, mesh in boundaries.items():
        pl_slice.add_mesh(mesh, opacity=0.5)
    foam_post.export_plot(pl_slice, current_path / "slice_plot.png")

    # Test plot_contour
    print("Generating a contour plot...")
    pl_contour = pv.Plotter(off_screen=True)
    pl_contour.add_mesh(cell_mesh, scalars='p', show_scalar_bar=True)
    foam_post.export_plot(pl_contour, current_path / "contour_plot.png")

    # Test plot_vectors
    print("Generating a vector plot...")
    pl_vectors = pv.Plotter(off_screen=True)
    cell_mesh.set_active_vectors('U')
    arrows = cell_mesh.glyph(orient='U', factor=0.001)
    pl_vectors.add_mesh(arrows, color='blue')
    foam_post.export_plot(pl_vectors, current_path / "vector_plot.png")

    # Test plot_mesh_style
    print("Generating a mesh style plot...")
    pl_mesh_style = pv.Plotter(off_screen=True)
    pl_mesh_style.add_mesh(cell_mesh, style='wireframe', show_edges=True, color='red')
    foam_post.export_plot(pl_mesh_style, current_path / "mesh_style_plot.png")

    print("\n--- Testing analysis features ---")

    # Test calculate_q_criterion
    print("Calculating Q-criterion...")
    mesh_with_q = foam_post.calculate_q_criterion(mesh=cell_mesh, velocity_field="U")
    if 'q_criterion' in mesh_with_q.point_data:
        print(f"Q-criterion calculated. Value range: {mesh_with_q.point_data['q_criterion'].min():.2e} to {mesh_with_q.point_data['q_criterion'].max():.2e}")
    else:
        print("Failed to calculate Q-criterion.")

    # Test calculate_vorticity
    print("Calculating vorticity...")
    mesh_with_vorticity = foam_post.calculate_vorticity(mesh=cell_mesh, velocity_field="U")
    if 'vorticity' in mesh_with_vorticity.point_data:
        print(f"Vorticity calculated. Value range: {mesh_with_vorticity.point_data['vorticity'].min():.2e} to {mesh_with_vorticity.point_data['vorticity'].max():.2e}")
    else:
        print("Failed to calculate vorticity.")

    print("\n--- Testing statistical analysis features ---")

    # Test get_scalar_statistics
    print("Calculating statistics for pressure field 'p'...")
    pressure_stats = foam_post.get_scalar_statistics(mesh=cell_mesh, scalar_field="p")
    print(f"Pressure statistics: {pressure_stats}")

    # Test get_time_series_data
    print("Extracting time series for pressure field 'p' at a given point...")
    point_to_probe = [0.0, 0.0, 0.0]
    time_series = foam_post.get_time_series_data(scalar_field="p", point_coordinates=point_to_probe)
    print(f"Pressure time series at point {point_to_probe}: {time_series['data']}")
    print(f"Corresponding time steps: {time_series['time_steps']}")

    # Test get_mesh_statistics
    print("Calculating mesh statistics...")
    mesh_stats = foam_post.get_mesh_statistics(cell_mesh)
    print(f"Mesh statistics: {mesh_stats}")

    # Test get_region_statistics (cell)
    print("Calculating statistics for 'cell' region and 'U' field...")
    cell_region_stats = foam_post.get_region_statistics(structure, "cell", "U")
    print(f"'Cell' region statistics for 'U': {cell_region_stats}")

    # Test get_region_statistics (boundary)
    if "boundary1" in boundaries:
        print("Calculating statistics for 'boundary1' region and 'p' field...")
        boundary_region_stats = foam_post.get_region_statistics(structure, "boundary1", "p")
        print(f"'Boundary1' region statistics for 'p': {boundary_region_stats}")

    # Test export_region_data_to_csv
    print("Exporting 'cell' region data to CSV file...")
    foam_post.export_region_data_to_csv(structure, "cell", ["U", "p"], current_path / "cell_data.csv")

    # Test export_statistics_to_json
    print("Exporting statistics to JSON file...")
    all_stats = {
        "mesh_stats": mesh_stats,
        "cell_region_stats_U": cell_region_stats,
        "boundary1_region_stats_p": boundary_region_stats if "boundary1" in boundaries else "N/A"
    }
    foam_post.export_statistics_to_json(all_stats, current_path / "all_stats.json")

    # Test create_animation
    print("Creating an animation...")
    foam_post.create_animation(scalars='U', filename='animation_test.gif', fps=5)
else:
    print("No time steps found, unable to test the class.")

print("\nTest completed.")
