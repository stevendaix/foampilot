from foampilot import incompressibleFluid,Meshing , commons,utilities,postprocess

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

import pyvista as pv
from pathlib import Path

cwd = Path.cwd()



#liste des fichiers vtk avec des numéros
list_time_step = [0,75]

# # paramètrer par défaut le dernier pas 
# structure = {
#     'cell': pv.read(cwd / "exemple2"/ "VTK"/"exemple2_75.vtk"),
#     'boundaries': {
#         "inlet" : pv.read(cwd / "exemple2"/ "VTK"/"inlet"/"inlet_75.vtk"),
#         "outlet" : pv.read(cwd / "exemple2"/ "VTK"/"outlet"/"outlet_75.vtk"),
#         "walls" : pv.read(cwd / "exemple2"/ "VTK"/"walls"/"walls_75.vtk")}
#     }

# # generate a slice in the XZ plane
# y_slice = structure["cell"].slice('z')

# pl = pv.Plotter()
# pl.add_mesh(y_slice, scalars='U', lighting=False, scalar_bar_args={'title': 'Flow Velocity'})
# pl.add_mesh(structure["cell"], color='w', opacity=0.25)
# pl.enable_anti_aliasing()
# pl.show()