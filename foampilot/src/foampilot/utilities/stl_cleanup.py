import pyvista as pv
import numpy as np
import gmsh
import os
from scipy.spatial import KDTree

class AortaVesselMaster:
def init(self, verbose=True):
self.verbose = verbose

def log(self, msg):  
    if self.verbose: print(f"[PROCESS] {msg}")  

def clean_and_prepare_surface(self, input_file, target_points=25000):  
    """Nettoyage, lissage Taubin et remaillage isotrope."""  
    mesh = pv.read(input_file).extract_geometry()  
      
    # 1. Isolation de l'aorte (plus grande région)  
    self.log("Cleaning topology...")  
    cleaned = mesh.connectivity(largest=True).clean(tolerance=1e-5)  
      
    # 2. Détection des zones à protéger (Inlets/Outlets)  
    # On utilise les bords libres du STL  
    boundary_edges = cleaned.extract_feature_edges(boundary_edges=True, feature_edges=False)  
    protected_coords = boundary_edges.points  
      
    # 3. Lissage Taubin (ne réduit pas le diamètre)  
    self.log("Smoothing (Taubin)...")  
    smoothed = cleaned.smooth_taubin(n_iter=40, pass_band=0.1)  
      
    # 4. Remaillage Isotrope (PyACVD)  
    self.log("Isotropic Remeshing...")  
    import pyacvd  
    clus = pyacvd.Clustering(smoothed)  
    clus.subdivide(3)  
    clus.cluster(target_points)  
    remeshed = clus.create_mesh()  
      
    # 5. Correction de la déformation des bords via KDTree  
    self.log("Restoring boundary precision...")  
    tree = KDTree(remeshed.points)  
    _, indices = tree.query(protected_coords)  
    remeshed.points[indices] = protected_coords  
      
    return remeshed  

def create_volume_mesh(self, surface_mesh, output_file, mesh_size=2.0):  
    """Génère un maillage volumique (tétraèdres) via GMSH."""  
    self.log("Starting GMSH Volume Meshing...")  
      
    gmsh.initialize()  
    gmsh.model.add("AortaVolume")  

    # Import de la surface STL nettoyée  
    # On passe par un fichier temporaire pour assurer la compatibilité GMSH  
    temp_stl = "temp_refined.stl"  
    surface_mesh.save(temp_stl)  
    gmsh.merge(temp_stl)  

    # Création de la géométrie volumique  
    surf_ids = gmsh.model.getEntities(2)  
    loops = gmsh.model.geo.addSurfaceLoop([s[1] for s in surf_ids])  
    gmsh.model.geo.addVolume([loops])  
    gmsh.model.geo.synchronize()  

    # Configuration du mailleur  
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.5)  
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)  
    gmsh.option.setNumber("Mesh.Algorithm3D", 1) # Delaunay  
      
    gmsh.model.mesh.generate(3)  
    gmsh.write(output_file)  
    gmsh.finalize()  
      
    if os.path.exists(temp_stl): os.remove(temp_stl)  
    self.log(f"Volume mesh saved to {output_file}")  

def visualize(self, surface, volume_path):  
    """Visualise la surface et une coupe du volume."""  
    vol = pv.read(volume_path)  
    plotter = pv.Plotter(shape=(1, 2))  
      
    plotter.subplot(0, 0)  
    plotter.add_text("Surface Isotrope")  
    plotter.add_mesh(surface, color="lightblue", show_edges=True)  
      
    plotter.subplot(0, 1)  
    plotter.add_text("Coupe Volumique (Tétraèdres)")  
    # Création d'une coupe (clip) pour voir l'intérieur  
    clipped = vol.clip(normal='x')  
    plotter.add_mesh(clipped, show_edges=True, color="white")  
      
    plotter.link_views()  
    plotter.show()

--- MAIN ---

if name == "main":
worker = AortaVesselMaster()

# 1. Prétraitement de la surface  
surf_cleaned = worker.clean_and_prepare_surface("votre_aorte.stl")  
  
# 2. Génération du volume (Fichier .msh ou .vtk)  
# Le format .vtk est facile à relire dans PyVista/Paraview  
worker.create_volume_mesh(surf_cleaned, "aorta_final.vtk", mesh_size=1.5)  
  
# 3. Affichage  
worker.visualize(surf_cleaned, "aorta_final.vtk")