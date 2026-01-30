import trimesh
import pyvista as pv
import pyacvd
import numpy as np
import os
from scipy.spatial import KDTree

class AortaSurfaceCleaner:
    def __init__(self, input_path):
        self.path = input_path
        self.mesh = None
        self.original_raw = pv.read(input_path)
        
        # Détection d'échelle auto au chargement
        if self.original_raw.length > 1.0:
            self.original_raw.scale([0.001, 0.001, 0.001], inplace=True)

    def log(self, msg):
        print(f"[PROCESS] {msg}")

    def run_pipeline(self, method="classic", decimate_ratio=0.5, target_points=25000):
        """
        Interrupteur de méthode :
        - 'classic' : Trimesh + Taubin + ACVD (Précision CFD)
        - 'wrap'    : Reconstruct Surface (Robustesse max pour scans brisés)
        """
        if method == "classic":
            return self._pipeline_classic(decimate_ratio, target_points)
        elif method == "wrap":
            return self._pipeline_wrapping(target_points)
        else:
            raise ValueError("Méthode inconnue. Utilisez 'classic' ou 'wrap'.")

    def _pipeline_classic(self, decimate_ratio, target_points):
        self.log("Exécution du pipeline CLASSIQUE...")
        # 1. Trimesh : Réparation et isolation
        t_mesh = trimesh.load(self.path)
        t_mesh.fill_holes()
        components = t_mesh.split(only_watertight=False)
        t_mesh = max(components, key=lambda x: x.area)
        
        # 2. PyVista : Décimation et Protection des bords
        pv_mesh = pv.wrap(t_mesh)
        if pv_mesh.length > 1.0: pv_mesh.scale([0.001, 0.001, 0.001], inplace=True)
        
        if decimate_ratio < 1.0:
            pv_mesh = pv_mesh.decimate(1.0 - decimate_ratio, preserve_topology=True)

        edges = pv_mesh.extract_feature_edges(boundary_edges=True, feature_edges=False)
        protected_coords = edges.points

        # 3. Lissage et Remaillage
        smoothed = pv_mesh.smooth_taubin(n_iter=50, pass_band=0.05)
        clus = pyacvd.Clustering(smoothed)
        clus.subdivide(3)
        clus.cluster(target_points)
        self.mesh = clus.create_mesh()

        # 4. Restauration KDTree
        if len(protected_coords) > 0:
            tree = KDTree(self.mesh.points)
            _, indices = tree.query(protected_coords)
            self.mesh.points[indices] = protected_coords
        
        return self.mesh

    def _pipeline_wrapping(self, target_points):
        self.log("Exécution du pipeline WRAPPING (Shrink-wrap)...")
        # Utilisation de la reconstruction de surface VTK
        # On traite le scan comme un nuage de points pour recréer une peau neuve
        points = pv.PolyData(self.original_raw.points)
        surface = points.reconstruct_surface(nbr_sz=20)
        
        # Extraction de la peau et remaillage pour uniformiser
        clus = pyacvd.Clustering(surface.connectivity(largest=True))
        clus.subdivide(2)
        clus.cluster(target_points)
        self.mesh = clus.create_mesh().smooth_taubin(n_iter=30)
        
        return self.mesh

    # --- OUTILS DE DIAGNOSTIC ---

    def show_comparison(self):
        p = pv.Plotter(shape=(1, 2), title="Diagnostic : Original vs Cleaned")
        p.subplot(0, 0)
        p.add_text("Scan Brut")
        p.add_mesh(self.original_raw, color="salmon", opacity=0.5)
        p.subplot(0, 1)
        p.add_text("Maillage Final")
        p.add_mesh(self.mesh, color="lightblue", show_edges=True)
        p.link_views()
        p.show()

    def check_quality(self):
        """Affiche la qualité des triangles (angles)."""
        qual = self.mesh.compute_cell_quality(quality_measure='min_angle')
        qual.plot(scalars="CellQuality", cmap="RdYlGn", show_edges=True, title="Qualité (Angle Min)")

    def save(self, output_path):
        if self.mesh:
            self.mesh.save(output_path)
            self.log(f"Exportation réussie : {output_path}")

# --- EXEMPLE D'UTILISATION ---
if __name__ == "__main__":
    cleaner = AortaSurfaceCleaner("aorta_scan.stl")

    # Si le scan est vraiment sale (trous partout), utilisez method="wrap"
    # Sinon, "classic" est meilleur pour garder la précision
    mesh_final = cleaner.run_pipeline(method="classic", decimate_ratio=0.3)

    cleaner.show_comparison()
    cleaner.check_quality()
    cleaner.save("aorta_for_gmsh.stl")


