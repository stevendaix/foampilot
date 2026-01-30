import trimesh
import pyvista as pv
import pyacvd
import numpy as np
import itertools
from pathlib import Path
from scipy.spatial import cKDTree, KDTree

class AortaSurfaceCleaner:
    def __init__(self, input_path):
        self.path = input_path
        self.mesh = None
        # Chargement et normalisation immédiate de l'original
        self.original_raw = pv.read(input_path)
        if self.original_raw.length > 1.0:
            self.original_raw.scale([0.001, 0.001, 0.001], inplace=True)

    def log(self, msg):
        print(f"[PROCESS] {msg}")

    # --- MÉTHODES DE CALCUL ---

    def compute_hausdorff(self, cleaned_mesh):
        """Calcule l'écart entre l'original et le maillage nettoyé."""
        tree = cKDTree(self.original_raw.points)
        distances, _ = tree.query(cleaned_mesh.points)
        return {
            "max": np.max(distances),
            "mean": np.mean(distances),
            "local_dist": distances
        }

    def get_mesh_quality(self, mesh):
        """Retourne l'angle minimal moyen des triangles."""
        qual = mesh.compute_cell_quality(quality_measure='min_angle')
        return np.mean(qual['CellQuality'])

    # --- PIPELINES ---

    def _pipeline_classic(self, decimate_ratio, target_points, smooth_iter=50):
        t_mesh = trimesh.load(self.path)
        t_mesh.fill_holes()
        t_mesh = max(t_mesh.split(only_watertight=False), key=lambda x: x.area)
        
        pv_mesh = pv.wrap(t_mesh)
        if pv_mesh.length > 1.0: pv_mesh.scale([0.001, 0.001, 0.001], inplace=True)
        if decimate_ratio < 1.0:
            pv_mesh = pv_mesh.decimate(1.0 - decimate_ratio, preserve_topology=True)

        edges = pv_mesh.extract_feature_edges(boundary_edges=True, feature_edges=False)
        smoothed = pv_mesh.smooth_taubin(n_iter=smooth_iter, pass_band=0.05)
        
        clus = pyacvd.Clustering(smoothed)
        clus.subdivide(3)
        clus.cluster(target_points)
        m = clus.create_mesh()

        if edges.n_points > 0:
            tree = KDTree(m.points)
            _, idx = tree.query(edges.points)
            m.points[idx] = edges.points
        return m

    def _pipeline_wrapping(self, target_points, nbr_sz=20):
        points = pv.PolyData(self.original_raw.points)
        surface = points.reconstruct_surface(nbr_sz=nbr_sz)
        clus = pyacvd.Clustering(surface.connectivity(largest=True))
        clus.subdivide(2)
        clus.cluster(target_points)
        return clus.create_mesh().smooth_taubin(n_iter=30)

    # --- L'OPTIMISEUR ---

    def optimize(self, runs=None):
        """
        Teste plusieurs combinaisons et retourne le meilleur compromis.
        """
        if runs is None:
            runs = [
                {"method": "classic", "decimate": 0.2, "points": 30000},
                {"method": "classic", "decimate": 0.5, "points": 20000},
                {"method": "wrap",    "decimate": 0.0, "points": 25000}
            ]
        
        results = []
        for r in runs:
            self.log(f"Testing {r['method']} | Pts: {r['points']}...")
            try:
                if r['method'] == "classic":
                    m = self._pipeline_classic(r['decimate'], r['points'])
                else:
                    m = self._pipeline_wrapping(r['points'])
                
                h = self.compute_hausdorff(m)
                q = self.get_mesh_quality(m)
                
                # Score : Haute qualité d'angle + Faible distance de Hausdorff
                # On pénalise si l'erreur max > 0.8mm (0.0008m)
                score = (q / 60.0) + (1.0 / (1.0 + h['max'] * 1500))
                if m.is_manifold: score += 0.5
                
                results.append({"mesh": m, "score": score, "h_max": h['max'], "quality": q, "params": r})
            except Exception as e:
                self.log(f"Failed combination: {e}")

        # On garde le meilleur
        results.sort(key=lambda x: x['score'], reverse=True)
        self.mesh = results[0]['mesh']
        self.log(f"Best model selected (Score: {results[0]['score']:.2f})")
        return results

    def show_best_results(self, results):
        """Visualise les 3 meilleurs avec l'erreur de Hausdorff en couleur."""
        p = pv.Plotter(shape=(1, min(3, len(results))))
        for i in range(min(3, len(results))):
            res = results[i]
            p.subplot(0, i)
            h = self.compute_hausdorff(res['mesh'])
            res['mesh']["Error_mm"] = h['local_dist'] * 1000 # Conversion en mm pour le plot
            p.add_mesh(res['mesh'], scalars="Error_mm", cmap="magma")
            p.add_text(f"Rank {i+1}: {res['params']['method']}\nMax Err: {h['max']*1000:.2f}mm", font_size=9)
        p.show()
