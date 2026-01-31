import trimesh
import pyvista as pv
import pyacvd
import numpy as np
import itertools
from pathlib import Path
from scipy.spatial import cKDTree, KDTree
import logging
from typing import List, Optional, Tuple, Dict, Union
from enum import Enum

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
        qual = mesh.cell_quality(quality_measure='min_angle')
        print(f"Type de 'qual': {type(qual)}")  # Debug
        if isinstance(qual, np.ndarray):
            return np.mean(qual)
        else:
            # Alternative : Calculer manuellement la qualité
            cells = mesh.cells
            points = mesh.points
            angles = []
            for cell in cells:
                # Exemple simplifié : calculer les angles d'un triangle
                if len(cell) == 3:  # Triangle
                    a, b, c = points[cell]
                    # Calculer les angles (à adapter selon tes besoins)
                    ab = b - a
                    ac = c - a
                    angle = np.arccos(np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac)))
                    angles.append(np.degrees(angle))
            return np.mean(angles) if angles else 0



        # --- PIPELINES ---

    def _pipeline_classic(self, decimate_ratio, target_points, smooth_iter=50):
        try:
            # Charger et nettoyer le maillage
            t_mesh = trimesh.load(self.path)
            t_mesh.fill_holes()
            t_mesh = max(t_mesh.split(only_watertight=False), key=lambda x: x.area)

            # Convertir en PolyData
            pv_mesh = pv.wrap(t_mesh)
            if pv_mesh.length > 1.0:
                pv_mesh.scale([0.001, 0.001, 0.001], inplace=True)

            # Décimation
            if decimate_ratio < 1.0:
                pv_mesh = pv_mesh.decimate_pro(reduction=decimate_ratio)

            # Lissage
            smoothed = pv_mesh.smooth_taubin(n_iter=smooth_iter, pass_band=0.05)

            # Clustering
            clus = pyacvd.Clustering(smoothed)
            clus.subdivide(3)
            clus.cluster(target_points)
            m = clus.create_mesh()

            # Extraction des bords
            edges = pv_mesh.extract_feature_edges(boundary_edges=True, feature_edges=False)
            if edges.n_points > 0:
                tree = KDTree(m.points)
                _, idx = tree.query(edges.points)
                m.points[idx] = edges.points

            return m

        except Exception as e:
            self.log(f"Erreur dans _pipeline_classic: {e}")
            raise  # Relance l'erreur pour qu'elle soit gérée dans `optimize`


    def _pipeline_wrapping(self, target_points, nbr_sz=20):
        try:
            points = pv.PolyData(self.original_raw.points)
            surface = points.reconstruct_surface(nbr_sz=nbr_sz)
            if surface.n_cells == 0:
                raise ValueError("La reconstruction de surface a échoué.")

            clus = pyacvd.Clustering(surface.connectivity(largest=True))
            clus.subdivide(2)
            clus.cluster(target_points)
            m = clus.create_mesh()
            return m.smooth_taubin(n_iter=30)

        except Exception as e:
            self.log(f"Erreur dans _pipeline_wrapping: {e}")
            raise


    # --- L'OPTIMISEUR ---
    def optimize(self, runs=None):
        if runs is None:
            runs = [{"method": "classic", "decimate": 0.3, "points": 10000},
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

                score = (q / 60.0) + (1.0 / (1.0 + h['max'] * 1500))
                if m.is_manifold: score += 0.5

                results.append({"mesh": m, "score": score, "h_max": h['max'], "quality": q, "params": r})

            except Exception as e:
                self.log(f"Échec de la combinaison {r}: {e}")
                results.append({"mesh": None, "score": -1, "h_max": float('inf'), "quality": 0, "params": r, "error": str(e)})

        # Filtrer les résultats valides
        valid_results = [r for r in results if r['mesh'] is not None]
        if not valid_results:
            raise ValueError("Aucun résultat valide généré. Vérifiez les paramètres et les entrées.")

        # Trier et sélectionner le meilleur
        valid_results.sort(key=lambda x: x['score'], reverse=True)
        self.mesh = valid_results[0]['mesh']
        self.log(f"Meilleur modèle sélectionné (Score: {valid_results[0]['score']:.2f})")
        return valid_results

    def show_best_results(self, results):
        """Visualise les 3 meilleurs avec l'erreur de Hausdorff en couleur."""
        pv.OFF_SCREEN = True
        p = pv.Plotter(shape=(1, min(3, len(results))))
        for i in range(min(3, len(results))):
            res = results[i]
            p.subplot(0, i)
            h = self.compute_hausdorff(res['mesh'])
            res['mesh']["Error_mm"] = h['local_dist'] * 1000 # Conversion en mm pour le plot
            p.add_mesh(res['mesh'], scalars="Error_mm", cmap="magma")
            p.add_text(f"Rank {i+1}: {res['params']['method']}\nMax Err: {h['max']*1000:.2f}mm", font_size=9)
        p.screenshot("best_mesh.png")



class AortaCapMethod(Enum):
    """Méthodes de capping spécialisées pour l'aorte."""
    DELAUNAY_ADAPTIVE = "delaunay_adaptive"
    FAN_MEDICAL = "fan_medical"
    EXTRUSION_SMOOTH = "extrusion_smooth"
    BSPLINE_PATCH = "bspline_patch"
    POISSON_MEDICAL = "poisson_medical"

class AortaCapper:
    """
    Classe spécialisée pour fermer les ouvertures des maillages d'aorte.
    Optimisée pour s'intégrer après AortaSurfaceCleaner.
    """
    
    def __init__(
        self,
        mesh: pv.PolyData,
        original_mesh: Optional[pv.PolyData] = None,
        max_planarity_deviation: float = 0.5,  # en mm après normalisation
        min_edge_length_ratio: float = 0.1,
        enable_smoothing: bool = True,
        preserve_boundary_shape: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            mesh: Maillage nettoyé (sortie de AortaSurfaceCleaner)
            original_mesh: Maillage original pour référence (optionnel)
            max_planarity_deviation: Tolérance de non-planéité (mm)
            min_edge_length_ratio: Ratio minimal longueur d'arête/moyenne
            enable_smoothing: Lisser les caps après création
            preserve_boundary_shape: Préserver la forme exacte du bord
            logger: Logger personnalisé (optionnel)
        """
        self.mesh = mesh.copy()
        self.original_mesh = original_mesh.copy() if original_mesh is not None else None
        
        # Paramètres pour l'aorte
        self.max_planarity_deviation = max_planarity_deviation / 1000.0  # conversion m
        self.min_edge_length_ratio = min_edge_length_ratio
        self.enable_smoothing = enable_smoothing
        self.preserve_boundary_shape = preserve_boundary_shape
        
        # Configuration du logging
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                self.logger.setLevel(logging.INFO)
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '[AortaCapper] %(levelname)s: %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        
        # Statistiques et métriques
        self.stats = {
            'total_openings': 0,
            'successful_caps': 0,
            'failed_caps': 0,
            'average_cap_area': 0.0,
            'max_boundary_error': 0.0,
            'methods_used': {},
            'processing_time': 0.0
        }
        
        # Cache pour les calculs
        self._boundary_cache = None
        self._mesh_center = None
        
    def _detect_aorta_openings(self) -> List[pv.PolyData]:
        """
        Détection spécialisée des ouvertures de l'aorte.
        Filtre les petits trous et identifie les principales ouvertures.
        """
        # Extraction des bords
        boundaries = self.mesh.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_edges=False,
            manifold_edges=False
        )
        
        if boundaries.n_points == 0:
            return []
        
        # Regroupement des boucles
        try:
            loops = boundaries.connectivity()
            region_ids = np.unique(loops["RegionId"])
        except:
            # Fallback pour anciennes versions de PyVista
            self.logger.warning("Connectivity failed, using manual detection")
            return self._manual_opening_detection(boundaries)
        
        openings = []
        min_points_per_opening = 10  # Ignorer les très petits trous
        
        for rid in region_ids:
            loop = loops.extract_points(loops["RegionId"] == rid)
            
            if loop.n_points < min_points_per_opening:
                self.logger.debug(f"Ignoring small opening with {loop.n_points} points")
                continue
            
            # Calcul de la taille de l'ouverture
            points = loop.points
            centroid = points.mean(axis=0)
            avg_radius = np.mean(np.linalg.norm(points - centroid, axis=1))
            
            # Filtrer les ouvertures trop petites (artefacts)
            if avg_radius < 0.001:  # 1mm
                self.logger.debug(f"Ignoring very small opening (radius: {avg_radius*1000:.2f}mm)")
                continue
            
            openings.append(loop)
            self.logger.info(f"Detected opening {len(openings)}: {loop.n_points} points, "
                           f"radius: {avg_radius*1000:.2f}mm")
        
        self.stats['total_openings'] = len(openings)
        return openings
    
    def _manual_opening_detection(self, boundaries: pv.PolyData) -> List[pv.PolyData]:
        """Détection manuelle des ouvertures."""
        points = boundaries.points
        if boundaries.n_cells == 0:
            return []
        
        lines = boundaries.lines.reshape(-1, 3)[:, 1:]
        visited = set()
        openings = []
        
        for start_idx in range(len(points)):
            if start_idx in visited:
                continue
            
            # Suivre la chaîne connectée
            current_idx = start_idx
            loop_indices = []
            
            while current_idx not in visited:
                loop_indices.append(current_idx)
                visited.add(current_idx)
                
                # Trouver le prochain point connecté
                next_idx = None
                for line in lines:
                    if current_idx in line:
                        other = line[0] if line[1] == current_idx else line[1]
                        if other not in visited:
                            next_idx = other
                            break
                
                if next_idx is None:
                    break
                current_idx = next_idx
            
            if len(loop_indices) >= 3:
                loop_points = points[loop_indices]
                loop_poly = pv.PolyData(loop_points)
                openings.append(loop_poly)
        
        return openings
    
    def _classify_aorta_opening(self, boundary: pv.PolyData) -> Dict:
        """
        Classifie l'ouverture pour choisir la meilleure méthode.
        Retourne: type, planérité, taille, orientation
        """
        points = boundary.points
        centroid = points.mean(axis=0)
        
        # Calcul PCA pour la normale et la planérité
        centered = points - centroid
        cov = centered.T @ centered / len(points)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # La normale est le vecteur propre de la plus petite valeur propre
        normal_idx = np.argmin(eigvals)
        normal = eigvecs[:, normal_idx]
        normal = normal / (np.linalg.norm(normal) + 1e-10)
        
        # Calcul de la planérité
        deviations = np.abs(centered @ normal)
        max_deviation = np.max(deviations)
        avg_deviation = np.mean(deviations)
        
        # Calcul de la taille et de la circularité
        distances = np.linalg.norm(centered, axis=1)
        avg_radius = np.mean(distances)
        radius_variance = np.var(distances)
        circularity = 1.0 / (1.0 + radius_variance / (avg_radius**2 + 1e-10))
        
        # Détermination du type
        if max_deviation < self.max_planarity_deviation:
            planarity_type = "planar"
        elif max_deviation < self.max_planarity_deviation * 3:
            planarity_type = "slightly_curved"
        else:
            planarity_type = "highly_curved"
        
        # Estimation de la taille
        if avg_radius < 0.005:  # 5mm
            size_type = "small"
        elif avg_radius < 0.015:  # 15mm
            size_type = "medium"
        else:
            size_type = "large"
        
        return {
            "type": planarity_type,
            "size": size_type,
            "normal": normal,
            "centroid": centroid,
            "avg_radius": avg_radius,
            "circularity": circularity,
            "max_deviation": max_deviation,
            "points": points,
            "n_points": len(points)
        }
    
    def _create_adaptive_delaunay_cap(self, classification: Dict) -> Optional[pv.PolyData]:
        """Création de cap avec Delaunay adaptatif pour l'aorte."""
        try:
            points = classification["points"]
            normal = classification["normal"]
            centroid = classification["centroid"]
            
            # Ajustement de la densité de points selon la taille
            n_points = len(points)
            if n_points > 100:
                # Rééchantillonnage pour les grandes ouvertures
                from scipy.interpolate import splprep, splev
                tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], s=0, per=1)
                new_u = np.linspace(0, 1, min(100, n_points))
                resampled = np.array(splev(new_u, tck)).T
                points = resampled
            
            # Projection sur plan optimal
            projected = []
            for point in points:
                v = point - centroid
                # Projection orthogonale
                proj = point - np.dot(v, normal) * normal
                projected.append(proj)
            projected = np.array(projected)
            
            # Tri des points par angle pour éviter les croisements
            vec_to_centroid = projected - centroid
            angles = np.arctan2(vec_to_centroid[:, 1], vec_to_centroid[:, 0])
            sorted_idx = np.argsort(angles)
            projected = projected[sorted_idx]
            
            # Création du cap avec vérification
            if len(projected) < 3:
                return None
            
            # Création d'un polygone
            poly = pv.PolyData(projected)
            poly.lines = np.hstack([
                [len(projected)],
                np.arange(len(projected)),
                [0]  # Fermer la boucle
            ])
            
            # Triangulation de Delaunay
            cap = poly.delaunay_2d(tol=1e-6)
            
            if cap.n_cells == 0:
                return None
            
            # Orientation cohérente
            cap_normals = cap.compute_normals(point_normals=False)
            if cap_normals["Normals"].shape[0] > 0:
                cap_normal = cap_normals["Normals"][0]
                if np.dot(cap_normal, normal) < 0:
                    cap = cap.triangulate()
                    cap.flip_normals(inplace=True)
            
            # Lissage léger si demandé
            if self.enable_smoothing:
                cap = cap.smooth_taubin(n_iter=10, pass_band=0.1)
            
            return cap
            
        except Exception as e:
            self.logger.warning(f"Adaptive Delaunay failed: {str(e)}")
            return None
    
    def _create_medical_fan_cap(self, classification: Dict) -> Optional[pv.PolyData]:
        """Création de cap en éventail adapté aux besoins médicaux."""
        try:
            points = classification["points"]
            centroid = classification["centroid"]
            normal = classification["normal"]
            
            # Tri des points par angle pour une triangulation propre
            vec_to_centroid = points - centroid
            angles = np.arctan2(vec_to_centroid[:, 1], vec_to_centroid[:, 0])
            sorted_idx = np.argsort(angles)
            sorted_points = points[sorted_idx]
            
            # Création du point central adaptatif
            # Pour les ouvertures non-planaires, on ajuste le centre
            if classification["max_deviation"] > self.max_planarity_deviation:
                # Utiliser le centroid projeté
                v = sorted_points - centroid
                heights = v @ normal
                adjusted_centroid = centroid + normal * np.mean(heights)
            else:
                adjusted_centroid = centroid
            
            # Création des triangles
            n = len(sorted_points)
            faces = []
            
            # Pour chaque segment du bord
            for i in range(n):
                j = (i + 1) % n
                faces.extend([3, i, j, n])  # n est l'index du point central
            
            # Points avec le centre ajouté à la fin
            all_points = np.vstack([sorted_points, adjusted_centroid.reshape(1, 3)])
            
            # Création du maillage
            cap = pv.PolyData(all_points, faces=np.array(faces))
            
            # Vérification de la qualité
            if cap.n_cells == 0:
                return None
            
            # Orientation
            cap_normals = cap.compute_normals(point_normals=False)
            if cap_normals["Normals"].shape[0] > 0:
                cap_normal = cap_normals["Normals"][0]
                if np.dot(cap_normal, normal) < 0:
                    cap.flip_normals(inplace=True)
            
            return cap
            
        except Exception as e:
            self.logger.warning(f"Medical fan cap failed: {str(e)}")
            return None
    
    def _create_extrusion_cap(self, classification: Dict) -> Optional[pv.PolyData]:
        """Création de cap par extrusion avec lissage."""
        try:
            points = classification["points"]
            normal = classification["normal"]
            centroid = classification["centroid"]
            
            # Détermination de l'épaisseur basée sur la taille
            avg_radius = classification["avg_radius"]
            thickness = min(avg_radius * 0.3, 0.002)  # Max 2mm
            
            # Création de deux anneaux décalés
            inner_points = points - normal * thickness * 0.5
            outer_points = points + normal * thickness * 0.5
            
            # Points combinés
            all_points = np.vstack([inner_points, outer_points])
            
            # Création des faces latérales (quadrangles)
            n = len(points)
            faces = []
            
            for i in range(n):
                j = (i + 1) % n
                # Quadrangle entre anneaux intérieur et extérieur
                faces.extend([4, i, j, n + j, n + i])
            
            # Création des caps d'extrémité
            inner_cap = self._create_medical_fan_cap({
                **classification,
                "points": inner_points,
                "centroid": centroid - normal * thickness * 0.5
            })
            
            outer_cap = self._create_medical_fan_cap({
                **classification,
                "points": outer_points,
                "centroid": centroid + normal * thickness * 0.5
            })
            
            # Combinaison des maillages
            combined = pv.PolyData(all_points, faces=np.array(faces))
            
            if inner_cap is not None and outer_cap is not None:
                combined = combined.merge([inner_cap, outer_cap])
            
            # Lissage pour une transition douce
            if self.enable_smoothing:
                combined = combined.smooth_taubin(n_iter=15, pass_band=0.08)
                combined = combined.compute_normals(auto_orient_normals=True)
            
            return combined
            
        except Exception as e:
            self.logger.warning(f"Extrusion cap failed: {str(e)}")
            return None
    
    def _select_best_capping_method(self, classification: Dict) -> AortaCapMethod:
        """Sélectionne la meilleure méthode basée sur les caractéristiques de l'ouverture."""
        opening_type = classification["type"]
        size_type = classification["size"]
        circularity = classification["circularity"]
        
        if opening_type == "planar":
            if circularity > 0.9 and size_type != "large":
                return AortaCapMethod.DELAUNAY_ADAPTIVE
            else:
                return AortaCapMethod.FAN_MEDICAL
        elif opening_type == "slightly_curved":
            if size_type == "small":
                return AortaCapMethod.FAN_MEDICAL
            else:
                return AortaCapMethod.EXTRUSION_SMOOTH
        else:  # highly_curved
            return AortaCapMethod.EXTRUSION_SMOOTH
    
    def _create_cap_for_opening(self, opening: pv.PolyData, opening_id: int) -> Optional[pv.PolyData]:
        """Crée un cap optimal pour une ouverture spécifique."""
        # Classification de l'ouverture
        classification = self._classify_aorta_opening(opening)
        
        # Sélection de la méthode
        method = self._select_best_capping_method(classification)
        self.logger.info(f"Opening {opening_id}: {classification['type']}, "
                        f"{classification['size']}, method: {method.value}")
        
        # Création du cap
        cap = None
        if method == AortaCapMethod.DELAUNAY_ADAPTIVE:
            cap = self._create_adaptive_delaunay_cap(classification)
        elif method == AortaCapMethod.FAN_MEDICAL:
            cap = self._create_medical_fan_cap(classification)
        elif method == AortaCapMethod.EXTRUSION_SMOOTH:
            cap = self._create_extrusion_cap(classification)
        
        # Validation du cap
        if cap is not None:
            validation = self._validate_cap(cap, classification)
            
            if validation["valid"]:
                self.stats['methods_used'][method.value] = \
                    self.stats['methods_used'].get(method.value, 0) + 1
                
                # Post-traitement
                if self.preserve_boundary_shape:
                    cap = self._constrain_boundary(cap, opening)
                
                return cap
            else:
                self.logger.warning(f"Cap validation failed for opening {opening_id}")
        
        # Fallback: essayer une autre méthode
        self.logger.info(f"Trying fallback methods for opening {opening_id}")
        
        fallback_methods = [
            AortaCapMethod.FAN_MEDICAL,
            AortaCapMethod.DELAUNAY_ADAPTIVE,
            AortaCapMethod.EXTRUSION_SMOOTH
        ]
        
        for fallback in fallback_methods:
            if fallback == method:
                continue
                
            if fallback == AortaCapMethod.FAN_MEDICAL:
                cap = self._create_medical_fan_cap(classification)
            elif fallback == AortaCapMethod.DELAUNAY_ADAPTIVE:
                cap = self._create_adaptive_delaunay_cap(classification)
            elif fallback == AortaCapMethod.EXTRUSION_SMOOTH:
                cap = self._create_extrusion_cap(classification)
            
            if cap is not None:
                validation = self._validate_cap(cap, classification)
                if validation["valid"]:
                    self.stats['methods_used'][fallback.value] = \
                        self.stats['methods_used'].get(fallback.value, 0) + 1
                    self.logger.info(f"Fallback successful with {fallback.value}")
                    
                    if self.preserve_boundary_shape:
                        cap = self._constrain_boundary(cap, opening)
                    
                    return cap
        
        self.stats['failed_caps'] += 1
        self.logger.error(f"All capping methods failed for opening {opening_id}")
        return None
    
    def _validate_cap(self, cap: pv.PolyData, classification: Dict) -> Dict:
        """Validation complète d'un cap créé."""
        validation = {
            "valid": False,
            "n_cells": cap.n_cells,
            "is_manifold": False,
            "has_boundary": True,
            "boundary_error": float('inf'),
            "self_intersection": False
        }
        
        if cap.n_cells == 0:
            return validation
        
        # Vérification manifold
        try:
            non_manifold = cap.extract_feature_edges(
                boundary_edges=False,
                non_manifold_edges=True,
                feature_edges=False
            )
            validation["is_manifold"] = non_manifold.n_points == 0
        except:
            pass
        
        # Vérification des bords
        boundaries = cap.extract_feature_edges(boundary_edges=True)
        validation["has_boundary"] = boundaries.n_points > 0
        
        # Calcul de l'erreur au bord
        if validation["has_boundary"]:
            tree = KDTree(classification["points"])
            distances, _ = tree.query(boundaries.points)
            validation["boundary_error"] = np.max(distances)
            
            # Mise à jour de l'erreur max globale
            self.stats['max_boundary_error'] = max(
                self.stats['max_boundary_error'],
                validation["boundary_error"]
            )
        
        # Vérification de l'auto-intersection (simplifiée)
        validation["self_intersection"] = False  # À implémenter si nécessaire
        
        # Critères de validation
        validation["valid"] = all([
            cap.n_cells > 0,
            validation["is_manifold"],
            not validation["has_boundary"],
            validation["boundary_error"] < self.max_planarity_deviation * 2
        ])
        
        return validation
    
    def _constrain_boundary(self, cap: pv.PolyData, original_boundary: pv.PolyData) -> pv.PolyData:
        """Contraint le bord du cap à correspondre exactement au bord original."""
        if not self.preserve_boundary_shape:
            return cap
        
        try:
            # Trouver les points du cap près du bord
            cap_boundaries = cap.extract_feature_edges(boundary_edges=True)
            if cap_boundaries.n_points == 0:
                return cap
            
            # Créer un arbre KD pour le bord original
            tree = KDTree(original_boundary.points)
            
            # Pour chaque point du bord du cap, trouver le point le plus proche
            # et ajuster sa position
            for i in range(cap_boundaries.n_points):
                point = cap_boundaries.points[i]
                dist, idx = tree.query(point)
                if dist < self.max_planarity_deviation:
                    # Trouver l'index correspondant dans le cap
                    cap_point_idx = np.argmin(
                        np.linalg.norm(cap.points - point, axis=1)
                    )
                    cap.points[cap_point_idx] = original_boundary.points[idx]
            
            return cap.compute_normals(auto_orient_normals=True)
            
        except Exception as e:
            self.logger.warning(f"Boundary constraint failed: {str(e)}")
            return cap
    
    def _merge_with_original(self, caps: List[pv.PolyData]) -> pv.PolyData:
        """Fusionne les caps avec le maillage original de manière optimisée."""
        if not caps:
            self.logger.info("No caps to merge")
            return self.mesh
        
        # Fusion progressive
        current_mesh = self.mesh
        for i, cap in enumerate(caps):
            try:
                # Vérifier l'orientation avant fusion
                cap_boundary = cap.extract_feature_edges(boundary_edges=True)
                if cap_boundary.n_points > 0:
                    self.logger.warning(f"Cap {i} still has boundary points")
                
                current_mesh = current_mesh.merge(cap)
                self.logger.debug(f"Merged cap {i+1}/{len(caps)}")
                
            except Exception as e:
                self.logger.error(f"Failed to merge cap {i}: {str(e)}")
        
        # Nettoyage final
        result = current_mesh.clean()
        result = result.remove_degenerate_cells()
        result = result.remove_unused_points()
        
        # Calcul des normales
        result = result.compute_normals(
            auto_orient_normals=True,
            consistent_normals=True,
            inplace=False
        )
        
        # Vérification finale
        final_boundaries = result.extract_feature_edges(boundary_edges=True)
        remaining_points = final_boundaries.n_points
        
        if remaining_points > 0:
            self.logger.warning(
                f"Mesh still has {remaining_points} boundary points "
                f"({remaining_points/self.mesh.n_points*100:.1f}%)"
            )
        else:
            self.logger.info("Mesh successfully closed - watertight")
        
        return result
    
    def close_all_openings(self) -> Tuple[pv.PolyData, Dict]:
        """
        Ferme toutes les ouvertures détectées dans le maillage.
        
        Returns:
            Tuple[maillage_fermé, statistiques]
        """
        import time
        start_time = time.time()
        
        self.logger.info("Starting aorta capping process...")
        
        # Détection des ouvertures
        openings = self._detect_aorta_openings()
        
        if not openings:
            self.logger.info("No openings detected - mesh is already closed")
            return self.mesh, self.stats
        
        self.logger.info(f"Found {len(openings)} openings to close")
        
        # Création des caps pour chaque ouverture
        caps = []
        for i, opening in enumerate(openings):
            self.logger.info(f"Processing opening {i+1}/{len(openings)} "
                           f"({opening.n_points} points)")
            
            cap = self._create_cap_for_opening(opening, i+1)
            
            if cap is not None:
                caps.append(cap)
                self.stats['successful_caps'] += 1
                
                # Calcul de la surface du cap
                area = cap.area
                self.stats['average_cap_area'] += area
            else:
                self.logger.error(f"Failed to create cap for opening {i+1}")
        
        # Calcul des statistiques
        if self.stats['successful_caps'] > 0:
            self.stats['average_cap_area'] /= self.stats['successful_caps']
        
        # Fusion avec le maillage original
        if caps:
            result = self._merge_with_original(caps)
        else:
            result = self.mesh
            self.logger.warning("No caps were created, returning original mesh")
        
        # Temps de traitement
        self.stats['processing_time'] = time.time() - start_time
        
        # Log des statistiques finales
        self._log_statistics()
        
        return result, self.stats
    
    def _log_statistics(self):
        """Affiche les statistiques de traitement."""
        self.logger.info("=" * 60)
        self.logger.info("AORTA CAPPING - FINAL STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total openings detected: {self.stats['total_openings']}")
        self.logger.info(f"Successful caps: {self.stats['successful_caps']}")
        self.logger.info(f"Failed caps: {self.stats['failed_caps']}")
        
        if self.stats['successful_caps'] > 0:
            self.logger.info(f"Average cap area: {self.stats['average_cap_area']*1e6:.2f} mm²")
            self.logger.info(f"Max boundary error: {self.stats['max_boundary_error']*1000:.2f} mm")
        
        if self.stats['methods_used']:
            self.logger.info("Methods used:")
            for method, count in self.stats['methods_used'].items():
                self.logger.info(f"  - {method}: {count}")
        
        self.logger.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        self.logger.info("=" * 60)


# Fonction utilitaire pour intégration avec AortaSurfaceCleaner
def create_closed_aorta_mesh(
    cleaned_mesh: pv.PolyData,
    original_mesh: Optional[pv.PolyData] = None,
    **capper_kwargs
) -> Tuple[pv.PolyData, Dict]:
    """
    Fonction helper pour intégration facile avec AortaSurfaceCleaner.
    
    Args:
        cleaned_mesh: Maillage nettoyé de AortaSurfaceCleaner.optimize()
        original_mesh: Maillage original (optionnel)
        **capper_kwargs: Paramètres supplémentaires pour AortaCapper
    
    Returns:
        Tuple[maillage_fermé, statistiques]
    """
    # Création du capper
    capper = AortaCapper(
        mesh=cleaned_mesh,
        original_mesh=original_mesh,
        **capper_kwargs
    )
    
    # Fermeture des ouvertures
    return capper.close_all_openings()


# Exemple d'utilisation avec AortaSurfaceCleaner
if __name__ == "__main__":
    # Simuler l'utilisation avec AortaSurfaceCleaner
    cleaner = AortaSurfaceCleaner("path/to/aorta.stl")
    results = cleaner.optimize()
    
    # Récupérer le meilleur maillage nettoyé
    best_mesh = results[0]["mesh"]
    
    # Fermer les ouvertures
    closed_mesh, capping_stats = create_closed_aorta_mesh(
        cleaned_mesh=best_mesh,
        original_mesh=cleaner.original_raw,
        max_planarity_deviation=0.5,  # 0.5 mm
        enable_smoothing=True,
        preserve_boundary_shape=True
    )
    
    # Afficher les résultats
    print("\n" + "="*60)
    print("AORTA PROCESSING COMPLETE")
    print("="*60)
    print(f"Original points: {cleaner.original_raw.n_points}")
    print(f"Cleaned points: {best_mesh.n_points}")
    print(f"Closed mesh points: {closed_mesh.n_points}")
    print(f"Watertight: {closed_mesh.is_manifold}")
    
    # Visualisation
    plotter = pv.Plotter(shape=(1, 3))
    
    plotter.subplot(0, 0)
    plotter.add_mesh(cleaner.original_raw, color='lightblue', show_edges=True)
    plotter.add_text("Original", font_size=10)
    
    plotter.subplot(0, 1)
    plotter.add_mesh(best_mesh, color='lightgreen', show_edges=True)
    plotter.add_text("Cleaned", font_size=10)
    
    plotter.subplot(0, 2)
    plotter.add_mesh(closed_mesh, color='salmon', show_edges=True)
    plotter.add_text("Closed", font_size=10)
    
    plotter.link_views()
    plotter.show()
