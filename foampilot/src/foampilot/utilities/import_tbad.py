import nibabel as nib
import numpy as np
from skimage import measure
import pyvista as pv

def save_nifti_to_obj(nifti_path, output_obj_path):
    # 1. Charger le fichier de segmentation
    img = nib.load(nifti_path)
    data = img.get_fdata()

    # 2. Appliquer Marching Cubes pour extraire la surface
    # On suppose que l'aorte est le label > 0
    verts, faces, normals, values = measure.marching_cubes(data, level=0.5)

    # 3. Sauvegarder au format OBJ
    with open(output_obj_path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            # +1 car les indices OBJ commencent à 1
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")



def prepare_tbad_mesh(nifti_label_path, output_stl_path, labels=(1, 2)):
    """
    Convert TBAD segmentation NIfTI → STL surface (with correct physical scale)
    """
    img = nib.load(nifti_label_path)
    data = img.get_fdata()

    # voxel spacing (mm)
    spacing = img.header.get_zooms()[:3]

    # extract aorta (TL + FL by default)
    binary_mask = np.isin(data, labels).astype(np.uint8)

    # marching cubes with physical scaling
    verts, faces, normals, values = measure.marching_cubes(
        binary_mask,
        level=0.5,
        spacing=spacing
    )

    # PyVista mesh
    pv_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    mesh = pv.PolyData(verts, pv_faces)

    mesh.save(output_stl_path)
    return output_stl_path




import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure
import trimesh
from pathlib import Path


def extract_geometry_from_nifti(
    nii_path,
    output_stl,
    label_value=1,
    zoom_factor=2,
    sdf_sigma_mm=0.6,
):
    """
    Reconstruct a smooth geometric surface from a voxelized NIfTI segmentation
    using a signed distance field (SDF).

    Parameters
    ----------
    nii_path : str | Path
        Path to *_label.nii or *.nii.gz
    output_stl : str | Path
        Output STL path
    label_value : int
        Label to extract (e.g. 1=true lumen, 2=false lumen)
    zoom_factor : int
        SDF upsampling factor (2 is usually enough)
    sdf_sigma_mm : float
        Gaussian smoothing of SDF in mm (TBAD-safe: 0.5–0.8)
    """

    nii_path = Path(nii_path)
    output_stl = Path(output_stl)

    print(f"Loading NIfTI: {nii_path.name}")
    img = nib.load(str(nii_path))
    data = img.get_fdata()

    spacing = img.header.get_zooms()[:3]
    print(f"Voxel spacing (mm): {spacing}")

    # ------------------------------------------------------------------
    # 1. Binary label extraction (NO interpolation here)
    # ------------------------------------------------------------------
    binary = (data == label_value)

    if not np.any(binary):
        raise ValueError(f"Label {label_value} not found in volume")

    # ------------------------------------------------------------------
    # 2. Signed Distance Field (physical units, mm)
    # ------------------------------------------------------------------
    print("Computing signed distance field (SDF)...")

    dist_out = ndimage.distance_transform_edt(~binary, sampling=spacing)
    dist_in  = ndimage.distance_transform_edt(binary,  sampling=spacing)

    sdf = dist_out - dist_in

    # ------------------------------------------------------------------
    # 3. Light SDF smoothing (geometry-safe)
    # ------------------------------------------------------------------
    print(f"Smoothing SDF (sigma = {sdf_sigma_mm} mm)...")

    sigma_vox = [sdf_sigma_mm / s for s in spacing]
    sdf = ndimage.gaussian_filter(sdf, sigma=sigma_vox)

    # ------------------------------------------------------------------
    # 4. SDF super-resolution (THIS is where zoom is allowed)
    # ------------------------------------------------------------------
    if zoom_factor > 1:
        print(f"Upsampling SDF (x{zoom_factor})...")
        sdf = ndimage.zoom(sdf, zoom_factor, order=3)
        spacing = tuple(s / zoom_factor for s in spacing)

    # ------------------------------------------------------------------
    # 5. Marching Cubes on SDF = 0
    # ------------------------------------------------------------------
    print("Extracting implicit surface...")
    verts, faces, _, _ = measure.marching_cubes(
        sdf,
        level=0.0,
        spacing=spacing,
    )

    # ------------------------------------------------------------------
    # 6. Apply affine (world coordinates)
    # ------------------------------------------------------------------
    affine = img.affine.copy()
    affine[:3, :3] /= zoom_factor

    verts = (np.c_[verts, np.ones(len(verts))] @ affine.T)[:, :3]

    # ------------------------------------------------------------------
    # 7. Mesh cleanup (minimal – geometry already smooth)
    # ------------------------------------------------------------------
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        process=True,
    )

    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.rezero()

    mesh.export(output_stl)
    print(f"✔ Geometry exported: {output_stl}")


import numpy as np
import nibabel as nib
import trimesh
import gc
import time
import warnings
from typing import Dict, Tuple, Optional, List
from scipy.ndimage import (
    distance_transform_edt, 
    binary_fill_holes, 
    label, 
    zoom, 
    gaussian_filter,
    binary_erosion
)
from joblib import Parallel, delayed
from skimage.measure import marching_cubes
from trimesh import repair

class AdaptiveSDFRefinerV4:
    """
    Version CFD-Ready : Intègre le Blending de blocs et le monitoring de volume.
    Améliorations : blending sigmoïde, validation de volume, détection automatique.
    """
    
    def __init__(self, mask: np.ndarray, refine_factor: int = 2, 
                 block_size: int = 64, n_jobs: int = -1):
        """
        Initialise le raffineur SDF.
        
        Args:
            mask: Masque binaire 3D
            refine_factor: Facteur de raffinement (ex: 2 pour double résolution)
            block_size: Taille des blocs pour traitement parallèle
            n_jobs: Nombre de jobs parallèles (-1 pour tous les cœurs)
        """
        if mask.ndim != 3:
            raise ValueError(f"Le masque doit être 3D, reçu: {mask.ndim}D")
        
        self.mask = mask.astype(bool)
        self.factor = refine_factor
        self.dmax = max(2, refine_factor)
        self.block_size = block_size
        self.n_jobs = n_jobs
        self.halo = 4
        self.sdf = None
        self.volume_ratio = None
        
    @staticmethod
    def compute_sdf(mask: np.ndarray) -> np.ndarray:
        """
        Calcule la Signed Distance Function (SDF) en float32.
        
        Args:
            mask: Masque binaire
            
        Returns:
            SDF avec valeurs positives à l'extérieur, négatives à l'intérieur
        """
        pos = distance_transform_edt(~mask)
        neg = distance_transform_edt(mask)
        return (pos - neg).astype(np.float32)
    
    def _get_blending_weight(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Crée une carte de poids sigmoïde pour le blending des blocs.
        
        Args:
            shape: Dimensions du bloc
            
        Returns:
            Carte de poids normalisée
        """
        w = np.ones(shape, dtype=np.float32)
        h = self.halo * self.factor
        
        for d in range(3):
            if shape[d] > 2 * h:
                # Fonction sigmoïde pour transition douce
                x = np.linspace(-3, 3, h)
                grad = 1 / (1 + np.exp(-x))  # Sigmoid
                
                # Appliquer aux deux extrémités
                slices_start = [slice(None)] * 3
                slices_start[d] = slice(0, h)
                w[tuple(slices_start)] *= grad.reshape(
                    [-1 if i == d else 1 for i in range(3)]
                )
                
                slices_end = [slice(None)] * 3
                slices_end[d] = slice(-h, None)
                w[tuple(slices_end)] *= grad[::-1].reshape(
                    [-1 if i == d else 1 for i in range(3)]
                )
        
        # Normalisation pour éviter les dépassements
        w_max = np.max(w)
        if w_max > 0:
            w = w / w_max
        
        return w
    
    def check_volume_conservation(self, initial_mask: np.ndarray, 
                                  final_mask: np.ndarray, 
                                  spacing: Tuple[float, float, float]) -> float:
        """
        Vérifie la conservation du volume pendant le raffinement.
        
        Args:
            initial_mask: Masque original
            final_mask: Masque raffiné
            spacing: Espacement des voxels (mm)
            
        Returns:
            Ratio volume_final / volume_initial
        """
        voxel_volume = np.prod(spacing)
        vol_initial = np.sum(initial_mask) * voxel_volume
        vol_final = np.sum(final_mask) * voxel_volume * (self.factor ** 3)
        
        ratio = vol_final / (vol_initial + 1e-8)
        self.volume_ratio = ratio
        
        if abs(ratio - 1) > 0.1:  # 10% de tolérance
            warnings.warn(
                f"Variation de volume importante: {ratio:.3f} "
                f"(attendu ~1.0, tolérance ±0.1)"
            )
        
        return ratio
    
    def iterate(self, initial_spacing: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        Exécute une itération complète de raffinement.
        
        Args:
            initial_spacing: Espacement pour vérification de volume
            
        Returns:
            SDF raffinée
        """
        print("  → Nettoyage topologique...")
        # Nettoyage et extraction de la composante principale
        labels, nlab = label(binary_fill_holes(self.mask))
        if nlab > 1:
            sizes = np.bincount(labels.ravel())
            main_component = np.argmax(sizes[1:]) + 1
            self.mask = labels == main_component
        else:
            self.mask = labels == 1
        
        print("  → Calcul SDF initiale...")
        self.sdf = self.compute_sdf(self.mask)
        
        # Sauvegarde pour vérification volume
        initial_mask_for_check = self.mask.copy()
        
        # Identification de la narrow band
        print("  → Identification narrow band...")
        band = np.abs(self.sdf) < self.dmax
        self.mask = None
        gc.collect()
        
        # Préparation des blocs
        print("  → Préparation des blocs...")
        bs = self.block_size
        slices = []
        
        for i in range(0, self.sdf.shape[0], bs):
            for j in range(0, self.sdf.shape[1], bs):
                for k in range(0, self.sdf.shape[2], bs):
                    slc = (
                        slice(max(i - self.halo, 0), 
                              min(i + bs + self.halo, self.sdf.shape[0])),
                        slice(max(j - self.halo, 0), 
                              min(j + bs + self.halo, self.sdf.shape[1])),
                        slice(max(k - self.halo, 0), 
                              min(k + bs + self.halo, self.sdf.shape[2]))
                    )
                    if np.any(band[slc]):
                        slices.append(slc)
        
        print(f"  → {len(slices)} blocs actifs à raffiner...")
        
        # Raffinement parallèle
        print("  → Raffinement parallèle...")
        refined_masks = Parallel(n_jobs=self.n_jobs)(
            delayed(lambda s: zoom(self.sdf[s] < 0, self.factor, order=1) > 0.5)(s)
            for s in slices
        )
        
        # Reconstruction avec blending
        print("  → Reconstruction avec blending...")
        new_shape = tuple(s * self.factor for s in self.sdf.shape)
        weight_sum = np.zeros(new_shape, dtype=np.float32)
        value_sum = np.zeros(new_shape, dtype=np.float32)
        
        for slc, rb in zip(slices, refined_masks):
            tgt = tuple(
                slice(s.start * self.factor, 
                      s.start * self.factor + rb.shape[d])
                for d, s in enumerate(slc)
            )
            w = self._get_blending_weight(rb.shape)
            value_sum[tgt] += rb.astype(np.float32) * w
            weight_sum[tgt] += w
        
        # Nettoyage mémoire intermédiaire
        del self.sdf, refined_masks, band
        gc.collect()
        
        # Seuil de vote majoritaire
        print("  → Fusion des blocs...")
        final_mask = (value_sum / (weight_sum + 1e-8)) > 0.5
        
        # Vérification volume si spacing fourni
        if initial_spacing is not None:
            self.check_volume_conservation(
                initial_mask_for_check, final_mask, initial_spacing
            )
        
        # Calcul SDF finale
        print("  → Calcul SDF finale...")
        self.sdf = self.compute_sdf(final_mask)
        
        return self.sdf

class CFDMeshValidator:
    """
    Valide un maillage pour les simulations CFD.
    Vérifie la topologie, la géométrie et la qualité des éléments.
    """
    
    CFD_REQUIREMENTS = {
        'watertight': True,
        'min_triangle_quality': 0.01,
        'max_aspect_ratio': 20.0,
        'max_hole_perimeter': 10.0,  # en mm
        'valid_euler_numbers': [0, 2],  # Accepte sphère ou tore
        'min_volume': 1.0,  # mm³
    }
    
    @classmethod
    def validate(cls, mesh: trimesh.Trimesh, 
                 spacing: Tuple[float, float, float]) -> Dict:
        """
        Valide un maillage pour les simulations CFD.
        
        Args:
            mesh: Maillage à valider
            spacing: Espacement des voxels (pour échelle)
            
        Returns:
            Dictionnaire de rapport de validation
        """
        report = {
            'passed': True,
            'requirements': cls.CFD_REQUIREMENTS.copy(),
            'metrics': {},
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        # 1. Vérifications topologiques de base
        report['metrics']['watertight'] = mesh.is_watertight
        report['metrics']['euler_number'] = mesh.euler_number
        report['metrics']['volume'] = mesh.volume
        report['metrics']['area'] = mesh.area
        
        if not mesh.is_watertight:
            report['errors'].append("Maillage non-fermé (non-watertight)")
            report['passed'] = False
        
        if mesh.euler_number not in cls.CFD_REQUIREMENTS['valid_euler_numbers']:
            report['warnings'].append(
                f"Nombre d'Euler suspect: {mesh.euler_number} "
                f"(attendu {cls.CFD_REQUIREMENTS['valid_euler_numbers']})"
            )
        
        if mesh.volume < cls.CFD_REQUIREMENTS['min_volume']:
            report['warnings'].append(
                f"Volume très faible: {mesh.volume:.3f} mm³"
            )
        
        # 2. Qualité géométrique
        triangles = mesh.triangles
        
        # Calcul qualité des triangles
        quality = cls.compute_triangle_quality(triangles)
        report['metrics']['triangle_quality_min'] = float(np.min(quality))
        report['metrics']['triangle_quality_mean'] = float(np.mean(quality))
        report['metrics']['triangle_quality_std'] = float(np.std(quality))
        report['metrics']['triangle_quality_p5'] = float(np.percentile(quality, 5))
        
        if report['metrics']['triangle_quality_min'] < cls.CFD_REQUIREMENTS['min_triangle_quality']:
            report['warnings'].append(
                f"Qualité triangle minimale trop basse: "
                f"{report['metrics']['triangle_quality_min']:.3f} "
                f"(minimum requis: {cls.CFD_REQUIREMENTS['min_triangle_quality']})"
            )
        
        # Calcul ratio d'aspect
        aspect_ratios = cls.compute_aspect_ratios(triangles)
        report['metrics']['aspect_ratio_max'] = float(np.max(aspect_ratios))
        
        if report['metrics']['aspect_ratio_max'] > cls.CFD_REQUIREMENTS['max_aspect_ratio']:
            report['warnings'].append(
                f"Ratio d'aspect maximum trop élevé: "
                f"{report['metrics']['aspect_ratio_max']:.1f} "
                f"(maximum: {cls.CFD_REQUIREMENTS['max_aspect_ratio']})"
            )
        
        # 3. Dimensions caractéristiques
        edge_stats = cls.compute_edge_stats(mesh, spacing)
        report['metrics']['edge_length_mean'] = edge_stats['mean']
        report['metrics']['edge_length_std'] = edge_stats['std']
        report['metrics']['edge_length_min'] = edge_stats['min']
        report['metrics']['edge_length_max'] = edge_stats['max']
        
        # 4. Courbure estimée
        curvature = cls.estimate_curvature(mesh)
        report['metrics']['curvature_mean'] = curvature['mean']
        report['metrics']['curvature_std'] = curvature['std']
        
        # 5. Suggestions d'amélioration
        if report['metrics']['triangle_quality_p5'] < 0.1:
            report['suggestions'].append(
                "Considérer une subdivision ou un remaillage pour améliorer "
                "la qualité des triangles"
            )
        
        if report['metrics']['edge_length_std'] / report['metrics']['edge_length_mean'] > 0.5:
            report['suggestions'].append(
                "Les arêtes ont des longueurs très variables, "
                "un lissage pourrait être bénéfique"
            )
        
        return report
    
    @staticmethod
    def compute_triangle_quality(triangles: np.ndarray) -> np.ndarray:
        """
        Calcule la qualité des triangles (0-1, 1=parfait).
        
        Args:
            triangles: Array de triangles (N, 3, 3)
            
        Returns:
            Qualité de chaque triangle
        """
        a = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
        b = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
        c = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
        
        # Ratio aire / (périmètre²) normalisé
        s = (a + b + c) / 2
        area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0))
        quality = 4 * np.sqrt(3) * area / (a**2 + b**2 + c**2 + 1e-8)
        
        return np.clip(quality, 0, 1)
    
    @staticmethod
    def compute_aspect_ratios(triangles: np.ndarray) -> np.ndarray:
        """
        Calcule le ratio d'aspect des triangles.
        
        Args:
            triangles: Array de triangles
            
        Returns:
            Ratio d'aspect de chaque triangle
        """
        a = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
        b = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
        c = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
        
        sides = np.column_stack([a, b, c])
        aspect_ratios = np.max(sides, axis=1) / (np.min(sides, axis=1) + 1e-8)
        
        return aspect_ratios
    
    @staticmethod
    def compute_edge_stats(mesh: trimesh.Trimesh, 
                          spacing: Tuple[float, float, float]) -> Dict:
        """
        Calcule les statistiques des longueurs d'arêtes.
        
        Args:
            mesh: Maillage
            spacing: Espacement pour échelle
            
        Returns:
            Statistiques des arêtes
        """
        edges = mesh.edges_unique_length
        edge_lengths_mm = edges * np.mean(spacing)
        
        return {
            'mean': float(np.mean(edge_lengths_mm)),
            'std': float(np.std(edge_lengths_mm)),
            'min': float(np.min(edge_lengths_mm)),
            'max': float(np.max(edge_lengths_mm)),
            'median': float(np.median(edge_lengths_mm))
        }
    
    @staticmethod
    def estimate_curvature(mesh: trimesh.Trimesh) -> Dict:
        """
        Estime la courbure locale du maillage.
        
        Args:
            mesh: Maillage
            
        Returns:
            Statistiques de courbure
        """
        # Approximation simple basée sur les normales voisines
        if len(mesh.vertices) == 0:
            return {'mean': 0.0, 'std': 0.0}
        
        # Pour chaque sommet, variance des normales des faces adjacentes
        vertex_normals = np.zeros((len(mesh.vertices), 3))
        vertex_counts = np.zeros(len(mesh.vertices))
        
        for face in mesh.faces:
            normal = mesh.face_normals[face[0]]
            for vertex in face:
                vertex_normals[vertex] += normal
                vertex_counts[vertex] += 1
        
        # Moyenne des normales
        valid = vertex_counts > 0
        vertex_normals[valid] /= vertex_counts[valid, np.newaxis]
        
        # Normalisation
        norms = np.linalg.norm(vertex_normals, axis=1)
        valid_norms = norms > 0
        vertex_normals[valid_norms] /= norms[valid_norms, np.newaxis]
        
        # Variance angulaire comme proxy de courbure
        curvature = np.zeros(len(mesh.vertices))
        for i, (vertex, normal) in enumerate(zip(mesh.vertices, vertex_normals)):
            if vertex_counts[i] > 0:
                # Distance moyenne aux voisins
                neighbors = mesh.vertex_neighbors[i]
                if neighbors:
                    neighbor_dists = np.linalg.norm(
                        mesh.vertices[neighbors] - vertex, axis=1
                    )
                    curvature[i] = np.mean(neighbor_dists)
        
        valid_curvature = curvature > 0
        if np.any(valid_curvature):
            return {
                'mean': float(np.mean(curvature[valid_curvature])),
                'std': float(np.std(curvature[valid_curvature]))
            }
        else:
            return {'mean': 0.0, 'std': 0.0}

def prepare_cfd_mesh_robust(verts: np.ndarray, faces: np.ndarray, 
                           spacing: Tuple[float, float, float], 
                           target_triangles: Optional[int] = None) -> Tuple[trimesh.Trimesh, Dict]:
    """
    Préparation robuste d'un maillage pour les simulations CFD.
    
    Args:
        verts: Vertices du maillage
        faces: Faces du maillage
        spacing: Espacement des voxels
        target_triangles: Nombre cible de triangles (optionnel)
        
    Returns:
        Tuple (maillage traité, statistiques)
    """
    print("  → Création du maillage...")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # 1. Nettoyage initial
    print("  → Nettoyage initial...")
    mesh.remove_infinite_values()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    
    # 2. Vérification de l'orientation
    if not mesh.is_winding_consistent:
        print("  → Correction de l'orientation...")
        mesh.fix_normals()
    
    # 3. Fermeture topologique garantie
    print("  → Vérification topologique...")
    if not mesh.is_watertight:
        print("  → Fermeture des trous...")
        try:
            mesh = repair.fill_holes(mesh)
        except Exception as e:
            print(f"    ⚠️  Échec fill_holes: {e}")
        
        # Vérification finale
        if not mesh.is_watertight:
            print("  → Tentative de réparation des faces...")
            try:
                mesh = repair.broken_faces(mesh)
            except Exception as e:
                print(f"    ⚠️  Échec broken_faces: {e}")
    
    # 4. Raffinement adaptatif si nécessaire
    if target_triangles and len(mesh.faces) < target_triangles * 0.5:
        print(f"  → Raffinement adaptatif...")
        target_edge_length = np.mean(spacing) * 0.7
        try:
            # Subdivision basée sur la longueur d'arête
            current_edges = mesh.edges_unique_length
            if np.mean(current_edges) > target_edge_length * 1.5:
                mesh = mesh.subdivide()
        except Exception as e:
            print(f"    ⚠️  Échec subdivision: {e}")
    
    # 5. Calcul des statistiques
    print("  → Calcul des statistiques...")
    stats = {
        'watertight': mesh.is_watertight,
        'euler_number': mesh.euler_number,
        'vertices': len(mesh.vertices),
        'faces': len(mesh.faces),
        'volume': float(mesh.volume),
        'area': float(mesh.area),
    }
    
    # Statistiques supplémentaires si le maillage est valide
    if len(mesh.faces) > 0:
        # Qualité des triangles
        triangles = mesh.triangles
        a = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
        b = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
        c = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
        s = (a + b + c) / 2
        area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0))
        quality = 4 * np.sqrt(3) * area / (a**2 + b**2 + c**2 + 1e-8)
        
        stats.update({
            'triangle_quality_min': float(np.min(quality)),
            'triangle_quality_mean': float(np.mean(quality)),
            'triangle_quality_std': float(np.std(quality)),
            'aspect_ratio_max': float(np.max(np.column_stack([a/b, b/c, c/a]))),
        })
    
    return mesh, stats

def detect_image_type(data: np.ndarray) -> Tuple[bool, float]:
    """
    Détecte automatiquement le type d'image et détermine un seuil approprié.
    
    Args:
        data: Données d'image 3D
        
    Returns:
        Tuple (is_ct, threshold)
    """
    # 1. Vérifier la plage dynamique
    data_range = np.percentile(data, 99.9) - np.percentile(data, 0.1)
    
    # 2. Vérifier la présence de valeurs négatives typiques des CT
    negative_fraction = np.sum(data < -50) / data.size
    
    # 3. Détection basée sur distribution
    is_ct = (data_range > 500) or (negative_fraction > 0.05)
    
    if is_ct:
        # Seuil adaptatif pour CT: percentile haut des tissus
        tissue_data = data[(data > -100) & (data < 1000)]
        if len(tissue_data) > 100:
            threshold = np.percentile(tissue_data, 97)
        else:
            threshold = 200  # Valeur par défaut pour l'os
    else:
        # Image binaire ou segmentée
        threshold = 0.5
    
    return is_ct, threshold

def full_cfd_pipeline_robust(nii_path: str, out_stl: str, factor: int = 2,
                            min_triangle_quality: float = 0.01,
                            save_intermediate: bool = False) -> Tuple[trimesh.Trimesh, Dict, bool]:
    """
    Pipeline complet CFD avec validation à chaque étape.
    
    Args:
        nii_path: Chemin vers le fichier NIfTI
        out_stl: Chemin de sortie pour le fichier STL
        factor: Facteur de raffinement
        min_triangle_quality: Qualité minimale des triangles acceptée
        save_intermediate: Sauvegarder les fichiers intermédiaires
        
    Returns:
        Tuple (maillage, statistiques, validation_passée)
    """
    print("=" * 70)
    print("CFD PIPELINE - VERSION ROBUSTE")
    print("=" * 70)
    
    total_start_time = time.time()
    all_stats = {}
    
    # 1. CHARGEMENT
    print("\n1. CHARGEMENT DE L'IMAGE")
    load_start = time.time()
    
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        affine = img.affine
        original_spacing = img.header.get_zooms()[:3]
        
        print(f"   ✓ Fichier chargé: {nii_path}")
        print(f"   ✓ Dimensions: {data.shape}")
        print(f"   ✓ Espacement: {original_spacing} mm")
        print(f"   ✓ Plage de valeurs: [{data.min():.1f}, {data.max():.1f}]")
    except Exception as e:
        print(f"   ❌ Erreur de chargement: {e}")
        return None, {}, False
    
    load_time = time.time() - load_start
    print(f"   ⏱️  Temps de chargement: {load_time:.1f}s")
    
    # 2. SEGMENTATION
    print("\n2. SEGMENTATION")
    seg_start = time.time()
    
    # Détection automatique du type d'image
    is_ct, auto_threshold = detect_image_type(data)
    
    if is_ct:
        print(f"   ✓ Mode CT-Scan détecté")
        print(f"   ✓ Seuil auto-détecté: {auto_threshold:.1f} HU")
        
        # Lissage adaptatif basé sur la résolution
        sigma = min(1.0, np.mean(original_spacing) / 2.0)
        data_smoothed = gaussian_filter(data, sigma=sigma)
        mask = data_smoothed >= auto_threshold
    else:
        print(f"   ✓ Mode segmentation binaire détecté")
        mask = data > 0.5
    
    mask = mask.astype(bool)
    voxel_count = np.sum(mask)
    voxel_percentage = voxel_count / mask.size * 100
    
    print(f"   ✓ Voxels sélectionnés: {voxel_count:,} ({voxel_percentage:.1f}%)")
    
    # Sauvegarde intermédiaire
    if save_intermediate:
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
        mask_path = out_stl.replace('.stl', '_mask.nii.gz')
        nib.save(mask_img, mask_path)
        print(f"   ✓ Masque sauvegardé: {mask_path}")
    
    seg_time = time.time() - seg_start
    print(f"   ⏱️  Temps de segmentation: {seg_time:.1f}s")
    
    # 3. RAFFINEMENT SDF
    print("\n3. RAFFINEMENT PAR SDF")
    refine_start = time.time()
    
    try:
        refiner = AdaptiveSDFRefinerV4(mask, refine_factor=factor)
        sdf = refiner.iterate(initial_spacing=original_spacing)
        
        if refiner.volume_ratio is not None:
            print(f"   ✓ Ratio de conservation volume: {refiner.volume_ratio:.3f}")
            all_stats['volume_ratio'] = refiner.volume_ratio
        
        # Sauvegarde SDF intermédiaire
        if save_intermediate:
            sdf_img = nib.Nifti1Image(sdf, affine)
            sdf_path = out_stl.replace('.stl', '_sdf.nii.gz')
            nib.save(sdf_img, sdf_path)
            print(f"   ✓ SDF sauvegardée: {sdf_path}")
            
    except Exception as e:
        print(f"   ❌ Erreur lors du raffinement: {e}")
        print("   → Utilisation du masque original...")
        # Fallback: utiliser le masque original
        sdf = refiner.compute_sdf(mask)
    
    refine_time = time.time() - refine_start
    print(f"   ⏱️  Temps de raffinement: {refine_time:.1f}s")
    
    # 4. EXTRACTION DE SURFACE
    print("\n4. EXTRACTION DE SURFACE")
    extract_start = time.time()
    
    new_spacing = [d / factor for d in original_spacing]
    target_triangles = int((voxel_count ** (2/3)) * 2)  # Estimation
    
    print(f"   ✓ Résolution cible: {new_spacing} mm/voxel")
    print(f"   ✓ Triangles estimés: {target_triangles:,}")
    
    try:
        # Extraction marching cubes
        verts, faces, normals, _ = marching_cubes(
            sdf,
            level=0.0,
            spacing=new_spacing,
            gradient_direction='descent',
            step_size=1,
            allow_degenerate=False
        )
        
        print(f"   ✓ Triangles extraits: {len(faces):,}")
        
    except ValueError as e:
        print(f"   ⚠️  Erreur marching cubes: {e}")
        print("   → Tentative avec paramètres alternatifs...")
        
        try:
            verts, faces, normals, _ = marching_cubes(
                sdf,
                level=0.0,
                spacing=new_spacing,
                step_size=2  # Step plus grand pour stabilité
            )
            print(f"   ✓ Triangles extraits (alternatif): {len(faces):,}")
        except Exception as e2:
            print(f"   ❌ Échec extraction alternative: {e2}")
            print("   → Extraction sur masque original...")
            
            # Dernier recours: extraction sur masque original
            verts, faces, normals, _ = marching_cubes(
                mask.astype(float),
                level=0.5,
                spacing=original_spacing
            )
            print(f"   ✓ Triangles extraits (fallback): {len(faces):,}")
    
    extract_time = time.time() - extract_start
    print(f"   ⏱️  Temps d'extraction: {extract_time:.1f}s")
    
    # 5. PRÉPARATION DU MAILLAGE CFD
    print("\n5. PRÉPARATION DU MAILLAGE CFD")
    prep_start = time.time()
    
    try:
        mesh, prep_stats = prepare_cfd_mesh_robust(
            verts, faces, new_spacing, target_triangles
        )
        all_stats.update(prep_stats)
        
        print(f"   ✓ Maillage préparé: {len(mesh.vertices)} sommets, "
              f"{len(mesh.faces)} faces")
        print(f"   ✓ Volume: {mesh.volume:.1f} mm³")
        print(f"   ✓ Surface: {mesh.area:.1f} mm²")
        
    except Exception as e:
        print(f"   ❌ Erreur préparation maillage: {e}")
        # Création d'un maillage minimal pour l'export de débogage
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        prep_stats = {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'volume': float(mesh.volume),
            'area': float(mesh.area),
            'watertight': False,
            'error': str(e)
        }
        all_stats.update(prep_stats)
    
    prep_time = time.time() - prep_start
    print(f"   ⏱️  Temps de préparation: {prep_time:.1f}s")
    
    # 6. VALIDATION CFD
    print("\n6. VALIDATION POUR CFD")
    valid_start = time.time()
    
    try:
        validator = CFDMeshValidator()
        validation_report = validator.validate(mesh, new_spacing)
        all_stats['validation'] = validation_report
        
        # Afficher les avertissements et erreurs
        if validation_report['warnings']:
            print(f"   ⚠️  Avertissements ({len(validation_report['warnings'])}):")
            for warning in validation_report['warnings'][:3]:  # Limiter à 3
                print(f"     • {warning}")
            if len(validation_report['warnings']) > 3:
                print(f"     • ... et {len(validation_report['warnings']) - 3} autres")
        
        if validation_report['errors']:
            print(f"   ❌ Erreurs ({len(validation_report['errors'])}):")
            for error in validation_report['errors']:
                print(f"     • {error}")
        
        if validation_report['suggestions']:
            print(f"   💡 Suggestions ({len(validation_report['suggestions'])}):")
            for suggestion in validation_report['suggestions']:
                print(f"     • {suggestion}")
        
        print(f"   ✓ Validation: {'PASS' if validation_report['passed'] else 'FAIL'}")
        
    except Exception as e:
        print(f"   ⚠️  Erreur lors de la validation: {e}")
        validation_report = {'passed': False, 'errors': [str(e)]}
        all_stats['validation'] = validation_report
    
    valid_time = time.time() - valid_start
    print(f"   ⏱️  Temps de validation: {valid_time:.1f}s")
    
    # 7. EXPORT
    print("\n7. EXPORT")
    export_start = time.time()
    
    validation_passed = validation_report.get('passed', False)
    
    if validation_passed or save_intermediate:
        # Export STL principal
        try:
            mesh.export(out_stl)
            print(f"   ✓ STL exporté: {out_stl}")
        except Exception as e:
            print(f"   ❌ Erreur export STL: {e}")
        
        # Export OBJ avec normales
        try:
            obj_path = out_stl.replace('.stl', '.obj')
            mesh.export(obj_path)
            print(f"   ✓ OBJ exporté: {obj_path}")
        except Exception as e:
            print(f"   ⚠️  Erreur export OBJ: {e}")
    else:
        print("   ⚠️  Export principal annulé - validation échouée")
        out_stl = out_stl.replace('.stl', '_DEBUG.stl')
    
    # Export de débogage si validation échouée
    if not validation_passed:
        debug_path = out_stl.replace('.stl', '_DEBUG.stl')
        try:
            mesh.export(debug_path)
            print(f"   ✓ Maillage de débogage: {debug_path}")
        except Exception as e:
            print(f"   ⚠️  Erreur export débogage: {e}")
    
    # Sauvegarde du rapport
    report_path = out_stl.replace('.stl', '_REPORT.txt')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("RAPPORT DE QUALITÉ MAILLAGE CFD\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("INFORMATIONS GÉNÉRALES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Fichier source: {nii_path}\n")
            f.write(f"Fichier sortie: {out_stl}\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Facteur de raffinement: {factor}\n")
            f.write(f"Résolution originale: {original_spacing} mm\n")
            f.write(f"Résolution finale: {new_spacing} mm\n\n")
            
            f.write("STATISTIQUES DE TRAITEMENT\n")
            f.write("-" * 40 + "\n")
            f.write(f"Temps total: {time.time() - total_start_time:.1f}s\n")
            f.write(f"Chargement: {load_time:.1f}s\n")
            f.write(f"Segmentation: {seg_time:.1f}s\n")
            f.write(f"Raffinement: {refine_time:.1f}s\n")
            f.write(f"Extraction: {extract_time:.1f}s\n")
            f.write(f"Préparation: {prep_time:.1f}s\n")
            f.write(f"Validation: {valid_time:.1f}s\n\n")
            
            f.write("STATISTIQUES DU MAILLAGE\n")
            f.write("-" * 40 + "\n")
            for key, value in all_stats.items():
                if key != 'validation':
                    f.write(f"{key}: {value}\n")
            
            if 'validation' in all_stats:
                f.write("\nRAPPORT DE VALIDATION\n")
                f.write("-" * 40 + "\n")
                f.write(f"Status: {'PASS' if validation_passed else 'FAIL'}\n\n")
                
                if all_stats['validation'].get('errors'):
                    f.write("ERREURS:\n")
                    for error in all_stats['validation']['errors']:
                        f.write(f"  • {error}\n")
                    f.write("\n")
                
                if all_stats['validation'].get('warnings'):
                    f.write("AVERTISSEMENTS:\n")
                    for warning in all_stats['validation']['warnings']:
                        f.write(f"  • {warning}\n")
                    f.write("\n")
                
                if all_stats['validation'].get('suggestions'):
                    f.write("SUGGESTIONS:\n")
                    for suggestion in all_stats['validation']['suggestions']:
                        f.write(f"  • {suggestion}\n")
                    f.write("\n")
                
                f.write("MÉTRIQUES DÉTAILLÉES:\n")
                for key, value in all_stats['validation'].get('metrics', {}).items():
                    f.write(f"  {key}: {value}\n")
        
        print(f"   ✓ Rapport sauvegardé: {report_path}")
    except Exception as e:
        print(f"   ⚠️  Erreur sauvegarde rapport: {e}")
    
    export_time = time.time() - export_start
    print(f"   ⏱️  Temps d'export: {export_time:.1f}s")
    
    # 8. RÉSUMÉ FINAL
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 70)
    print("RÉSUMÉ FINAL")
    print("=" * 70)
    
    print(f"⏱️  Temps total: {total_time:.1f}s")
    print(f"📏 Résolution: {new_spacing} mm/voxel")
    print(f"📐 Maillage: {all_stats.get('vertices', 0):,} sommets, "
          f"{all_stats.get('faces', 0):,} faces")
    
    if 'triangle_quality_mean' in all_stats:
        print(f"⭐ Qualité triangles: "
              f"min={all_stats.get('triangle_quality_min', 0):.3f}, "
              f"moy={all_stats.get('triangle_quality_mean', 0):.3f}")
    
    print(f"🔒 Watertight: {all_stats.get('watertight', False)}")
    print(f"📦 Volume: {all_stats.get('volume', 0):.1f} mm³")
    
    if 'validation' in all_stats:
        status = "✅ PASS" if validation_passed else "❌ FAIL"
        print(f"📋 Validation CFD: {status}")
    
    print(f"💾 Fichiers générés:")
    print(f"   • {out_stl}")
    if validation_passed or save_intermediate:
        print(f"   • {out_stl.replace('.stl', '.obj')}")
    print(f"   • {report_path}")
    
    if save_intermediate:
        print(f"   • {out_stl.replace('.stl', '_mask.nii.gz')}")
        print(f"   • {out_stl.replace('.stl', '_sdf.nii.gz')}")
    
    print("=" * 70)
    
    return mesh, all_stats, validation_passed

def batch_process_pipeline(input_files: List[str], output_dir: str, 
                          factor: int = 2, **kwargs):
    """
    Traite plusieurs fichiers en batch.
    
    Args:
        input_files: Liste des fichiers NIfTI d'entrée
        output_dir: Répertoire de sortie
        factor: Facteur de raffinement
        **kwargs: Arguments supplémentaires pour full_cfd_pipeline_robust
    """
    import os
    
    print(f"🚀 Démarrage du traitement batch ({len(input_files)} fichiers)")
    print(f"📁 Répertoire de sortie: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Création du répertoire de sortie")
    
    results = []
    
    for i, input_file in enumerate(input_files, 1):
        print(f"\n{'='*50}")
        print(f"Traitement du fichier {i}/{len(input_files)}")
        print(f"Fichier: {os.path.basename(input_file)}")
        print(f"{'='*50}")
        
        # Générer le nom de sortie
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        if base_name.endswith('.nii'):
            base_name = base_name[:-4]
        
        output_file = os.path.join(output_dir, f"{base_name}_cfd.stl")
        
        try:
            mesh, stats, passed = full_cfd_pipeline_robust(
                input_file, output_file, factor=factor, **kwargs
            )
            
            results.append({
                'input': input_file,
                'output': output_file,
                'passed': passed,
                'stats': stats
            })
            
            print(f"\n✅ Fichier {i} traité: {'PASS' if passed else 'FAIL'}")
            
        except Exception as e:
            print(f"\n❌ Erreur lors du traitement: {e}")
            results.append({
                'input': input_file,
                'error': str(e),
                'passed': False
            })
    
    # Résumé du batch
    print(f"\n{'='*50}")
    print("RÉSUMÉ DU BATCH")
    print(f"{'='*50}")
    
    passed_count = sum(1 for r in results if r.get('passed', False))
    failed_count = len(results) - passed_count
    
    print(f"Total fichiers: {len(results)}")
    print(f"✅ Succès: {passed_count}")
    print(f"❌ Échecs: {failed_count}")
    
    if failed_count > 0:
        print("\nFichiers en échec:")
        for r in results:
            if not r.get('passed', False):
                print(f"  • {os.path.basename(r['input'])}: {r.get('error', 'Unknown error')}")
    
    return results

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple 1: Traitement d'un seul fichier
    # mesh, stats, passed = full_cfd_pipeline_robust(
    #     input_path="patient_scan.nii.gz",
    #     output_path="patient_cfd_mesh.stl",
    #     factor=2,
    #     save_intermediate=True
    # )
    
    # Exemple 2: Traitement batch
    # input_files = [
    #     "data/scan1.nii.gz",
    #     "data/scan2.nii.gz",
    #     "data/scan3.nii.gz"
    # ]
    # results = batch_process_pipeline(
    #     input_files=input_files,
    #     output_dir="output_meshes",
    #     factor=2,
    #     save_intermediate=False
    # )
    
    print("🎯 Pipeline CFD prêt à l'emploi")
    print("Utilisation:")
    print("  full_cfd_pipeline_robust('input.nii.gz', 'output.stl', factor=2)")