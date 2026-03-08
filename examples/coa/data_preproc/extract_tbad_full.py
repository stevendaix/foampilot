#!/usr/bin/env python3
"""
Extracteur TBAD complet
=======================
Extrait True Lumen, False Lumen et Paroi Externe depuis les fichiers NIfTI.

Usage:
    python extract_tbad_full.py <patient_id> [OPTIONS]
    
Exemples:
    python extract_tbad_full.py 58                    # Patient 58
    python extract_tbad_full.py 58 -o output          # Output personnalisé
    python extract_tbad_full.py 58 --viz             # Visualisation
"""

import argparse
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import json
import time

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_fill_holes
from skimage.measure import marching_cubes
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TBADExtractorConfig:
    """Configuration pour l'extraction TBAD."""
    # Seuils CT (unités Hounsfield)
    tissue_threshold: int = 50           # Seuil pour tissu aortique
    lumen_dilation: int = 5              # Itérations dilation lumen
    
    # Paramètres STL
    target_triangles_tl: int = 50000
    target_triangles_fl: int = 30000
    
    # Répertoires
    data_dir: str = "imageTBAD"
    output_dir: str = "tbad_output"


# ============================================================================
# EXTRACTEUR LUMENS
# ============================================================================

def extract_lumens(nifti_path: Path, output_dir: Path, 
                   target_tl: int = 50000, target_fl: int = 30000,
                   verbose: bool = True) -> dict:
    """
    Extrait True Lumen et False Lumen depuis un label NIfTI.
    Utilise le convertisseur existant (test_import.py).
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from test_import import TbadExtractor, TBADConfig
    
    config = TBADConfig(
        target_triangles_tl=target_tl,
        target_triangles_fl=target_fl
    )
    
    extractor = TbadExtractor(
        output_dir=output_dir,
        extract_patches=False,  # Désactivé (bugs API)
        verbose=verbose
    )
    extractor.config = config
    extractor._setup_converters()
    
    result = extractor.extract(nifti_path, tl_label=1, fl_label=2)
    
    return {
        "success": result.success,
        "tl_path": result.tl_stl_path,
        "fl_path": result.fl_stl_path,
        "tl_stats": result.tl_stats.to_dict() if result.tl_stats else None,
        "fl_stats": result.fl_stats.to_dict() if result.fl_stats else None,
    }


# ============================================================================
# EXTRACTEUR PAROI EXTERNE
# ============================================================================

def extract_aorta_wall(image_path: Path, label_path: Path, 
                       output_path: Path, 
                       tissue_threshold: int = 50,
                       dilation_iterations: int = 5) -> dict:
    """
    Extrait la paroi externe de l'aorte depuis l'image CT.
    
    Args:
        image_path: Chemin vers l'image NIfTI
        label_path: Chemin vers le label NIfTI
        output_path: Chemin pour sauvegarder le STL
        tissue_threshold: Seuil CT pour le tissu (HU)
        dilation_iterations: Itérations de dilation du lumen
    
    Returns:
        Statistiques du maillage extrait
    """
    logger.info(f"📥 Chargement: {image_path.name}")
    
    img = nib.load(image_path)
    label = nib.load(label_path)
    
    img_data = img.get_fdata()
    label_data = label.get_fdata()
    
    # Récupérer ROI depuis le label
    lumen = (label_data == 1) | (label_data == 2)
    coords = np.array(np.where(lumen)).T
    
    if len(coords) == 0:
        raise ValueError("Aucun lumen trouvé dans le label")
    
    # Padding pour inclure la paroi
    pad = 25
    min_z = max(0, coords[:, 0].min() - pad)
    max_z = min(img_data.shape[0], coords[:, 0].max() + pad)
    min_y = max(0, coords[:, 1].min() - pad)
    max_y = min(img_data.shape[1], coords[:, 1].max() + pad)
    min_x = max(0, coords[:, 2].min() - pad)
    max_x = min(img_data.shape[2], coords[:, 2].max() + pad)
    
    logger.info(f"   ROI: z=[{min_z}:{max_z}], y=[{min_y}:{max_y}], x=[{min_x}:{max_x}]")
    
    # Crop ROI
    roi_img = img_data[min_z:max_z, min_y:max_y, min_x:max_x]
    roi_label = label_data[min_z:max_z, min_y:max_y, min_x:max_x]
    
    # Créer masque de paroi
    lumen_mask = (roi_label == 1) | (roi_label == 2)
    lumen_dil = binary_dilation(lumen_mask, iterations=dilation_iterations)
    wall_mask = (roi_img > tissue_threshold) & (~lumen_dil)
    
    # Remplir les trous
    wall_filled = binary_fill_holes(wall_mask)
    
    # Extraction surface
    logger.info(f"   🔨 Extraction surface...")
    verts, faces, normals, values = marching_cubes(
        wall_filled.astype(float), 
        level=0.5, 
        spacing=(1, 1, 1)
    )
    
    # Créer et sauvegarder le maillage
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(output_path)
    
    stats = {
        "vertices": len(verts),
        "faces": len(faces),
        "volume_mm3": float(mesh.volume),
        "surface_area_mm2": float(mesh.area),
        "is_watertight": mesh.is_watertight,
    }
    
    logger.info(f"   ✅ Paroi extraite: {len(verts):,} vertices, {len(faces):,} faces")
    
    return stats


# ============================================================================
# VISUALISATION
# ============================================================================

def visualize_full(tl_path: Path, fl_path: Path, wall_path: Path, 
                  output_path: Optional[Path] = None):
    """Visualisation 3D de TL + FL + Paroi."""
    logger.info("🎨 Génération visualisation...")
    
    tl = trimesh.load(tl_path)
    fl = trimesh.load(fl_path)
    wall = trimesh.load(wall_path)
    
    mm_to_cm = 0.1
    
    fig = plt.figure(figsize=(18, 10))
    
    # 4 angles de vue
    angles = [(30, 45), (30, 135), (60, 45), (15, 0)]
    titles = ['Vue 1', 'Vue 2', 'Vue 3 (dessus)', 'Vue 4']
    
    tl_center = tl.vertices.mean(axis=0)
    fl_center = fl.vertices.mean(axis=0)
    wall_center = wall.vertices.mean(axis=0)
    
    for i, ((elev, azim), title) in enumerate(zip(angles, titles)):
        ax = fig.add_subplot(2, 4, i + 1, projection='3d')
        
        # Centrer et convertir
        tl_v = (tl.vertices - tl_center) * mm_to_cm
        fl_v = (fl.vertices - fl_center) * mm_to_cm
        wall_v = (wall.vertices - wall_center) * mm_to_cm
        
        # Afficher: paroi (gris transparent), TL (bleu), FL (rouge)
        ax.plot_trisurf(wall_v[:, 0], wall_v[:, 1], wall_v[:, 2],
            triangles=wall.faces, color='gray', alpha=0.3, edgecolor='none')
        ax.plot_trisurf(tl_v[:, 0], tl_v[:, 1], tl_v[:, 2],
            triangles=tl.faces, color='blue', alpha=0.8, edgecolor='none')
        ax.plot_trisurf(fl_v[:, 0], fl_v[:, 1], fl_v[:, 2],
            triangles=fl.faces, color='red', alpha=0.8, edgecolor='none')
        
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
    
    # Panneau statistiques
    ax_stats = fig.add_subplot(2, 4, (5, 8))
    ax_stats.axis('off')
    
    stats_text = f"""
    TBAD - Résumé
    ════════════════════════════
    
    [Gris]  Paroi externe: {len(wall.faces):,} faces
    
    [Bleu]  True Lumen:    {len(tl.faces):,} faces
    
    [Rouge] False Lumen:   {len(fl.faces):,} faces
    
    ════════════════════════════
    Bleu + Rouge = à l'intérieur
    du gris (paroi externe)
    """
    
    ax_stats.text(0.2, 0.5, stats_text, transform=ax_stats.transAxes,
        fontsize=11, verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('TBAD - True Lumen + False Lumen + Paroi Externe', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"💾 Image sauvegardée: {output_path}")
    else:
        plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extracteur TBAD complet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python extract_tbad_full.py 58
  python extract_tbad_full.py 58 -o output --viz
        """
    )
    
    parser.add_argument("patient_id", type=int, help="ID du patient (ex: 58)")
    parser.add_argument("-o", "--output", default="tbad_output", help="Répertoire de sortie")
    parser.add_argument("--viz", action="store_true", help="Générer visualisation")
    parser.add_argument("--no-lumens", action="store_true", help="Skip extraction lumens")
    parser.add_argument("--no-wall", action="store_true", help="Skip extraction paroi")
    parser.add_argument("--target-tl", type=int, default=50000, help="Triangles TL")
    parser.add_argument("--target-fl", type=int, default=30000, help="Triangles FL")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mode verbeux")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Chemins - imageTBAD est dans examples/coa/
    data_dir = Path(__file__).parent.parent / "imageTBAD"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_path = data_dir / f"{args.patient_id}_image.nii.gz"
    label_path = data_dir / f"{args.patient_id}_label.nii.gz"
    
    if not image_path.exists():
        logger.error(f"❌ Image non trouvée: {image_path}")
        return 1
    
    if not label_path.exists():
        logger.error(f"❌ Label non trouvé: {label_path}")
        return 1
    
    start_time = time.time()
    results = {"patient_id": args.patient_id}
    
    # Extraction lumens
    if not args.no_lumens:
        logger.info(f"\n{'='*50}")
        logger.info(f"🫀 Extraction Lumens (Patient {args.patient_id})")
        logger.info(f"{'='*50}")
        
        lumens_result = extract_lumens(
            label_path, 
            output_dir,
            target_tl=args.target_tl,
            target_fl=args.target_fl,
            verbose=args.verbose
        )
        
        results["lumens"] = lumens_result
        
        if not lumens_result["success"]:
            logger.error("❌ Échec extraction lumens")
            return 1
    
    # Extraction paroi
    if not args.no_wall:
        logger.info(f"\n{'='*50}")
        logger.info(f"🧱 Extraction Paroi Externe")
        logger.info(f"{'='*50}")
        
        wall_path = output_dir / f"patient{args.patient_id}_wall.stl"
        
        try:
            wall_stats = extract_aorta_wall(
                image_path, 
                label_path,
                wall_path,
                tissue_threshold=50,
                dilation_iterations=5
            )
            results["wall"] = wall_stats
        except Exception as e:
            logger.error(f"❌ Échec extraction paroi: {e}")
            results["wall"] = {"error": str(e)}
    
    # Visualisation
    if args.viz and not args.no_lumens and not args.no_wall:
        logger.info(f"\n{'='*50}")
        logger.info(f"🎨 Génération visualisation")
        logger.info(f"{'='*50}")
        
        tl_path = output_dir / f"patient{args.patient_id}_TL_walls.stl"
        fl_path = output_dir / f"patient{args.patient_id}_FL_walls.stl"
        
        # Fallback si les noms sont différents
        if not tl_path.exists():
            tl_path = output_dir / "tbad_TL_walls.stl"
        if not fl_path.exists():
            fl_path = output_dir / "tbad_FL_walls.stl"
        
        if tl_path.exists() and fl_path.exists() and wall_path.exists():
            visualize_full(tl_path, fl_path, wall_path,
                         output_path=output_dir / f"patient{args.patient_id}_full.png")
        else:
            logger.warning("⚠️ Fichiers STL non trouvés pour visualisation")
    
    # Sauvegarder rapport (convertir paths en strings)
    def convert_path(p):
        return str(p) if isinstance(p, Path) else p
    
    # Convert paths in results
    if "lumens" in results and "tl_path" in results["lumens"]:
        results["lumens"]["tl_path"] = convert_path(results["lumens"]["tl_path"])
        results["lumens"]["fl_path"] = convert_path(results["lumens"]["fl_path"])
    
    report_path = output_dir / f"patient{args.patient_id}_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    elapsed = time.time() - start_time
    logger.info(f"\n✅ Terminé en {elapsed:.1f}s")
    logger.info(f"📁 Résultats dans: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
