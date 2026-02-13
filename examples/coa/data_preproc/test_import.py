#!/usr/bin/env python3
"""
Script d'extraction et visualisation STL pour TBAD
=====================================================
Utilise le convertisseur NIfTI → STL propre avec visualisation interactive.

Usage:
    python extract_and_visualize_tbad.py [--tl-label 1] [--fl-label 2] [--refine 2] [--viz]
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import trimesh
import pyvista as pv
import kagglehub

# Import du convertisseur propre
from nifti_to_stl_converter import NiftiToSTLConverter, SurfaceParameters

# ============================================================================
# CONSTANTES
# ============================================================================

# Paramètres par défaut optimisés pour TBAD
DEFAULT_TL_LABEL = 1
DEFAULT_FL_LABEL = 2
DEFAULT_REFINE_FACTOR = 2  # Équilibre qualité/performance
DEFAULT_SDF_SIGMA = 0.25   # Validé sur cohorte TBAD
DEFAULT_TARGET_TRIANGLES_TL = 200_000  # Suffisant pour visualisation
DEFAULT_TARGET_TRIANGLES_FL = 150_000
DEFAULT_MIN_PATCH_DIAMETER = 4.0  # mm

# Conversion unités
MM_TO_M = 0.001

# ============================================================================
# FONCTIONS DE RECHERCHE DE FICHIERS
# ============================================================================

def find_tbad_nifti(base_path: Optional[Path] = None) -> Path:
    """
    Recherche automatique d'un fichier NIfTI TBAD.
    
    Stratégie:
    1. Cherche spécifiquement 126_label.nii.gz
    2. Cherche *label*.nii.gz
    3. Propose de télécharger depuis Kaggle
    """
    if base_path is None:
        base_path = Path.cwd()
    
    # Essai 1: Fichier spécifique 126
    specific_path = base_path.parent / "imageTBAD" / "126_label.nii.gz"
    if specific_path.exists():
        print(f"✅ Fichier trouvé: {specific_path}")
        return specific_path
    
    # Essai 2: Recherche récursive du pattern 126_label
    nifti_files = list(base_path.parent.rglob("*126_label*.nii*"))
    if nifti_files:
        print(f"✅ Fichier trouvé: {nifti_files[0]}")
        return nifti_files[0]
    
    # Essai 3: Recherche de tout fichier label
    nifti_files = list(base_path.parent.rglob("*label*.nii*"))
    if nifti_files:
        print(f"⚠️  Utilisation de: {nifti_files[0]} (pas le 126 spécifique)")
        return nifti_files[0]
    
    # Essai 4: Recherche dans le répertoire courant
    nifti_files = list(base_path.glob("*.nii*"))
    if nifti_files:
        print(f"⚠️  Utilisation de: {nifti_files[0]} (répertoire courant)")
        return nifti_files[0]
    
    # Rien trouvé → proposer téléchargement
    print("\n❌ Aucun fichier NIfTI trouvé.")
    response = input("Voulez-vous télécharger le dataset TBAD depuis Kaggle? (o/N): ")
    
    if response.lower() in ['o', 'oui', 'y', 'yes']:
        print("📥 Téléchargement depuis Kaggle...")
        try:
            path = kagglehub.dataset_download("xiaoweixumedicalai/imagetbad")
            print(f"✅ Dataset téléchargé dans: {path}")
            print("⚠️  Veuillez décompresser l'archive et réexécuter le script.")
        except Exception as e:
            print(f"❌ Erreur téléchargement: {e}")
    
    raise FileNotFoundError(
        "Aucun fichier NIfTI trouvé. Placez un fichier .nii.gz dans le répertoire "
        "ou téléchargez depuis https://www.kaggle.com/datasets/xiaoweixumedicalai/imagetbad"
    )

# ============================================================================
# VISUALISATEUR TBAD
# ============================================================================

class TbadVisualizer:
    """Visualisation interactive des STL TBAD."""
    
    def __init__(self):
        # Configuration PyVista
        pv.set_plot_theme('document')
        pv.OFF_SCREEN = False  # Mode interactif
        
        # Couleurs thématiques
        self.colors = {
            'tl': '#1E88E5',      # Bleu électrique
            'fl': '#DC143C',      # Rouge cramoisi
            'patch': '#2E7D32',   # Vert forêt
            'background': 'white',
            'text': 'black'
        }
    
    def visualize_pair(
        self,
        tl_stl: Path,
        fl_stl: Path,
        tl_patches: Optional[list] = None,
        fl_patches: Optional[list] = None,
        title: str = "TBAD - True Lumen + False Lumen"
    ):
        """
        Visualisation interactive côte à côte.
        
        Args:
            tl_stl: Fichier STL du True Lumen
            fl_stl: Fichier STL du False Lumen
            tl_patches: Liste des patches TL
            fl_patches: Liste des patches FL
            title: Titre de la fenêtre
        """
        print("\n🎬 Lancement de la visualisation interactive...")
        print("   Contrôles: Souris pour rotation, Molette zoom, 'q' pour quitter")
        
        # Chargement des maillages
        mesh_tl = trimesh.load(tl_stl)
        mesh_fl = trimesh.load(fl_stl)
        
        # Conversion PyVista et mise à l'échelle mm → m
        pv_tl = pv.wrap(mesh_tl)
        pv_fl = pv.wrap(mesh_fl)
        pv_tl.points *= MM_TO_M
        pv_fl.points *= MM_TO_M
        
        # Création du plotter
        plotter = pv.Plotter(
            title=title,
            window_size=[1400, 900],
            lighting='three lights'
        )
        
        # Ajout TL
        plotter.add_mesh(
            pv_tl,
            color=self.colors['tl'],
            opacity=0.7,
            smooth_shading=True,
            label=f"True Lumen ({len(mesh_tl.faces):,} faces)",
            lighting=True,
            ambient=0.2,
            specular=0.3
        )
        
        # Ajout FL
        plotter.add_mesh(
            pv_fl,
            color=self.colors['fl'],
            opacity=0.5,
            smooth_shading=True,
            label=f"False Lumen ({len(mesh_fl.faces):,} faces)",
            lighting=True,
            ambient=0.2,
            specular=0.3
        )
        
        # Ajout patches TL
        if tl_patches:
            for i, patch_path in enumerate(tl_patches):
                if Path(patch_path).exists():
                    patch_mesh = trimesh.load(patch_path)
                    pv_patch = pv.wrap(patch_mesh)
                    pv_patch.points *= MM_TO_M
                    
                    # Opacité et couleur selon le type
                    if i == 0:  # Inlet présumé
                        color = '#2E7D32'  # Vert
                        label = f"TL Inlet"
                    else:  # Outlets
                        color = '#FFA000'  # Orange
                        label = f"TL Outlet {i}"
                    
                    plotter.add_mesh(
                        pv_patch,
                        color=color,
                        opacity=0.9,
                        show_edges=True,
                        edge_color='black',
                        line_width=1,
                        label=label
                    )
        
        # Ajout patches FL
        if fl_patches:
            for i, patch_path in enumerate(fl_patches):
                if Path(patch_path).exists():
                    patch_mesh = trimesh.load(patch_path)
                    pv_patch = pv.wrap(patch_mesh)
                    pv_patch.points *= MM_TO_M
                    
                    color = '#9C27B0' if i == 0 else '#FF9800'  # Violet ou orange
                    label = f"FL {'Inlet' if i == 0 else f'Outlet {i}'}"
                    
                    plotter.add_mesh(
                        pv_patch,
                        color=color,
                        opacity=0.9,
                        show_edges=True,
                        edge_color='black',
                        line_width=1,
                        label=label
                    )
        
        # Légende
        plotter.add_legend(
            face='line',
            font_size=10,
            bcolor='white',
            border=True
        )
        
        # Titre et axes
        plotter.add_text(
            title,
            font_size=14,
            position='upper_edge',
            color=self.colors['text']
        )
        plotter.add_axes()
        
        # Statistiques
        stats_text = (
            f"TL: {len(mesh_tl.vertices):,} vertices, {len(mesh_tl.faces):,} faces\n"
            f"FL: {len(mesh_fl.vertices):,} vertices, {len(mesh_fl.faces):,} faces\n"
            f"Patches: {len(tl_patches or []) + len(fl_patches or [])}"
        )
        plotter.add_text(
            stats_text,
            font_size=10,
            position='lower_edge',
            color=self.colors['text']
        )
        
        # Affichage
        plotter.show()
        plotter.close()
    
    def visualize_single(
        self,
        stl_path: Path,
        color: str = "#1E88E5",
        title: str = "Visualisation STL",
        patches: Optional[list] = None
    ):
        """Visualisation d'un seul maillage."""
        mesh = trimesh.load(stl_path)
        pv_mesh = pv.wrap(mesh)
        pv_mesh.points *= MM_TO_M
        
        plotter = pv.Plotter(title=title, window_size=[1200, 800])
        plotter.add_mesh(
            pv_mesh,
            color=color,
            opacity=0.8,
            smooth_shading=True,
            label=f"{stl_path.name} ({len(mesh.faces):,} faces)"
        )
        
        if patches:
            for i, patch_path in enumerate(patches):
                if Path(patch_path).exists():
                    patch = pv.wrap(trimesh.load(patch_path))
                    patch.points *= MM_TO_M
                    plotter.add_mesh(
                        patch,
                        color='#2E7D32',
                        opacity=0.9,
                        show_edges=True,
                        label=f"Patch {i}"
                    )
        
        plotter.add_legend()
        plotter.add_text(title, font_size=14, position='upper_edge')
        plotter.add_axes()
        plotter.show()
        plotter.close()

# ============================================================================
# PIPELINE D'EXTRACTION LÉGER
# ============================================================================

class TbadExtractor:
    """
    Pipeline léger d'extraction STL pour TBAD.
    Version simplifiée sans dépendances CFD.
    """
    
    def __init__(
        self,
        output_dir: str = "tbad_stl_output",
        refine_factor: int = DEFAULT_REFINE_FACTOR,
        sdf_sigma: float = DEFAULT_SDF_SIGMA,
        target_triangles_tl: int = DEFAULT_TARGET_TRIANGLES_TL,
        target_triangles_fl: int = DEFAULT_TARGET_TRIANGLES_FL,
        extract_patches: bool = True,
        verbose: bool = True
    ):
        """
        Initialise l'extracteur TBAD.
        
        Args:
            output_dir: Dossier de sortie
            refine_factor: Facteur de raffinement
            sdf_sigma: Sigma lissage SDF
            target_triangles_tl: Triangles cible TL
            target_triangles_fl: Triangles cible FL
            extract_patches: Détecter les patches
            verbose: Mode verbeux
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.extract_patches = extract_patches
        self.verbose = verbose
        
        # Paramètres pour TL
        self.params_tl = SurfaceParameters(
            refine_factor=refine_factor,
            sdf_sigma_mm=sdf_sigma,
            target_triangles=target_triangles_tl if target_triangles_tl > 0 else None,
            min_patch_diameter_mm=DEFAULT_MIN_PATCH_DIAMETER
        )
        
        # Paramètres pour FL (moins de triangles)
        self.params_fl = SurfaceParameters(
            refine_factor=refine_factor,
            sdf_sigma_mm=sdf_sigma,
            target_triangles=target_triangles_fl if target_triangles_fl > 0 else None,
            min_patch_diameter_mm=DEFAULT_MIN_PATCH_DIAMETER
        )
        
        # Convertisseurs
        self.converter_tl = NiftiToSTLConverter(
            params=self.params_tl,
            verbose=verbose
        )
        
        self.converter_fl = NiftiToSTLConverter(
            params=self.params_fl,
            verbose=verbose
        )
    
    def extract(
        self,
        nifti_path: Path,
        tl_label: int = DEFAULT_TL_LABEL,
        fl_label: int = DEFAULT_FL_LABEL
    ) -> dict:
        """
        Extrait les STL TL et FL.
        
        Returns:
            Dict avec chemins des fichiers générés
        """
        print("\n" + "=" * 60)
        print("🏥 EXTRACTION STL POUR TBAD")
        print("=" * 60)
        print(f"📁 Source: {nifti_path.name}")
        print(f"📁 Output: {self.output_dir}")
        print(f"🔍 Raffinement: {self.params_tl.refine_factor}x")
        print(f"🎯 Triangles TL: {self.params_tl.target_triangles or 'auto'}")
        print(f"🎯 Triangles FL: {self.params_fl.target_triangles or 'auto'}")
        print(f"📍 Patches: {'Oui' if self.extract_patches else 'Non'}")
        
        results = {}
        
        # --------------------------------------------------------------------
        # Extraction True Lumen
        # --------------------------------------------------------------------
        print("\n" + "🔵" * 40)
        print("🔵 TRUE LUMEN (TL) - Extraction")
        print("🔵" * 40)
        
        tl_stl = self.output_dir / "tbad_TL_walls.stl"
        
        result_tl = self.converter_tl.convert(
            nifti_path=nifti_path,
            label_value=tl_label,
            output_stl=tl_stl,
            extract_patches=self.extract_patches,
            target_triangles=self.params_tl.target_triangles
        )
        
        results['tl'] = {
            'walls': tl_stl,
            'mesh': result_tl.mesh,
            'stats': result_tl.stats,
            'patches': []
        }
        
        # Sauvegarde des patches TL
        if self.extract_patches and result_tl.patches:
            for i, patch in enumerate(result_tl.patches):
                patch_name = f"tbad_TL_patch_{i}.stl"
                patch_path = self.output_dir / patch_name
                
                if patch['mesh']:
                    patch['mesh'].export(patch_path)
                    results['tl']['patches'].append(str(patch_path))
                    
                    # Métadonnées du patch
                    patch_meta = patch_path.with_suffix('.json')
                    with open(patch_meta, 'w') as f:
                        import json
                        json.dump({
                            'diameter_mm': patch['diameter_mm'],
                            'area_mm2': patch['area_mm2'],
                            'circularity': patch['circularity'],
                            'normal': patch['normal'],
                            'center': patch['center']
                        }, f, indent=2)
        
        # --------------------------------------------------------------------
        # Extraction False Lumen
        # --------------------------------------------------------------------
        print("\n" + "🔴" * 40)
        print("🔴 FALSE LUMEN (FL) - Extraction")
        print("🔴" * 40)
        
        fl_stl = self.output_dir / "tbad_FL_walls.stl"
        
        result_fl = self.converter_fl.convert(
            nifti_path=nifti_path,
            label_value=fl_label,
            output_stl=fl_stl,
            extract_patches=self.extract_patches,
            target_triangles=self.params_fl.target_triangles
        )
        
        results['fl'] = {
            'walls': fl_stl,
            'mesh': result_fl.mesh,
            'stats': result_fl.stats,
            'patches': []
        }
        
        # Sauvegarde des patches FL
        if self.extract_patches and result_fl.patches:
            for i, patch in enumerate(result_fl.patches):
                patch_name = f"tbad_FL_patch_{i}.stl"
                patch_path = self.output_dir / patch_name
                
                if patch['mesh']:
                    patch['mesh'].export(patch_path)
                    results['fl']['patches'].append(str(patch_path))
        
        # --------------------------------------------------------------------
        # Rapport sommaire
        # --------------------------------------------------------------------
        print("\n" + "=" * 40)
        print("📊 RAPPORT D'EXTRACTION")
        print("=" * 40)
        
        print(f"\n🔵 TRUE LUMEN:")
        print(f"   ├─ STL: {tl_stl.name}")
        print(f"   ├─ Vertices: {len(result_tl.mesh.vertices):,}")
        print(f"   ├─ Faces: {len(result_tl.mesh.faces):,}")
        print(f"   ├─ Volume: {result_tl.mesh.volume * 1000:.1f} mm³")
        print(f"   ├─ Étanche: {result_tl.mesh.is_watertight}")
        print(f"   └─ Patches: {len(results['tl']['patches'])}")
        
        print(f"\n🔴 FALSE LUMEN:")
        print(f"   ├─ STL: {fl_stl.name}")
        print(f"   ├─ Vertices: {len(result_fl.mesh.vertices):,}")
        print(f"   ├─ Faces: {len(result_fl.mesh.faces):,}")
        print(f"   ├─ Volume: {result_fl.mesh.volume * 1000:.1f} mm³")
        print(f"   ├─ Étanche: {result_fl.mesh.is_watertight}")
        print(f"   └─ Patches: {len(results['fl']['patches'])}")
        
        # Sauvegarde du rapport global
        report_path = self.output_dir / "extraction_report.json"
        with open(report_path, 'w') as f:
            import json
            json.dump({
                'timestamp': np.datetime64('now').astype(str),
                'nifti_file': str(nifti_path.name),
                'parameters': {
                    'refine_factor': self.params_tl.refine_factor,
                    'sdf_sigma_mm': self.params_tl.sdf_sigma_mm,
                    'target_triangles_tl': self.params_tl.target_triangles,
                    'target_triangles_fl': self.params_fl.target_triangles,
                    'extract_patches': self.extract_patches
                },
                'tl': {
                    'walls': tl_stl.name,
                    'patches': [Path(p).name for p in results['tl']['patches']],
                    'stats': results['tl']['stats']
                },
                'fl': {
                    'walls': fl_stl.name,
                    'patches': [Path(p).name for p in results['fl']['patches']],
                    'stats': results['fl']['stats']
                }
            }, f, indent=2)
        
        print(f"\n📋 Rapport: {report_path.name}")
        print("\n✅ Extraction terminée!")
        
        return results

# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Extraction et visualisation STL pour TBAD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Extraction + visualisation interactive
  python extract_and_visualize_tbad.py --viz

  # Extraction seule avec paramètres personnalisés
  python extract_and_visualize_tbad.py --refine 3 --target-tl 300000 --no-viz

  # Visualisation seulement de fichiers existants
  python extract_and_visualize_tbad.py --viz-only --tl-stl tbad_TL_walls.stl --fl-stl tbad_FL_walls.stl

  # Avec téléchargement automatique
  python extract_and_visualize_tbad.py --download
        """
    )
    
    # Groupe: fichier d'entrée
    input_group = parser.add_argument_group("Entrée")
    input_group.add_argument(
        "nifti",
        nargs="?",
        help="Fichier NIfTI (optionnel, recherche auto sinon)"
    )
    input_group.add_argument(
        "--download",
        action="store_true",
        help="Télécharger le dataset TBAD depuis Kaggle"
    )
    
    # Groupe: extraction
    extract_group = parser.add_argument_group("Extraction")
    extract_group.add_argument(
        "-o", "--output",
        default="tbad_stl_output",
        help="Dossier de sortie (défaut: tbad_stl_output)"
    )
    extract_group.add_argument(
        "--tl-label",
        type=int,
        default=DEFAULT_TL_LABEL,
        help=f"Label True Lumen (défaut: {DEFAULT_TL_LABEL})"
    )
    extract_group.add_argument(
        "--fl-label",
        type=int,
        default=DEFAULT_FL_LABEL,
        help=f"Label False Lumen (défaut: {DEFAULT_FL_LABEL})"
    )
    extract_group.add_argument(
        "-r", "--refine",
        type=int,
        default=DEFAULT_REFINE_FACTOR,
        help=f"Facteur raffinement (défaut: {DEFAULT_REFINE_FACTOR})"
    )
    extract_group.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SDF_SIGMA,
        help=f"Sigma lissage SDF (défaut: {DEFAULT_SDF_SIGMA})"
    )
    extract_group.add_argument(
        "--target-tl",
        type=int,
        default=DEFAULT_TARGET_TRIANGLES_TL,
        help=f"Triangles cible TL (défaut: {DEFAULT_TARGET_TRIANGLES_TL})"
    )
    extract_group.add_argument(
        "--target-fl",
        type=int,
        default=DEFAULT_TARGET_TRIANGLES_FL,
        help=f"Triangles cible FL (défaut: {DEFAULT_TARGET_TRIANGLES_FL})"
    )
    extract_group.add_argument(
        "--no-patches",
        action="store_true",
        help="Ne pas détecter les patches"
    )
    extract_group.add_argument(
        "--no-extract",
        action="store_true",
        help="Ne pas extraire (visualisation seulement)"
    )
    
    # Groupe: visualisation
    viz_group = parser.add_argument_group("Visualisation")
    viz_group.add_argument(
        "--viz",
        action="store_true",
        help="Lancer la visualisation interactive après extraction"
    )
    viz_group.add_argument(
        "--viz-only",
        action="store_true",
        help="Visualisation seulement (sans extraction)"
    )
    viz_group.add_argument(
        "--tl-stl",
        help="Fichier STL TL pour visualisation seule"
    )
    viz_group.add_argument(
        "--fl-stl",
        help="Fichier STL FL pour visualisation seule"
    )
    viz_group.add_argument(
        "--single",
        help="Visualiser un seul fichier STL"
    )
    
    # Groupe: divers
    misc_group = parser.add_argument_group("Divers")
    misc_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Mode silencieux"
    )
    misc_group.add_argument(
        "--no-warn",
        action="store_true",
        help="Supprimer les warnings"
    )
    
    args = parser.parse_args()
    
    # Suppression warnings
    if args.no_warn:
        import warnings
        warnings.filterwarnings("ignore")
    
    # ========================================================================
    # MODE VISUALISATION SEULE
    # ========================================================================
    if args.viz_only:
        if not args.tl_stl and not args.fl_stl and not args.single:
            print("❌ Mode visualisation: spécifiez --tl-stl et --fl-stl ou --single")
            return 1
        
        viz = TbadVisualizer()
        
        if args.single:
            viz.visualize_single(Path(args.single))
        else:
            tl_patches = list(Path(args.tl_stl).parent.glob("tbad_TL_patch_*.stl")) if args.tl_stl else []
            fl_patches = list(Path(args.fl_stl).parent.glob("tbad_FL_patch_*.stl")) if args.fl_stl else []
            
            viz.visualize_pair(
                tl_stl=Path(args.tl_stl),
                fl_stl=Path(args.fl_stl),
                tl_patches=tl_patches,
                fl_patches=fl_patches
            )
        return 0
    
    # ========================================================================
    # MODE EXTRACTION
    # ========================================================================
    
    # Téléchargement si demandé
    if args.download:
        print("📥 Téléchargement du dataset TBAD...")
        try:
            path = kagglehub.dataset_download("xiaoweixumedicalai/imagetbad")
            print(f"✅ Dataset téléchargé dans: {path}")
            print("⚠️  Veuillez décompresser l'archive et réexécuter le script avec le chemin du fichier .nii.gz")
            return 0
        except Exception as e:
            print(f"❌ Erreur téléchargement: {e}")
            return 1
    
    # Recherche du fichier NIfTI
    if args.nifti:
        nifti_path = Path(args.nifti)
        if not nifti_path.exists():
            print(f"❌ Fichier non trouvé: {nifti_path}")
            return 1
    else:
        try:
            nifti_path = find_tbad_nifti()
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return 1
    
    # Extraction
    if not args.no_extract:
        extractor = TbadExtractor(
            output_dir=args.output,
            refine_factor=args.refine,
            sdf_sigma=args.sigma,
            target_triangles_tl=args.target_tl,
            target_triangles_fl=args.target_fl,
            extract_patches=not args.no_patches,
            verbose=not args.quiet
        )
        
        results = extractor.extract(
            nifti_path=nifti_path,
            tl_label=args.tl_label,
            fl_label=args.fl_label
        )
    else:
        # Utiliser des fichiers existants
        results = {
            'tl': {
                'walls': Path(args.output) / "tbad_TL_walls.stl",
                'patches': list(Path(args.output).glob("tbad_TL_patch_*.stl"))
            },
            'fl': {
                'walls': Path(args.output) / "tbad_FL_walls.stl",
                'patches': list(Path(args.output).glob("tbad_FL_patch_*.stl"))
            }
        }
        
        # Vérification
        if not results['tl']['walls'].exists() or not results['fl']['walls'].exists():
            print("❌ Fichiers STL existants non trouvés")
            return 1
    
    # ========================================================================
    # VISUALISATION
    # ========================================================================
    
    if args.viz:
        viz = TbadVisualizer()
        
        viz.visualize_pair(
            tl_stl=results['tl']['walls'],
            fl_stl=results['fl']['walls'],
            tl_patches=results['tl']['patches'],
            fl_patches=results['fl']['patches']
        )
    
    return 0

# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == "__main__":
    # Fix seed pour reproductibilité
    np.random.seed(42)
    
    # Exécution
    sys.exit(main())