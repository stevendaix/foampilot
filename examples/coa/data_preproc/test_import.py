#!/usr/bin/env python3
"""
Script d'extraction et visualisation STL pour TBAD (Type B Aortic Dissection)
=============================================================================
Utilise le convertisseur NIfTI → STL avec visualisation interactive PyVista.

Fonctionnalités:
- Extraction automatique des True/False Lumens depuis des labels NIfTI
- Détection et export des patches (conditions limites CFD)
- Visualisation 3D interactive avec coloration thématique
- Téléchargement automatique depuis Kaggle si nécessaire
- Rapport JSON pour traçabilité et reproductibilité

Usage:
    python extract_and_visualize_tbad.py [OPTIONS]
    
Exemples:
    # Extraction + visualisation interactive
    python extract_and_visualize_tbad.py --viz
    
    # Avec paramètres personnalisés
    python extract_and_visualize_tbad.py --refine 3 --target-tl 300000 --viz
    
    # Visualisation seule de fichiers existants
    python extract_and_visualize_tbad.py --viz-only --tl-stl output/tbad_TL_walls.stl
    
    # Téléchargement du dataset
    python extract_and_visualize_tbad.py --download
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import datetime

import numpy as np
import trimesh
import pyvista as pv

# Imports optionnels avec gestion d'erreur
try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

try:
    from nifti_to_stl_converter import NiftiToSTLConverter, SurfaceParameters, ConversionResult
    CONVERTER_AVAILABLE = True
except ImportError:
    CONVERTER_AVAILABLE = False
    warnings.warn(
        "⚠️  nifti_to_stl_converter non trouvé. L'extraction ne fonctionnera pas.\n"
        "Installez-le avec: pip install -e .",
        ImportWarning,
        stacklevel=2
    )

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES ET CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class TBADConfig:
    """Configuration par défaut optimisée pour TBAD."""
    
    # Labels NIfTI
    tl_label: int = 1
    fl_label: int = 2
    
    # Paramètres d'extraction
    refine_factor: int = 2
    sdf_sigma_mm: float = 0.25
    target_triangles_tl: int = 200_000
    target_triangles_fl: int = 150_000
    min_patch_diameter_mm: float = 4.0
    
    # Unités
    mm_to_m: float = 0.001
    
    # Visualisation
    window_size: tuple[int, int] = (1400, 900)
    tl_color: str = "#1E88E5"      # Bleu électrique
    fl_color: str = "#DC143C"      # Rouge cramoisi
    patch_tl_inlet_color: str = "#2E7D32"   # Vert
    patch_tl_outlet_color: str = "#FFA000"  # Orange
    patch_fl_color: str = "#9C27B0"         # Violet
    
    # Kaggle
    kaggle_dataset: str = "xiaoweixumedicalai/imagetbad"
    
    # Fichiers
    default_output_dir: str = "tbad_stl_output"
    default_nifti_pattern: str = "*126_label*.nii*"


# ============================================================================
# STRUCTURES DE DONNÉES
# ============================================================================

@dataclass
class ExtractionStats:
    """Statistiques d'extraction pour un lumen."""
    vertices: int
    faces: int
    volume_mm3: float
    surface_area_mm2: float
    is_watertight: bool
    patches_count: int
    processing_time_sec: Optional[float] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractionResult:
    """Résultat complet d'une extraction TBAD."""
    success: bool
    tl_stats: Optional[ExtractionStats] = None
    fl_stats: Optional[ExtractionStats] = None
    tl_stl_path: Optional[Path] = None
    fl_stl_path: Optional[Path] = None
    tl_patches: List[Path] = field(default_factory=list)
    fl_patches: List[Path] = field(default_factory=list)
    error_message: Optional[str] = None
    config: Optional[dict] = None
    
    def to_report(self, nifti_path: Path, output_dir: Path) -> dict:
        """Génère un rapport JSON sérialisable."""
        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "nifti_file": nifti_path.name,
                "output_directory": str(output_dir),
                "config": self.config or {}
            },
            "success": self.success,
            "error": self.error_message,
            "true_lumen": {
                "stl_file": self.tl_stl_path.name if self.tl_stl_path else None,
                "stats": self.tl_stats.to_dict() if self.tl_stats else None,
                "patches": [p.name for p in self.tl_patches]
            },
            "false_lumen": {
                "stl_file": self.fl_stl_path.name if self.fl_stl_path else None,
                "stats": self.fl_stats.to_dict() if self.fl_stats else None,
                "patches": [p.name for p in self.fl_patches]
            }
        }


# ============================================================================
# UTILITAIRES DE FICHIERS
# ============================================================================

def find_nifti_file(
    search_path: Optional[Path] = None,
    pattern: str = TBADConfig.default_nifti_pattern,
    specific_name: Optional[str] = "126_label.nii.gz"
) -> Optional[Path]:
    """
    Recherche intelligente d'un fichier NIfTI TBAD.
    
    Stratégie de recherche (par ordre de priorité):
    1. Nom spécifique exact (ex: 126_label.nii.gz)
    2. Pattern avec identifiant patient
    3. Tout fichier *label*.nii*
    4. Tout fichier .nii* dans le répertoire courant
    
    Returns:
        Path du fichier trouvé, ou None si aucun trouvé
    """
    if search_path is None:
        search_path = Path.cwd()
    
    strategies = [
        ("spécifique", search_path.parent / "imageTBAD" / specific_name if specific_name else None),
        ("pattern patient", list(search_path.parent.rglob(pattern)) if pattern else []),
        ("pattern label", list(search_path.parent.rglob("*label*.nii*"))),
        ("courant", list(search_path.glob("*.nii*"))),
    ]
    
    for strategy_name, candidate in strategies:
        if candidate is None:
            continue
            
        if isinstance(candidate, Path):
            if candidate.exists():
                logger.info(f"✅ Fichier trouvé ({strategy_name}): {candidate}")
                return candidate
        elif isinstance(candidate, list) and candidate:
            logger.info(f"✅ Fichier trouvé ({strategy_name}): {candidate[0]}")
            if "pas le spécifique" in strategy_name or "courant" in strategy_name:
                logger.warning(f"⚠️  Utilisation d'un fichier générique: {candidate[0].name}")
            return candidate[0]
    
    return None


def download_tbad_dataset(output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Télécharge le dataset TBAD depuis Kaggle.
    
    Returns:
        Chemin du dataset téléchargé, ou None en cas d'échec
    """
    if not KAGGLE_AVAILABLE:
        logger.error("❌ kagglehub non installé. Exécutez: pip install kagglehub")
        return None
    
    try:
        logger.info(f"📥 Téléchargement du dataset: {TBADConfig.kaggle_dataset}")
        path = Path(kagglehub.dataset_download(TBADConfig.kaggle_dataset))
        logger.info(f"✅ Dataset téléchargé dans: {path}")
        logger.info("⚠️  Décompressez l'archive avant de réexécuter le script")
        return path
    except Exception as e:
        logger.error(f"❌ Erreur de téléchargement: {e}")
        return None


# ============================================================================
# VISUALISATEUR TBAD
# ============================================================================

class TbadVisualizer:
    """Visualisation interactive 3D des maillages STL TBAD avec PyVista."""
    
    def __init__(self, config: TBADConfig = TBADConfig()):
        self.config = config
        self._setup_pyvista()
    
    def _setup_pyvista(self):
        """Configuration de l'environnement PyVista."""
        pv.set_plot_theme('document')
        pv.OFF_SCREEN = False
        pv.global_theme.font.size = 10
        pv.global_theme.font.title_size = 14
    
    def _load_and_scale_mesh(self, stl_path: Path) -> Optional[pv.PolyData]:
        """Charge un STL et convertit mm → m."""
        try:
            mesh = trimesh.load(stl_path, force='mesh')
            if mesh.is_empty:
                logger.error(f"❌ Maillage vide: {stl_path}")
                return None
            
            pv_mesh = pv.wrap(mesh)
            pv_mesh.points *= self.config.mm_to_m
            return pv_mesh
        except Exception as e:
            logger.error(f"❌ Erreur chargement {stl_path}: {e}")
            return None
    
    def _add_mesh_to_plotter(
        self,
        plotter: pv.Plotter,
        mesh: pv.PolyData,
        color: str,
        opacity: float,
        label: str,
        show_edges: bool = False
    ):
        """Ajoute un maillage au plotter avec style cohérent."""
        plotter.add_mesh(
            mesh,
            color=color,
            opacity=opacity,
            smooth_shading=True,
            label=label,
            lighting=True,
            ambient=0.2,
            specular=0.3,
            show_edges=show_edges,
            edge_color='black' if show_edges else None,
            line_width=1 if show_edges else None
        )
    
    def visualize_pair(
        self,
        tl_stl: Path,
        fl_stl: Path,
        tl_patches: Optional[List[Path]] = None,
        fl_patches: Optional[List[Path]] = None,
        title: Optional[str] = None
    ) -> bool:
        """
        Visualisation interactive côte à côte: True Lumen + False Lumen.
        
        Returns:
            True si la visualisation a réussi, False sinon
        """
        if title is None:
            title = "TBAD - True Lumen (bleu) + False Lumen (rouge)"
        
        logger.info(f"🎬 Lancement visualisation: {title}")
        logger.info("   🖱️ Contrôles: rotation (clic+glisser), zoom (molette), quitter (q)")
        
        # Chargement des maillages principaux
        pv_tl = self._load_and_scale_mesh(tl_stl)
        pv_fl = self._load_and_scale_mesh(fl_stl)
        
        if pv_tl is None or pv_fl is None:
            logger.error("❌ Impossible de charger un ou plusieurs maillages principaux")
            return False
        
        # Création du plotter
        plotter = pv.Plotter(
            title=title,
            window_size=self.config.window_size,
            lighting='three lights'
        )
        
        # Ajout True Lumen
        tl_mesh = trimesh.load(tl_stl)
        self._add_mesh_to_plotter(
            plotter, pv_tl, self.config.tl_color, 0.7,
            f"True Lumen ({len(tl_mesh.faces):,} faces)"
        )
        
        # Ajout False Lumen
        fl_mesh = trimesh.load(fl_stl)
        self._add_mesh_to_plotter(
            plotter, pv_fl, self.config.fl_color, 0.5,
            f"False Lumen ({len(fl_mesh.faces):,} faces)"
        )
        
        # Ajout patches TL
        if tl_patches:
            for i, patch_path in enumerate(tl_patches):
                if not patch_path.exists():
                    continue
                pv_patch = self._load_and_scale_mesh(patch_path)
                if pv_patch is None:
                    continue
                    
                color = self.config.patch_tl_inlet_color if i == 0 else self.config.patch_tl_outlet_color
                label = f"TL {'Inlet' if i == 0 else f'Outlet {i}'}"
                
                self._add_mesh_to_plotter(plotter, pv_patch, color, 0.9, label, show_edges=True)
        
        # Ajout patches FL
        if fl_patches:
            for i, patch_path in enumerate(fl_patches):
                if not patch_path.exists():
                    continue
                pv_patch = self._load_and_scale_mesh(patch_path)
                if pv_patch is None:
                    continue
                    
                color = self.config.patch_fl_color if i == 0 else self.config.patch_tl_outlet_color
                label = f"FL {'Inlet' if i == 0 else f'Outlet {i}'}"
                
                self._add_mesh_to_plotter(plotter, pv_patch, color, 0.9, label, show_edges=True)
        
        # Décorations
        plotter.add_legend(face='line', bcolor='white', border=True)
        plotter.add_text(title, position='upper_edge', color='black')
        plotter.add_axes()
        
        # Statistiques en bas
        stats = (
            f"TL: {len(tl_mesh.vertices):,} vtx, {len(tl_mesh.faces):,} faces | "
            f"FL: {len(fl_mesh.vertices):,} vtx, {len(fl_mesh.faces):,} faces | "
            f"Patches: {len(tl_patches or []) + len(fl_patches or [])}"
        )
        plotter.add_text(stats, position='lower_edge', color='black', font_size=9)
        
        # Affichage
        try:
            plotter.show()
            return True
        except Exception as e:
            logger.error(f"❌ Erreur d'affichage: {e}")
            return False
        finally:
            plotter.close()
    
    def visualize_single(
        self,
        stl_path: Path,
        color: Optional[str] = None,
        title: Optional[str] = None,
        patches: Optional[List[Path]] = None
    ) -> bool:
        """Visualisation d'un seul maillage STL."""
        if color is None:
            color = self.config.tl_color
        if title is None:
            title = f"Visualisation: {stl_path.name}"
        
        pv_mesh = self._load_and_scale_mesh(stl_path)
        if pv_mesh is None:
            return False
        
        mesh = trimesh.load(stl_path)
        plotter = pv.Plotter(title=title, window_size=(1200, 800))
        
        self._add_mesh_to_plotter(
            plotter, pv_mesh, color, 0.8,
            f"{stl_path.name} ({len(mesh.faces):,} faces)"
        )
        
        # Patches optionnels
        if patches:
            for i, patch_path in enumerate(patches):
                if not patch_path.exists():
                    continue
                pv_patch = self._load_and_scale_mesh(patch_path)
                if pv_patch:
                    self._add_mesh_to_plotter(
                        plotter, pv_patch, self.config.patch_tl_inlet_color, 0.9,
                        f"Patch {i}", show_edges=True
                    )
        
        plotter.add_legend()
        plotter.add_text(title, position='upper_edge')
        plotter.add_axes()
        
        try:
            plotter.show()
            return True
        finally:
            plotter.close()


# ============================================================================
# EXTRACTEUR TBAD
# ============================================================================

class TbadExtractor:
    """
    Pipeline d'extraction STL pour TBAD.
    
    Extrait les maillages 3D du True Lumen et False Lumen depuis
    des volumes NIfTI segmentés, avec détection automatique des
    patches pour conditions limites CFD.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path] = TBADConfig.default_output_dir,
        config: TBADConfig = TBADConfig(),
        extract_patches: bool = True,
        verbose: bool = True
    ):
        if not CONVERTER_AVAILABLE:
            raise ImportError(
                "nifti_to_stl_converter requis. Installez-le avec: pip install -e ."
            )
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.extract_patches = extract_patches
        self.verbose = verbose
        
        # Configuration des convertisseurs
        self._setup_converters()
    
    def _setup_converters(self):
        """Initialise les convertisseurs pour TL et FL."""
        # Paramètres TL (plus de détails)
        params_tl = SurfaceParameters(
            refine_factor=self.config.refine_factor,
            sdf_sigma_mm=self.config.sdf_sigma_mm,
            target_triangles=self.config.target_triangles_tl if self.config.target_triangles_tl > 0 else None,
            min_patch_diameter_mm=self.config.min_patch_diameter_mm
        )
        
        # Paramètres FL (légèrement simplifié)
        params_fl = SurfaceParameters(
            refine_factor=self.config.refine_factor,
            sdf_sigma_mm=self.config.sdf_sigma_mm,
            target_triangles=self.config.target_triangles_fl if self.config.target_triangles_fl > 0 else None,
            min_patch_diameter_mm=self.config.min_patch_diameter_mm
        )
        
        self.converter_tl = NiftiToSTLConverter(params=params_tl, verbose=self.verbose)
        self.converter_fl = NiftiToSTLConverter(params=params_fl, verbose=self.verbose)
    
    def _extract_stats(self, mesh: trimesh.Trimesh, processing_time: Optional[float]) -> ExtractionStats:
        """Extrait les statistiques d'un maillage."""
        return ExtractionStats(
            vertices=len(mesh.vertices),
            faces=len(mesh.faces),
            volume_mm3=mesh.volume / (self.config.mm_to_m ** 3),  # m³ → mm³
            surface_area_mm2=mesh.area / (self.config.mm_to_m ** 2),  # m² → mm²
            is_watertight=mesh.is_watertight,
            patches_count=0,  # Sera mis à jour après extraction des patches
            processing_time_sec=processing_time
        )
    
    def _save_patch_metadata(self, patch_path: Path, patch_info: dict):
        """Sauvegarde les métadonnées d'un patch au format JSON."""
        meta_path = patch_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'diameter_mm': patch_info.get('diameter_mm'),
                'area_mm2': patch_info.get('area_mm2'),
                'circularity': patch_info.get('circularity'),
                'normal': patch_info.get('normal', [0, 0, 1]),
                'center': patch_info.get('center', [0, 0, 0])
            }, f, indent=2)
    
    def extract(
        self,
        nifti_path: Path,
        tl_label: Optional[int] = None,
        fl_label: Optional[int] = None
    ) -> ExtractionResult:
        """
        Exécute le pipeline complet d'extraction TL + FL.
        
        Args:
            nifti_path: Chemin vers le fichier NIfTI segmenté
            tl_label: Valeur du label True Lumen (défaut: config)
            fl_label: Valeur du label False Lumen (défaut: config)
            
        Returns:
            ExtractionResult avec chemins et statistiques
        """
        import time
        
        tl_label = tl_label or self.config.tl_label
        fl_label = fl_label or self.config.fl_label
        
        logger.info(f"🏥 Démarrage extraction TBAD")
        logger.info(f"📁 Source: {nifti_path.name}")
        logger.info(f"📁 Output: {self.output_dir}")
        
        start_time = time.time()
        result = ExtractionResult(success=False, config=asdict(self.config))
        
        try:
            # ----------------------------------------------------------------
            # Extraction True Lumen
            # ----------------------------------------------------------------
            logger.info(f"\n🔵 TRUE LUMEN (label={tl_label}) - Extraction")
            tl_start = time.time()
            
            tl_stl = self.output_dir / "tbad_TL_walls.stl"
            conv_result_tl: ConversionResult = self.converter_tl.convert(
                nifti_path=nifti_path,
                label_value=tl_label,
                output_stl=tl_stl,
                extract_patches=self.extract_patches,
                target_triangles=self.config.target_triangles_tl
            )
            
            tl_time = time.time() - tl_start
            stats_tl = self._extract_stats(conv_result_tl.mesh, tl_time)
            
            # Sauvegarde patches TL
            tl_patches = []
            if self.extract_patches and conv_result_tl.patches:
                for i, patch in enumerate(conv_result_tl.patches):
                    if not patch.get('mesh'):
                        continue
                    patch_name = f"tbad_TL_patch_{i}.stl"
                    patch_path = self.output_dir / patch_name
                    patch['mesh'].export(patch_path)
                    tl_patches.append(patch_path)
                    self._save_patch_metadata(patch_path, patch)
                    logger.debug(f"   └─ Patch TL #{i}: {patch.get('diameter_mm', 0):.1f} mm")
            
            stats_tl = ExtractionStats(
                **{**asdict(stats_tl), 'patches_count': len(tl_patches)}
            )
            
            # ----------------------------------------------------------------
            # Extraction False Lumen
            # ----------------------------------------------------------------
            logger.info(f"\n🔴 FALSE LUMEN (label={fl_label}) - Extraction")
            fl_start = time.time()
            
            fl_stl = self.output_dir / "tbad_FL_walls.stl"
            conv_result_fl: ConversionResult = self.converter_fl.convert(
                nifti_path=nifti_path,
                label_value=fl_label,
                output_stl=fl_stl,
                extract_patches=self.extract_patches,
                target_triangles=self.config.target_triangles_fl
            )
            
            fl_time = time.time() - fl_start
            stats_fl = self._extract_stats(conv_result_fl.mesh, fl_time)
            
            # Sauvegarde patches FL
            fl_patches = []
            if self.extract_patches and conv_result_fl.patches:
                for i, patch in enumerate(conv_result_fl.patches):
                    if not patch.get('mesh'):
                        continue
                    patch_name = f"tbad_FL_patch_{i}.stl"
                    patch_path = self.output_dir / patch_name
                    patch['mesh'].export(patch_path)
                    fl_patches.append(patch_path)
                    self._save_patch_metadata(patch_path, patch)
                    logger.debug(f"   └─ Patch FL #{i}: {patch.get('diameter_mm', 0):.1f} mm")
            
            stats_fl = ExtractionStats(
                **{**asdict(stats_fl), 'patches_count': len(fl_patches)}
            )
            
            # ----------------------------------------------------------------
            # Résultat final
            # ----------------------------------------------------------------
            total_time = time.time() - start_time
            
            result = ExtractionResult(
                success=True,
                tl_stats=stats_tl,
                fl_stats=stats_fl,
                tl_stl_path=tl_stl,
                fl_stl_path=fl_stl,
                tl_patches=tl_patches,
                fl_patches=fl_patches,
                config=asdict(self.config)
            )
            
            # Rapport console
            self._print_summary(result, total_time)
            
            # Sauvegarde rapport JSON
            report_path = self.output_dir / "extraction_report.json"
            with open(report_path, 'w') as f:
                json.dump(result.to_report(nifti_path, self.output_dir), f, indent=2)
            logger.info(f"📋 Rapport: {report_path.name}")
            
        except Exception as e:
            logger.error(f"❌ Erreur d'extraction: {e}", exc_info=self.verbose)
            result.error_message = str(e)
            result.success = False
        
        return result
    
    def _print_summary(self, result: ExtractionResult, total_time: float):
        """Affiche un résumé formaté de l'extraction."""
        if not result.success:
            logger.error(f"💥 Extraction échouée: {result.error_message}")
            return
        
        logger.info(f"\n{'='*50}")
        logger.info(f"📊 RAPPORT D'EXTRACTION ({total_time:.1f}s)")
        logger.info(f"{'='*50}")
        
        if result.tl_stats:
            s = result.tl_stats
            logger.info(f"\n🔵 TRUE LUMEN:")
            logger.info(f"   ├─ Vertices: {s.vertices:,} | Faces: {s.faces:,}")
            logger.info(f"   ├─ Volume: {s.volume_mm3:,.0f} mm³ | Surface: {s.surface_area_mm2:,.0f} mm²")
            logger.info(f"   ├─ Étanche: {'✅' if s.is_watertight else '❌'}")
            logger.info(f"   └─ Patches: {s.patches_count}")
        
        if result.fl_stats:
            s = result.fl_stats
            logger.info(f"\n🔴 FALSE LUMEN:")
            logger.info(f"   ├─ Vertices: {s.vertices:,} | Faces: {s.faces:,}")
            logger.info(f"   ├─ Volume: {s.volume_mm3:,.0f} mm³ | Surface: {s.surface_area_mm2:,.0f} mm²")
            logger.info(f"   ├─ Étanche: {'✅' if s.is_watertight else '❌'}")
            logger.info(f"   └─ Patches: {s.patches_count}")
        
        logger.info(f"\n✅ Extraction terminée!")


# ============================================================================
# INTERFACE EN LIGNE DE COMMANDE
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse et valide les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Extraction et visualisation STL pour TBAD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Extraction + visualisation interactive
  python %(prog)s --viz
  
  # Paramètres personnalisés
  python %(prog)s --refine 3 --target-tl 300000 --sigma 0.5 --viz
  
  # Visualisation seule (fichiers existants)
  python %(prog)s --viz-only --tl-stl output/tbad_TL_walls.stl --fl-stl output/tbad_FL_walls.stl
  
  # Téléchargement du dataset
  python %(prog)s --download
        """
    )
    
    # Groupe: Entrée
    g_input = parser.add_argument_group("📥 Entrée")
    g_input.add_argument(
        "nifti", nargs="?", default=None,
        help="Fichier NIfTI (recherche auto si omis)"
    )
    g_input.add_argument(
        "--download", action="store_true",
        help="Télécharger le dataset TBAD depuis Kaggle"
    )
    
    # Groupe: Extraction
    g_extract = parser.add_argument_group("⚙️ Extraction")
    g_extract.add_argument(
        "-o", "--output", default=TBADConfig.default_output_dir,
        help=f"Dossier de sortie (défaut: {TBADConfig.default_output_dir})"
    )
    g_extract.add_argument(
        "--tl-label", type=int, default=TBADConfig.tl_label,
        help=f"Label True Lumen (défaut: {TBADConfig.tl_label})"
    )
    g_extract.add_argument(
        "--fl-label", type=int, default=TBADConfig.fl_label,
        help=f"Label False Lumen (défaut: {TBADConfig.fl_label})"
    )
    g_extract.add_argument(
        "-r", "--refine", type=int, default=TBADConfig.refine_factor,
        help=f"Facteur de raffinement (défaut: {TBADConfig.refine_factor})"
    )
    g_extract.add_argument(
        "--sigma", type=float, default=TBADConfig.sdf_sigma_mm,
        help=f"Sigma lissage SDF (défaut: {TBADConfig.sdf_sigma_mm})"
    )
    g_extract.add_argument(
        "--target-tl", type=int, default=TBADConfig.target_triangles_tl,
        help=f"Triangles cible TL (défaut: {TBADConfig.target_triangles_tl})"
    )
    g_extract.add_argument(
        "--target-fl", type=int, default=TBADConfig.target_triangles_fl,
        help=f"Triangles cible FL (défaut: {TBADConfig.target_triangles_fl})"
    )
    g_extract.add_argument(
        "--no-patches", action="store_true",
        help="Désactiver la détection des patches"
    )
    g_extract.add_argument(
        "--no-extract", action="store_true",
        help="Skip extraction (visualisation seule de fichiers existants)"
    )
    
    # Groupe: Visualisation
    g_viz = parser.add_argument_group("🎨 Visualisation")
    g_viz.add_argument(
        "--viz", action="store_true",
        help="Lancer la visualisation interactive après extraction"
    )
    g_viz.add_argument(
        "--viz-only", action="store_true",
        help="Visualisation seule sans extraction"
    )
    g_viz.add_argument(
        "--tl-stl", help="Fichier STL TL pour visualisation seule"
    )
    g_viz.add_argument(
        "--fl-stl", help="Fichier STL FL pour visualisation seule"
    )
    g_viz.add_argument(
        "--single", help="Visualiser un seul fichier STL"
    )
    
    # Groupe: Divers
    g_misc = parser.add_argument_group("🔧 Divers")
    g_misc.add_argument(
        "-q", "--quiet", action="store_true",
        help="Mode silencieux (erreurs seulement)"
    )
    g_misc.add_argument(
        "-v", "--verbose", action="store_true",
        help="Mode verbeux avec stack traces"
    )
    g_misc.add_argument(
        "--no-warn", action="store_true",
        help="Supprimer les warnings Python"
    )
    
    return parser.parse_args()


def main() -> int:
    """Point d'entrée principal avec gestion d'erreurs."""
    args = parse_arguments()
    
    # Configuration logging
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Suppression warnings si demandé
    if args.no_warn:
        warnings.filterwarnings("ignore")
    
    try:
        # ====================================================================
        # MODE: Visualisation seule
        # ====================================================================
        if args.viz_only:
            if args.single:
                # Visualisation mono-fichier
                viz = TbadVisualizer()
                success = viz.visualize_single(Path(args.single))
                return 0 if success else 1
            
            # Visualisation paire TL/FL
            if not args.tl_stl or not args.fl_stl:
                logger.error("❌ Mode visualisation: spécifiez --tl-stl ET --fl-stl")
                return 1
            
            viz = TbadVisualizer()
            tl_patches = list(Path(args.tl_stl).parent.glob("tbad_TL_patch_*.stl"))
            fl_patches = list(Path(args.fl_stl).parent.glob("tbad_FL_patch_*.stl"))
            
            success = viz.visualize_pair(
                tl_stl=Path(args.tl_stl),
                fl_stl=Path(args.fl_stl),
                tl_patches=tl_patches or None,
                fl_patches=fl_patches or None
            )
            return 0 if success else 1
        
        # ====================================================================
        # MODE: Téléchargement
        # ====================================================================
        if args.download:
            if not KAGGLE_AVAILABLE:
                logger.error("❌ kagglehub requis pour --download: pip install kagglehub")
                return 1
            
            path = download_tbad_dataset()
            return 0 if path else 1
        
        # ====================================================================
        # MODE: Extraction (+ visualisation optionnelle)
        # ====================================================================
        
        # Résolution du fichier NIfTI
        if args.nifti:
            nifti_path = Path(args.nifti)
            if not nifti_path.exists():
                logger.error(f"❌ Fichier introuvable: {nifti_path}")
                return 1
        else:
            nifti_path = find_nifti_file()
            if not nifti_path:
                logger.error(
                    "❌ Aucun fichier NIfTI trouvé.\n"
                    "   • Placez un fichier .nii.gz dans le répertoire\n"
                    "   • Ou utilisez --download pour télécharger le dataset TBAD\n"
                    "   • Ou spécifiez le chemin directement"
                )
                return 1
        
        # Extraction
        if not args.no_extract:
            extractor = TbadExtractor(
                output_dir=args.output,
                extract_patches=not args.no_patches,
                verbose=not args.quiet
            )
            # Override config avec args CLI
            extractor.config = TBADConfig(
                refine_factor=args.refine,
                sdf_sigma_mm=args.sigma,
                target_triangles_tl=args.target_tl,
                target_triangles_fl=args.target_fl,
                tl_label=args.tl_label,
                fl_label=args.fl_label
            )
            extractor._setup_converters()  # Reconfigure avec nouveaux params
            
            result = extractor.extract(
                nifti_path=nifti_path,
                tl_label=args.tl_label,
                fl_label=args.fl_label
            )
            
            if not result.success:
                logger.error(f"💥 Extraction échouée: {result.error_message}")
                return 1
        else:
            # Mode "no-extract": utiliser fichiers existants
            result = ExtractionResult(
                success=True,
                tl_stl_path=Path(args.output) / "tbad_TL_walls.stl",
                fl_stl_path=Path(args.output) / "tbad_FL_walls.stl",
                tl_patches=list(Path(args.output).glob("tbad_TL_patch_*.stl")),
                fl_patches=list(Path(args.output).glob("tbad_FL_patch_*.stl"))
            )
            if not result.tl_stl_path.exists() or not result.fl_stl_path.exists():
                logger.error("❌ Fichiers STL existants introuvables")
                return 1
        
        # Visualisation post-extraction
        if args.viz:
            viz = TbadVisualizer()
            viz.visualize_pair(
                tl_stl=result.tl_stl_path,
                fl_stl=result.fl_stl_path,
                tl_patches=result.tl_patches or None,
                fl_patches=result.fl_patches or None
            )
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interruption par l'utilisateur")
        return 130
    except Exception as e:
        logger.critical(f"💥 Erreur fatale: {e}", exc_info=args.verbose if hasattr(args, 'verbose') else False)
        return 2


# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == "__main__":
    # Seed pour reproductibilité des opérations aléatoires
    np.random.seed(42)
    
    # Exécution avec code de sortie approprié
    sys.exit(main())