#!/usr/bin/env python3
"""
Visualisation TBAD - Script de visualisation des résultats d'extraction STL
============================================================================
Génère des visualisations 3D des maillages True Lumen et False Lumen.

Usage:
    python visualize_tbad.py                    # Visualise les fichiers par défaut
    python visualize_tbad.py --tl tl.stl       # Visualise un fichier spécifique
    python visualize_tbad.py --compare          # Compare TL et FL
    python visualize_tbad.py --interactive     # Ouvre PyVista (si disponible)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh

# Configuration
DEFAULT_TL = "tbad_stl_output/tbad_TL_walls.stl"
DEFAULT_FL = "tbad_stl_output/tbad_FL_walls.stl"


def load_mesh(path: str) -> trimesh.Trimesh:
    """Charge un maillage STL."""
    mesh = trimesh.load(path, force='mesh')
    if mesh.is_empty:
        raise ValueError(f"Maillage vide: {path}")
    return mesh


def plot_single_mesh(ax: Axes3D, mesh: trimesh.Trimesh, color: str, title: str, 
                     elev: int = 30, azim: int = 45, alpha: float = 0.8):
    """Affiche un maillage unique sur un axe."""
    ax.plot_trisurf(
        mesh.vertices[:, 0], 
        mesh.vertices[:, 1], 
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        cmap=color,
        alpha=alpha,
        edgecolor='none'
    )
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')


def visualize_single(tl_path: str, fl_path: str, output: str = None):
    """Visualisation côte à côte TL et FL."""
    print(f"📂 Chargement des maillages...")
    
    tl = load_mesh(tl_path)
    fl = load_mesh(fl_path)
    
    print(f"   True Lumen:  {len(tl.faces):,} faces")
    print(f"   False Lumen: {len(fl.faces):,} faces")
    
    fig = plt.figure(figsize=(16, 6))
    
    # True Lumen
    ax1 = fig.add_subplot(121, projection='3d')
    plot_single_mesh(ax1, tl, 'Blues', f'True Lumen\n({len(tl.faces):,} faces)')
    
    # False Lumen
    ax2 = fig.add_subplot(122, projection='3d')
    plot_single_mesh(ax2, fl, 'Reds', f'False Lumen\n({len(fl.faces):,} faces)')
    
    plt.suptitle('TBAD - True Lumen (bleu) + False Lumen (rouge)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"💾 Image sauvegardée: {output}")
    else:
        plt.show()


def visualize_compare(tl_path: str, fl_path: str, output: str = None):
    """Visualisation comparée avec plusieurs angles."""
    print(f"📂 Chargement des maillages...")
    
    tl = load_mesh(tl_path)
    fl = load_mesh(fl_path)
    
    fig = plt.figure(figsize=(18, 10))
    
    angles = [(30, 45), (30, 135), (60, 45), (15, 0)]
    titles = ['Vue 1 (45°)', 'Vue 2 (135°)', 'Vue 3 (dessus)', 'Vue 4 (côté)']
    
    for i, ((elev, azim), title) in enumerate(zip(angles, titles)):
        ax = fig.add_subplot(2, 4, i + 1, projection='3d')
        
        # Centrer les maillages
        tl_centered = tl.vertices - tl.centroid
        fl_centered = fl.vertices - fl.centroid
        
        ax.plot_trisurf(
            tl_centered[:, 0], tl_centered[:, 1], tl_centered[:, 2],
            triangles=tl.faces, color='blue', alpha=0.5, edgecolor='none'
        )
        ax.plot_trisurf(
            fl_centered[:, 0], fl_centered[:, 1], fl_centered[:, 2],
            triangles=fl.faces, color='red', alpha=0.5, edgecolor='none'
        )
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    # Stats
    ax_stats = fig.add_subplot(2, 4, (5, 8))
    ax_stats.axis('off')
    
    stats_text = f"""
    📊 STATISTIQUES TBAD
    ═════════════════════════════════════
    
    🔵 TRUE LUMEN:
       • Vertices: {len(tl.vertices):,}
       • Faces: {len(tl.faces):,}
       • Volume: {tl.volume:.2f} m³
       • Surface: {tl.area:.2f} m²
       • Étanche: {'Oui' if tl.is_watertight else 'Non'}
    
    🔴 FALSE LUMEN:
       • Vertices: {len(fl.vertices):,}
       • Faces: {len(fl.faces):,}
       • Volume: {fl.volume:.2f} m³
       • Surface: {fl.area:.2f} m²
       • Étanche: {'Oui' if fl.is_watertight else 'Non'}
    
    ═════════════════════════════════════
    TL = True Lumen (bleu)
    FL = False Lumen (rouge)
    """
    
    ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                  fontsize=11, verticalalignment='center',
                  fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('TBAD - Comparaison True/False Lumen', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"💾 Image sauvegardée: {output}")
    else:
        plt.show()


def visualize_pyvista(tl_path: str, fl_path: str = None):
    """Visualisation interactive PyVista."""
    try:
        import pyvista as pv
        import trimesh
        
        print("🎬 Lancement PyVista...")
        
        plotter = pv.Plotter(window_size=[1400, 900])
        
        # True Lumen
        if tl_path:
            tl = trimesh.load(tl_path)
            pv_tl = pv.wrap(tl)
            pv_tl.points *= 0.001  # mm → m
            plotter.add_mesh(pv_tl, color='#1E88E5', opacity=0.7, 
                            smooth_shading=True, label='True Lumen')
        
        # False Lumen
        if fl_path:
            fl = trimesh.load(fl_path)
            pv_fl = pv.wrap(fl)
            pv_fl.points *= 0.001  # mm → m
            plotter.add_mesh(pv_fl, color='#DC143C', opacity=0.5,
                            smooth_shading=True, label='False Lumen')
        
        plotter.add_legend()
        plotter.add_axes()
        plotter.show()
        
    except ImportError:
        print("❌ PyVista non installé. Utilisez matplotlib à la place.")
        print("   pip install pyvista")
    except Exception as e:
        print(f"❌ Erreur PyVista: {e}")


def analyze_cross_sections(mesh: trimesh.Trimesh, n_slices: int = 5):
    """Analyse des sections transversales."""
    print(f"\n📐 Analyse des sections transversales...")
    
    # Obtenir les bounds
    bounds = mesh.bounds
    z_min, z_max = bounds[0, 2], bounds[1, 2]
    z_range = z_max - z_min
    slice_positions = np.linspace(z_min, z_max, n_slices)
    
    # Pour chaque position, calculer l'aire de la section
    # (approximation simple utilisant les vertices)
    for z in slice_positions:
        # Trouver les vertices proche de ce z
        z_slice = np.abs(mesh.vertices[:, 2] - z) < z_range / 20
        n_verts = np.sum(z_slice)
        print(f"   Z={z:.3f}: ~{n_verts} vertices")


def main():
    parser = argparse.ArgumentParser(description="Visualisation TBAD STL")
    parser.add_argument("--tl", default=DEFAULT_TL, help="Fichier STL True Lumen")
    parser.add_argument("--fl", default=DEFAULT_FL, help="Fichier STL False Lumen")
    parser.add_argument("--compare", action="store_true", help="Mode comparaison")
    parser.add_argument("--interactive", action="store_true", help="Mode PyVista interactif")
    parser.add_argument("-o", "--output", help="Fichier de sortie PNG")
    parser.add_argument("--analyze", action="store_true", help="Analyse détaillée")
    
    args = parser.parse_args()
    
    # Vérifier les fichiers
    if not Path(args.tl).exists():
        print(f"❌ Fichier introuvable: {args.tl}")
        return 1
    
    if args.interactive:
        visualize_pyvista(args.tl, args.fl if Path(args.fl).exists() else None)
    elif args.compare:
        visualize_compare(args.tl, args.fl if Path(args.fl).exists() else DEFAULT_FL, args.output)
    else:
        visualize_single(args.tl, args.fl if Path(args.fl).exists() else DEFAULT_FL, args.output)
    
    if args.analyze:
        tl = load_mesh(args.tl)
        fl = load_mesh(args.fl) if Path(args.fl).exists() else None
        
        analyze_cross_sections(tl)
        if fl:
            analyze_cross_sections(fl)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
