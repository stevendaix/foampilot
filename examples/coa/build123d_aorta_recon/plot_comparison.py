#!/usr/bin/env python3
"""
Visual comparison of original STL vs reconstructed CAD geometry.

Generates a side-by-side 3D plot:
- Left: Original STL mesh
- Right: Reconstructed CAD (loft from STL sections)

Also shows section contours and centerline overlay.
"""

import sys
from pathlib import Path

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

from foampilot.geometry.topology import TopologySectionExtractor
from foampilot.geometry.topology.section_extractor import _process_section_polylines

BASE_DIR = Path(__file__).resolve().parent
STL_PATH = BASE_DIR.parent / "patient58_cfd_example" / "constant" / "triSurface" / "tbad_TL_walls.stl"
CENTERLINE_PATH = BASE_DIR.parent / "patient58_cfd_example" / "centerline.npy"
CAD_STL_PATH = BASE_DIR / "aorta_loft_stl_sections.stl"


def load_and_sample_mesh(path: Path, n_samples: int = 5000) -> np.ndarray:
    """Load mesh and sample points for visualization."""
    mesh = trimesh.load(str(path), process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    
    # Sample points from mesh surface
    points, _ = trimesh.sample.sample_surface(mesh, n_samples)
    return points


def extract_sections_for_viz(mesh, centerline, n_sections=11):
    """Extract sections for visualization overlay."""
    axis = centerline[-1] - centerline[0]
    axis = axis / np.linalg.norm(axis)
    
    sections = []
    step = max(1, len(centerline) // n_sections)
    for i in range(0, len(centerline), step):
        center = centerline[i]
        try:
            section = mesh.section(plane_origin=center, plane_normal=axis)
            points = _process_section_polylines(section, axis, center, n_resample=64)
            if points is not None:
                sections.append(points)
        except Exception:
            pass
    
    return sections


def plot_comparison():
    """Generate side-by-side comparison plot."""
    print("Loading meshes...")
    
    # Load original STL
    stl_points = load_and_sample_mesh(STL_PATH, n_samples=8000)
    
    # Load reconstructed CAD
    cad_points = load_and_sample_mesh(CAD_STL_PATH, n_samples=8000)
    
    # Load centerline
    centerline = np.load(str(CENTERLINE_PATH))
    
    # Extract sections from original STL
    mesh = trimesh.load(str(STL_PATH), process=True)
    sections = extract_sections_for_viz(mesh, centerline, n_sections=11)
    
    print(f"STL points: {len(stl_points)}")
    print(f"CAD points: {len(cad_points)}")
    print(f"Sections: {len(sections)}")
    print(f"Centerline points: {len(centerline)}")
    
    # Create figure with 2 subplots side by side
    fig = plt.figure(figsize=(16, 8))
    
    # Left: Original STL
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(stl_points[:, 0], stl_points[:, 1], stl_points[:, 2], 
                c='steelblue', s=0.5, alpha=0.6, label='STL patient')
    
    # Overlay centerline
    ax1.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], 
             'r-', linewidth=2, label='Centerline')
    
    # Overlay sections
    for i, section_pts in enumerate(sections):
        ax1.scatter(section_pts[:, 0], section_pts[:, 1], section_pts[:, 2],
                   c='orange', s=10, alpha=0.8)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('STL Patient + Sections + Centerline')
    ax1.legend()
    
    # Right: Reconstructed CAD
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(cad_points[:, 0], cad_points[:, 1], cad_points[:, 2],
                c='forestgreen', s=0.5, alpha=0.6, label='CAD Loft')
    
    # Overlay centerline
    ax2.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2],
             'r-', linewidth=2, label='Centerline')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('CAD Reconstructed (Loft STL Sections)')
    ax2.legend()
    
    # Set equal aspect ratio
    for ax in [ax1, ax2]:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        z_lim = ax.get_zlim()
        max_range = max(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0])
        mid_x = np.mean(x_lim)
        mid_y = np.mean(y_lim)
        mid_z = np.mean(z_lim)
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    output_path = BASE_DIR / "comparison_stl_vs_cad.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Also create a figure showing only the sections
    fig2 = plt.figure(figsize=(10, 8))
    ax3 = fig2.add_subplot(111, projection='3d')
    
    # Plot all sections with different colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(sections)))
    for i, (section_pts, color) in enumerate(zip(sections, colors)):
        ax3.scatter(section_pts[:, 0], section_pts[:, 1], section_pts[:, 2],
                   c=[color], s=20, alpha=0.8, label=f'Section {i}')
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('Extracted Sections Along Centerline')
    ax3.legend()
    
    output_path2 = BASE_DIR / "sections_only.png"
    plt.savefig(str(output_path2), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path2}")
    
    plt.show()


if __name__ == "__main__":
    plot_comparison()
