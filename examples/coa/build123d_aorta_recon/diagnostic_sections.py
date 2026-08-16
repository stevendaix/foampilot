#!/usr/bin/env python3
"""
Diagnostic plot for loft vs STL volume discrepancy.

Shows:
1. Section areas along the centerline
2. Volume integration vs loft volume
3. Radius/diameter variation
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

import trimesh
from foampilot.geometry.topology import TopologySectionExtractor
from foampilot.geometry.topology.section_extractor import _process_section_polylines

BASE_DIR = Path(__file__).resolve().parent
STL_PATH = BASE_DIR.parent / "patient58_cfd_example" / "constant" / "triSurface" / "tbad_TL_walls.stl"
CENTERLINE_PATH = BASE_DIR.parent / "patient58_cfd_example" / "centerline.npy"


def main():
    mesh = trimesh.load(str(STL_PATH), process=True)
    centerline = np.load(str(CENTERLINE_PATH))
    
    axis = centerline[-1] - centerline[0]
    axis = axis / np.linalg.norm(axis)
    
    # Extract sections at centerline points (like pipeline does)
    sections = []
    step = max(1, len(centerline) // 10)
    for i in range(0, len(centerline), step):
        center = centerline[i]
        try:
            section = mesh.section(plane_origin=center, plane_normal=axis)
            points = _process_section_polylines(section, axis, center, n_resample=64)
            if points is not None:
                sections.append(points)
        except Exception:
            pass
    
    print(f"Extracted {len(sections)} sections")
    
    # Compute metrics for each section
    areas = []
    eq_diameters = []
    positions = []
    
    for i, pts in enumerate(sections):
        center_pt = pts.mean(axis=0)
        distances = np.linalg.norm(pts - center_pt, axis=1)
        mean_r = np.mean(distances)
        area = np.pi * mean_r**2
        
        idx = min(i * step, len(centerline) - 1)
        positions.append(centerline[idx])
        
        areas.append(area)
        eq_diameters.append(mean_r * 2)
    
    areas = np.array(areas)
    eq_diameters = np.array(eq_diameters)
    positions = np.array(positions)
    
    # Compute cumulative arc length along centerline
    arc_lengths = np.zeros(len(positions))
    for i in range(1, len(positions)):
        arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(positions[i] - positions[i-1])
    
    # Compute volume by integration
    integrated_volume = 0.0
    for i in range(len(areas) - 1):
        ds = np.linalg.norm(positions[i+1] - positions[i])
        integrated_volume += (areas[i] + areas[i+1]) / 2.0 * ds
    
    # Known volumes
    sweep_volume = 5.144965e-06
    loft_volume = 6.499805e-05
    
    print(f"\n=== Volume Analysis ===")
    print(f"Integrated volume from sections: {integrated_volume:.6e} m³")
    print(f"Loft volume: {loft_volume:.6e} m³")
    print(f"Sweep volume: {sweep_volume:.6e} m³")
    print(f"Loft / Integrated ratio: {loft_volume / integrated_volume:.2f}x")
    print(f"Sweep / Integrated ratio: {sweep_volume / integrated_volume:.2f}x")
    
    # Create diagnostic figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Equivalent diameter along centerline
    ax1 = axes[0, 0]
    ax1.plot(arc_lengths * 1000, eq_diameters * 1000, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Distance along centerline (mm)')
    ax1.set_ylabel('Equivalent diameter (mm)')
    ax1.set_title('Section Diameter vs Centerline Position')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Area along centerline
    ax2 = axes[0, 1]
    ax2.plot(arc_lengths * 1000, areas * 1e6, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Distance along centerline (mm)')
    ax2.set_ylabel('Cross-sectional area (mm²)')
    ax2.set_title('Section Area vs Centerline Position')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Volume comparison bar chart
    ax3 = axes[1, 0]
    volumes = [integrated_volume * 1e6, sweep_volume * 1e6, loft_volume * 1e6]
    labels = ['Integrated\n(sections)', 'Sweep\n(constant)', 'Loft\n(STL sections)']
    colors = ['steelblue', 'coral', 'forestgreen']
    bars = ax3.bar(labels, volumes, color=colors, alpha=0.7)
    ax3.set_ylabel('Volume (cm³)')
    ax3.set_title('Volume Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, vol in zip(bars, volumes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{vol:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 4: Section shapes at key stations
    ax4 = axes[1, 1]
    key_indices = [0, len(sections)//2, -1] if len(sections) >= 3 else [0, -1]
    colors_shapes = ['red', 'blue', 'green']
    
    for idx, color in zip(key_indices, colors_shapes):
        if idx < 0:
            idx = len(sections) + idx
        pts = sections[idx]
        # Center the points
        center_pt = pts.mean(axis=0)
        pts_centered = pts - center_pt
        
        ax4.plot(pts_centered[:, 0] * 1000, pts_centered[:, 1] * 1000, 
                color=color, linewidth=2, label=f'Station {idx}')
    
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_title('Section Shapes at Key Stations')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    output_path = BASE_DIR / "diagnostic_sections_volume.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"\nSaved diagnostic plot: {output_path}")
    
    # Print section table
    print("\n=== Section Table ===")
    print(f"{'Station':>7} {'Pos Z (mm)':>12} {'Area (mm²)':>12} {'EqDiam (mm)':>12}")
    for i in range(len(sections)):
        z_mm = positions[i][2] * 1000
        area_mm2 = areas[i] * 1e6
        diam_mm = eq_diameters[i] * 1000
        print(f"{i:7d} {z_mm:12.2f} {area_mm2:12.2f} {diam_mm:12.2f}")


if __name__ == "__main__":
    main()
