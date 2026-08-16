#!/usr/bin/env python3
"""Test script for improved section extraction on patient58 aorta STL."""

import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

from foampilot.geometry.topology.section_extractor import TopologySectionExtractor

BASE_DIR = Path(__file__).resolve().parent
STL_PATH = BASE_DIR.parent / "patient58_cfd_example" / "constant" / "triSurface" / "tbad_TL_walls.stl"
CENTERLINE_PATH = BASE_DIR.parent / "patient58_cfd_example" / "centerline.npy"
OUTPUT_PNG = BASE_DIR / "sections_visualization.png"


def main() -> int:
    print("=" * 60)
    print("IMPROVED SECTION EXTRACTION TEST")
    print("=" * 60)

    print("\n[1/4] Loading STL mesh...")
    mesh = trimesh.load(str(STL_PATH), process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"  Bounds: {mesh.bounds}")

    print("\n[2/4] Loading centerline...")
    centerline = np.load(str(CENTERLINE_PATH))
    print(f"  Centerline: {len(centerline)} points")
    total_length = np.sum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))
    print(f"  Length: {total_length * 1000:.1f} mm")

    print("\n[3/4] Extracting sections from centerline...")
    extractor = TopologySectionExtractor()
    sections = extractor.extract_sections_from_centerline(
        mesh, centerline, n_sections=90, n_resample=64
    )
    print(f"  Extracted {len(sections)} valid sections")

    if not sections:
        print("ERROR: No sections extracted")
        return 1

    radii_mm = [s.radius * 1000.0 for s in sections if s.radius > 0]
    areas_mm2 = [s.area * 1e6 for s in sections if s.area > 0]
    points_per_section = [len(s.points) for s in sections]

    print(f"\n[4/4] Statistics:")
    print(f"  Number of sections:      {len(sections)}")
    if radii_mm:
        print(f"  Average radius:          {np.mean(radii_mm):.2f} mm")
        print(f"  Min radius:              {np.min(radii_mm):.2f} mm")
        print(f"  Max radius:              {np.max(radii_mm):.2f} mm")
        print(f"  Std radius:              {np.std(radii_mm):.2f} mm")
    if areas_mm2:
        print(f"  Average area:            {np.mean(areas_mm2):.2f} mm^2")
        print(f"  Min area:                {np.min(areas_mm2):.2f} mm^2")
        print(f"  Max area:                {np.max(areas_mm2):.2f} mm^2")
    print(f"  Points per section:      {np.mean(points_per_section):.1f} (target 64)")
    print(f"  Min points per section:  {np.min(points_per_section)}")
    print(f"  Max points per section:  {np.max(points_per_section)}")

    print("\n[5/5] Visualizing with PyVista...")
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1200))

    pv_mesh = pv.read(str(STL_PATH))
    plotter.add_mesh(pv_mesh, color='lightblue', opacity=0.25, show_edges=False)

    centerline_pv = pv.lines_from_points(centerline)
    plotter.add_mesh(centerline_pv, color='blue', line_width=4, label='Centerline')

    n_plot = min(len(sections), 30)
    step = max(1, len(sections) // n_plot)
    plotted = 0
    for i, section in enumerate(sections):
        if i % step != 0 and i != len(sections) - 1:
            continue
        pts = section.points
        n = len(pts)
        lines = np.zeros(n * 3, dtype=np.int32)
        for j in range(n):
            lines[j * 3] = 2
            lines[j * 3 + 1] = j
            lines[j * 3 + 2] = (j + 1) % n
        contour = pv.PolyData(pts, lines=lines)
        plotter.add_mesh(contour, color='red', line_width=2)
        plotted += 1

    plotter.add_axes()
    plotter.set_background('white')
    plotter.camera_position = 'iso'

    print(f"  Saving PNG to {OUTPUT_PNG} ...")
    plotter.screenshot(str(OUTPUT_PNG))
    plotter.close()
    print(f"  Saved {OUTPUT_PNG} ({plotted} contours plotted)")

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
