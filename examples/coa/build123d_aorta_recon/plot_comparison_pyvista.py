#!/usr/bin/env python3
"""
Side-by-side visualization using PyVista:
- Left : STL patient + centerline + sections
- Right : CAD reconstruite + centerline

Requires VTK >= 9.6.2 with EGL support for off-screen rendering under WSLg.
"""

import os
os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")

import sys
from pathlib import Path

import numpy as np
import trimesh
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

from foampilot.geometry.topology import TopologySectionExtractor
from foampilot.geometry.topology.section_extractor import _process_section_polylines

BASE_DIR = Path(__file__).resolve().parent
STL_PATH = BASE_DIR.parent / "patient58_cfd_example" / "constant" / "triSurface" / "tbad_TL_walls.stl"
CENTERLINE_PATH = BASE_DIR.parent / "patient58_cfd_example" / "centerline.npy"
CAD_STL_PATH = BASE_DIR / "aorta_loft_stl_sections.stl"


def load_mesh_pv(path: Path) -> pv.PolyData:
    mesh = trimesh.load(str(path), process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    return pv.wrap(mesh)


def extract_sections(mesh: trimesh.Trimesh, centerline: np.ndarray, n_sections: int = 11):
    sections = []
    step = max(1, len(centerline) // n_sections)
    for i in range(0, len(centerline), step):
        center = centerline[i]
        if i == 0:
            tangent = centerline[1] - centerline[0]
        elif i >= len(centerline) - 1:
            tangent = centerline[-1] - centerline[-2]
        else:
            tangent = centerline[i + 1] - centerline[i - 1]
        tangent = tangent / np.linalg.norm(tangent)
        try:
            section = mesh.section(plane_origin=center, plane_normal=tangent)
            points = _process_section_polylines(section, tangent, center, n_resample=64)
            if points is not None:
                sections.append(points)
        except Exception:
            pass
    return sections


def main():
    print("Loading meshes...")
    stl_pv = load_mesh_pv(STL_PATH)
    cad_pv = load_mesh_pv(CAD_STL_PATH)
    centerline = np.load(str(CENTERLINE_PATH))

    mesh = trimesh.load(str(STL_PATH), process=True)
    sections = extract_sections(mesh, centerline, n_sections=11)

    print(f"STL: {stl_pv.n_points} points")
    print(f"CAD: {cad_pv.n_points} points")
    print(f"Sections: {len(sections)}")

    plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 800), off_screen=True)
    plotter.set_background("white")

    # --- Left: STL patient ---
    plotter.subplot(0, 0)
    plotter.add_text("STL Patient", font_size=12, color="black")
    plotter.add_mesh(stl_pv, color="steelblue", opacity=0.6, smooth_shading=True, label="STL")

    # Centerline
    centerline_pv = pv.lines_from_points(centerline)
    plotter.add_mesh(centerline_pv, color="red", line_width=3, label="Centerline")

    # Sections
    for i, pts in enumerate(sections):
        pts_closed = np.vstack([pts, pts[0]])
        section_line = pv.lines_from_points(pts_closed)
        plotter.add_mesh(section_line, color="orange", line_width=2)

    plotter.add_legend()
    plotter.view_isometric()

    # --- Right: CAD reconstruite ---
    plotter.subplot(0, 1)
    plotter.add_text("CAD Reconstructed (Loft STL Sections)", font_size=12, color="black")
    plotter.add_mesh(cad_pv, color="forestgreen", opacity=0.8, smooth_shading=True, label="CAD Loft")

    # Centerline
    centerline_pv = pv.lines_from_points(centerline)
    plotter.add_mesh(centerline_pv, color="red", line_width=3, label="Centerline")

    plotter.add_legend()
    plotter.view_isometric()

    output_path = BASE_DIR / "comparison_pyvista.png"
    plotter.screenshot(str(output_path))
    print(f"Screenshot saved: {output_path}")


if __name__ == "__main__":
    main()
