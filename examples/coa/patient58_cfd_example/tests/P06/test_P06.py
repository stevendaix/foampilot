#!/usr/bin/env python3
"""
Test P06 — Méthodes par courbure (Section 6)
Courbure moyenne/gaussienne/principale, segmentation par seuil
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyvista as pv
from common import load_reader, compute_vessel_axis, get_boundary_surface, write_results


def segment_by_curvature(curv_vals, thresh_low=0.02, thresh_high=0.15):
    labels = np.full(len(curv_vals), "uncertain", dtype=object)
    labels[curv_vals < thresh_low] = "cap"
    labels[curv_vals > thresh_high] = "wall"
    return labels


def main():
    print("[P06] Curvature-based detection (Section 6)")
    reader, mesh = load_reader()
    boundary_poly = get_boundary_surface(reader)
    boundary_poly = boundary_poly.clean().triangulate()
    boundary_poly = boundary_poly.compute_normals(
        cell_normals=True, point_normals=True
    )

    mean_curv = boundary_poly.curvature(curv_type="mean")
    gauss_curv = boundary_poly.curvature(curv_type="gaussian")
    max_curv = boundary_poly.curvature(curv_type="maximum")
    min_curv = boundary_poly.curvature(curv_type="minimum")

    mean_vals = np.nan_to_num(np.asarray(mean_curv), nan=0.0, posinf=0.0, neginf=0.0)
    gauss_vals = np.nan_to_num(np.asarray(gauss_curv), nan=0.0, posinf=0.0, neginf=0.0)
    max_vals = np.nan_to_num(np.asarray(max_curv), nan=0.0, posinf=0.0, neginf=0.0)
    min_vals = np.nan_to_num(np.asarray(min_curv), nan=0.0, posinf=0.0, neginf=0.0)

    labels = segment_by_curvature(mean_vals, thresh_low=0.02, thresh_high=0.15)

    cap_mask = labels == "cap"
    wall_mask = labels == "wall"
    n_cap = int(cap_mask.sum())
    n_wall = int(wall_mask.sum())
    n_uncertain = int((labels == "uncertain").sum())

    lines = ["# P06 — Détection par courbure\n",
             "## Courbures globales\n",
             f"- Courbure moyenne : min={mean_vals.min():.6f}, max={mean_vals.max():.6f}, mean={mean_vals.mean():.6f}\n",
             f"- Courbure gaussienne : min={gauss_vals.min():.6f}, max={gauss_vals.max():.6f}, mean={gauss_vals.mean():.6f}\n",
             f"- Courbure maximale : min={max_vals.min():.6f}, max={max_vals.max():.6f}, mean={max_vals.mean():.6f}\n",
             f"- Courbure minimale : min={min_vals.min():.6f}, max={min_vals.max():.6f}, mean={min_vals.mean():.6f}\n",
             "\n## Segmentation par seuil de courbure moyenne\n",
             f"- Seuil bas (cap) : < 0.02\n",
             f"- Seuil haut (wall) : > 0.15\n",
             f"- Faces cap : {n_cap}\n",
             f"- Faces wall : {n_wall}\n",
             f"- Faces uncertain : {n_uncertain}\n"]

    cap_pts = boundary_poly.points[cap_mask] if n_cap > 0 else np.empty((0, 3))
    if n_cap > 0:
        lines.append(f"- Centre cap moyen : [{cap_pts.mean(axis=0)[0]:.4f}, {cap_pts.mean(axis=0)[1]:.4f}, {cap_pts.mean(axis=0)[2]:.4f}]\n")

    write_results(6, "results_P06.md", "".join(lines))

    plotter = pv.Plotter(shape=(1, 1), off_screen=True)
    plotter.set_background("white")
    plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4, show_edges=False)

    if n_cap > 0:
        cap_poly = pv.PolyData(cap_pts)
        plotter.add_mesh(cap_poly, color="red", opacity=0.9, point_size=4, render_points_as_spheres=False)
    if n_wall > 0:
        wall_pts = boundary_poly.points[wall_mask]
        wall_poly = pv.PolyData(wall_pts)
        plotter.add_mesh(wall_poly, color="green", opacity=0.9, point_size=4, render_points_as_spheres=False)

    plotter.add_text("P06: Curvature segmentation", position="upper_left")
    plotter.view_isometric()
    img_path = Path(__file__).resolve().parent / "curvature_P06.png"
    plotter.screenshot(str(img_path), window_size=(1600, 1200))
    plotter.close()
    print(f"  -> {img_path}")
    print("[P06] Done.")


if __name__ == "__main__":
    main()
