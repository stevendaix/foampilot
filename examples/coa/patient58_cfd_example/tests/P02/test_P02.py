#!/usr/bin/env python3
"""
Test P02 — Centerline simulée + extrémités (Section 2)
Méthode : PCA axis -> s-coordinate -> endpoints
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA
from common import load_reader, compute_vessel_axis, get_boundary_surface, write_results, screenshot_4panel

def main():
    print("[P02] Centerline simulation + endpoints")
    reader, mesh = load_reader()
    axis, centroids, U_mag = compute_vessel_axis(mesh)
    
    s = np.dot(centroids - centroids.mean(axis=0), axis)
    s_min_idx = np.argmin(s)
    s_max_idx = np.argmax(s)
    end1 = centroids[s_min_idx]
    end2 = centroids[s_max_idx]
    
    lines = ["# P02 — Centerline simulée\n",
             f"- Axe : [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]\n",
             f"- Extrémité 1 (inlet) : [{end1[0]:.4f}, {end1[1]:.4f}, {end1[2]:.4f}]\n",
             f"- Extrémité 2 (outlet) : [{end2[0]:.4f}, {end2[1]:.4f}, {end2[2]:.4f}]\n",
             f"- s[extrémité 1] : {s[s_min_idx]:.4f}, s[extrémité 2] : {s[s_max_idx]:.4f}\n"]
    write_results(2, "results_P02.md", "".join(lines))
    
    boundary_poly = get_boundary_surface(reader).clean().triangulate()
    plotter = pv.Plotter(shape=(1,1), off_screen=True)
    plotter.set_background("white")
    plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)
    cl = centroids[np.argsort(s)]
    plotter.add_mesh(pv.PolyData(cl), color="black", point_size=4, render_points_as_spheres=True)
    plotter.add_arrows(end1, axis*0.05, mag=0.05, color="red")
    plotter.add_arrows(end2, axis*0.05, mag=0.05, color="blue")
    plotter.add_text("P02: Centerline + endpoints", position="upper_left")
    plotter.view_isometric()
    screenshot_4panel(2, "centerline_P02.png", reader.boundary_patches, reader._faces, reader._points, axis)
    print("[P02] Done.")

if __name__ == "__main__":
    main()
