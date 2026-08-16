#!/usr/bin/env python3
"""
Test P10 — Squelette géométrique / medial axis (Section 10)
Image simplifiée pour éviter le timeout de rendu 4-panel.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA
from common import load_reader, compute_vessel_axis, get_boundary_surface, write_results

def main():
    print("[P10] Geometric skeleton / medial axis (simulated)")
    reader, mesh = load_reader()
    axis, centroids, _ = compute_vessel_axis(mesh)
    
    s = np.dot(centroids - centroids.mean(axis=0), axis)
    skel = centroids[np.argsort(s)]
    end1 = skel[0]
    end2 = skel[-1]
    
    lines = ["# P10 — Squelette géométrique simulé\n",
             f"- Axe : [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]\n",
             f"- Extrémité 1 : [{end1[0]:.4f}, {end1[1]:.4f}, {end1[2]:.4f}]\n",
             f"- Extrémité 2 : [{end2[0]:.4f}, {end2[1]:.4f}, {end2[2]:.4f}]\n"]
    write_results(10, "results_P10.md", "".join(lines))
    
    boundary_poly = get_boundary_surface(reader)
    plotter = pv.Plotter(shape=(1, 1), off_screen=True)
    plotter.set_background("white")
    plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)
    plotter.add_mesh(pv.PolyData(skel), color="black", point_size=4, render_points_as_spheres=True)
    plotter.add_text("P10: Geometric skeleton", position="upper_left")
    plotter.view_isometric()
    img_path = Path(__file__).resolve().parent / "skeleton_P10.png"
    plotter.screenshot(str(img_path), window_size=(1600, 1200))
    plotter.close()
    print(f"  -> {img_path}")
    print("[P10] Done.")

if __name__ == "__main__":
    main()
