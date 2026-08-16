#!/usr/bin/env python3
"""
Test P09 — PCA globale et locale (Section 9)
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
    print("[P09] PCA global and local axes")
    reader, mesh = load_reader()
    axis, centroids, _ = compute_vessel_axis(mesh)
    
    pca = PCA(n_components=3)
    pca.fit(centroids)
    ev = pca.explained_variance_ratio_
    
    lines = ["# P09 — PCA global et local\n",
             f"- Composante 1 : [{pca.components_[0][0]:.4f}, {pca.components_[0][1]:.4f}, {pca.components_[0][2]:.4f}] (var {ev[0]:.4f})\n",
             f"- Composante 2 : [{pca.components_[1][0]:.4f}, {pca.components_[1][1]:.4f}, {pca.components_[1][2]:.4f}] (var {ev[1]:.4f})\n",
             f"- Composante 3 : [{pca.components_[2][0]:.4f}, {pca.components_[2][1]:.4f}, {pca.components_[2][2]:.4f}] (var {ev[2]:.4f})\n"]
    write_results(9, "results_P09.md", "".join(lines))
    
    boundary_poly = get_boundary_surface(reader)
    plotter = pv.Plotter(shape=(1, 1), off_screen=True)
    plotter.set_background("white")
    plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)
    c = centroids.mean(axis=0)
    for i, comp in enumerate(pca.components_):
        plotter.add_arrows(c, comp * 0.1, mag=0.1, color=["red", "green", "blue"][i])
    plotter.add_text("P09: PCA axes", position="upper_left")
    plotter.view_isometric()
    img_path = Path(__file__).resolve().parent / "pca_P09.png"
    plotter.screenshot(str(img_path), window_size=(1600, 1200))
    plotter.close()
    print(f"  -> {img_path}")
    print("[P09] Done.")

if __name__ == "__main__":
    main()
