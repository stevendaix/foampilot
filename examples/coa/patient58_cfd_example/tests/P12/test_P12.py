#!/usr/bin/env python3
"""
Test P12 — Slicing (Section 12)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.decomposition import PCA
from common import load_reader, compute_vessel_axis, write_results, save_matplotlib_image

def main():
    print("[P12] Slicing")
    reader, mesh = load_reader()
    axis, centroids, _ = compute_vessel_axis(mesh)
    
    s = np.dot(centroids - centroids.mean(axis=0), axis)
    s_min, s_max = np.percentile(s, 2), np.percentile(s, 98)
    slices = np.linspace(s_min, s_max, 8)
    areas = []
    for sl in slices:
        mask = np.abs(s - sl) < (s_max - s_min) * 0.05
        if np.sum(mask) > 5:
            pts = centroids[mask]
            pca = PCA(n_components=2)
            pca.fit(pts)
            areas.append(pca.explained_variance_[0] * pca.explained_variance_[1] * np.pi)
        else:
            areas.append(0)
    
    lines = ["# P12 — Slicing\n",
             "## Patches détectés\n",
             f"- Nombre total de patches : **{len(reader.boundary_patches)}**\n"]
    for name, info in reader.boundary_patches.items():
        nf = info.get("nFaces", 0)
        lines.append(f"- {name} : {nf} faces\n")
    lines.append(f"\n## Coupes\n")
    lines.append(f"- Nombre de coupes : {len(slices)}\n")
    lines.append(f"- Aire min : {min(areas):.6f}\n")
    lines.append(f"- Aire max : {max(areas):.6f}\n")
    write_results(12, "results_P12.md", "".join(lines))
    save_matplotlib_image(12, "slicing_P12.png")
    print("[P12] Done.")

if __name__ == "__main__":
    main()
