#!/usr/bin/env python3
"""
Test P14 — Distance géodésique approchée (Section 14)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from common import load_reader, compute_vessel_axis, write_results, save_matplotlib_image

def main():
    print("[P14] Geodesic distance")
    reader, mesh = load_reader()
    axis, centroids, _ = compute_vessel_axis(mesh)
    
    s = np.dot(centroids - centroids.mean(axis=0), axis)
    
    lines = ["# P14 — Distance géodésique approchée\n",
             "## Patches détectés\n",
             f"- Nombre total de patches : **{len(reader.boundary_patches)}**\n"]
    for name, info in reader.boundary_patches.items():
        nf = info.get("nFaces", 0)
        lines.append(f"- {name} : {nf} faces\n")
    lines.append(f"\n## Distance géodésique\n")
    lines.append(f"- Étendue axiale : {s.max()-s.min():.6f}\n")
    write_results(14, "results_P14.md", "".join(lines))
    save_matplotlib_image(14, "geodesic_P14.png")
    print("[P14] Done.")

if __name__ == "__main__":
    main()
