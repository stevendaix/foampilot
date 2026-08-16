#!/usr/bin/env python3
"""
Test P18 — Convention de labellisation (Section 18)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from common import load_reader, compute_vessel_axis, write_results, save_matplotlib_image

def main():
    print("[P18] Convention labeling")
    reader, mesh = load_reader()
    axis, centroids, _ = compute_vessel_axis(mesh)
    
    s = np.dot(centroids - centroids.mean(axis=0), axis)
    s_min, s_max = np.percentile(s, 2), np.percentile(s, 98)
    
    lines = ["# P18 — Convention de labellisation\n",
             "## Patches détectés\n",
             f"- Nombre total de patches : **{len(reader.boundary_patches)}**\n"]
    for name, info in reader.boundary_patches.items():
        nf = info.get("nFaces", 0)
        lines.append(f"- {name} : {nf} faces\n")
    lines.append(f"\n## Convention\n")
    lines.append(f"- s_min -> inlet : {s_min:.6f}\n")
    lines.append(f"- s_max -> outlet : {s_max:.6f}\n")
    write_results(18, "results_P18.md", "".join(lines))
    save_matplotlib_image(18, "convention_P18.png")
    print("[P18] Done.")

if __name__ == "__main__":
    main()
