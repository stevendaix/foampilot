#!/usr/bin/env python3
"""
Test P11 — Champ de distance (Section 11)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from common import load_reader, compute_vessel_axis, write_results, save_matplotlib_image

def main():
    print("[P11] Distance field")
    reader, mesh = load_reader()
    axis, centroids, _ = compute_vessel_axis(mesh)
    
    s = np.dot(centroids - centroids.mean(axis=0), axis)
    s_min, s_max = np.percentile(s, 2), np.percentile(s, 98)
    dist_field = np.minimum(np.abs(s - s_min), np.abs(s - s_max))
    
    lines = ["# P11 — Champ de distance\n",
             "## Patches détectés\n",
             f"- Nombre total de patches : **{len(reader.boundary_patches)}**\n"]
    for name, info in reader.boundary_patches.items():
        nf = info.get("nFaces", 0)
        lines.append(f"- {name} : {nf} faces\n")
    lines.append(f"\n## Distance aux ouvertures\n")
    lines.append(f"- Distance min : {dist_field.min():.6f}\n")
    lines.append(f"- Distance max : {dist_field.max():.6f}\n")
    lines.append(f"- s_min : {s_min:.4f}, s_max : {s_max:.4f}\n")
    write_results(11, "results_P11.md", "".join(lines))
    save_matplotlib_image(11, "distance_field_P11.png")
    print("[P11] Done.")

if __name__ == "__main__":
    main()
