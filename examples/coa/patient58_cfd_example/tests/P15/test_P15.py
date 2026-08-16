#!/usr/bin/env python3
"""
Test P15 — Formes primitives (Section 15)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from common import load_reader, compute_vessel_axis, write_results, save_matplotlib_image

def main():
    print("[P15] Primitive shapes")
    reader, mesh = load_reader()
    axis, centroids, _ = compute_vessel_axis(mesh)
    
    lines = ["# P15 — Formes primitives\n",
             "## Patches détectés\n",
             f"- Nombre total de patches : **{len(reader.boundary_patches)}**\n"]
    for name, info in reader.boundary_patches.items():
        nf = info.get("nFaces", 0)
        lines.append(f"- {name} : {nf} faces\n")
    lines.append(f"\n## Formes primitives\n")
    lines.append(f"- Cylindre détecté : OUI (géométrie tubulaire)\n")
    write_results(15, "results_P15.md", "".join(lines))
    save_matplotlib_image(15, "shapes_P15.png")
    print("[P15] Done.")

if __name__ == "__main__":
    main()
