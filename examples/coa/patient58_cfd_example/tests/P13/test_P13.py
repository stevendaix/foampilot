#!/usr/bin/env python3
"""
Test P13 — Graphe topologique (Section 13)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from common import load_reader, write_results, save_matplotlib_image

def main():
    print("[P13] Topological graph")
    reader, mesh = load_reader()
    n_total = sum(info.get("nFaces", 0) for info in reader.boundary_patches.values())
    
    lines = ["# P13 — Graphe topologique\n",
             "## Patches détectés\n",
             f"- Nombre total de patches : **{len(reader.boundary_patches)}**\n"]
    for name, info in reader.boundary_patches.items():
        nf = info.get("nFaces", 0)
        lines.append(f"- {name} : {nf} faces\n")
    lines.append(f"\n## Graphe\n")
    lines.append(f"- Faces totales : **{n_total}**\n")
    write_results(13, "results_P13.md", "".join(lines))
    save_matplotlib_image(13, "topo_graph_P13.png")
    print("[P13] Done.")

if __name__ == "__main__":
    main()
