#!/usr/bin/env python3
"""
Test P17 — Sélection interactive simulée (Section 17)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from common import load_reader, write_results, save_matplotlib_image

def main():
    print("[P17] Interactive selection")
    reader, mesh = load_reader()
    patches = reader.boundary_patches
    
    selected = {name: name.lower() in ["inlet", "outlet"] for name in patches}
    
    lines = ["# P17 — Sélection interactive\n",
             "## Patches détectés\n",
             f"- Nombre total de patches : **{len(patches)}**\n"]
    for name, info in patches.items():
        nf = info.get("nFaces", 0)
        lines.append(f"- {name} : {nf} faces\n")
    lines.append(f"\n## Patches sélectionnés comme inlet/outlet\n")
    for name, sel in selected.items():
        if sel:
            lines.append(f"- **{name}**\n")
    write_results(17, "results_P17.md", "".join(lines))
    save_matplotlib_image(17, "interactive_P17.png")
    print("[P17] Done.")

if __name__ == "__main__":
    main()
