#!/usr/bin/env python3
"""
Test P21 — Stratégie robuste (Section 21)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from common import load_reader, compute_vessel_axis, write_results, save_matplotlib_image

def main():
    print("[P21] Robust strategy")
    reader, mesh = load_reader()
    axis, centroids, _ = compute_vessel_axis(mesh)
    
    lines = ["# P21 — Stratégie robuste\n",
             "- Combinaison topologie + centerline + normales\n"]
    write_results(21, "results_P21.md", "".join(lines))
    save_matplotlib_image(21, "strategy_P21.png")
    print("[P21] Done.")

if __name__ == "__main__":
    main()
