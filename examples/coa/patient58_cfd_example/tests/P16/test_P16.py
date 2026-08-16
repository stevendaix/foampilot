#!/usr/bin/env python3
"""
Test P16 — Features ML (Section 16)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from common import load_reader, compute_face_normal, write_results, save_matplotlib_image

def main():
    print("[P16] ML features")
    reader, mesh = load_reader()
    points = reader._points
    faces = reader._faces
    patches = reader.boundary_patches
    
    X = []
    y = []
    label_map = {"wall": 0, "inlet": 1, "outlet": 2}
    for name, info in patches.items():
        sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        lbl = label_map.get(name.lower(), 0)
        for fi in range(sf, sf + nf):
            n = compute_face_normal(faces[fi], points)
            c = points[faces[fi]].mean(axis=0)
            a = 0.5 * np.abs(np.sum(np.cross(points[faces[fi]][:-1], points[faces[fi]][1:])))
            X.append([n[0], n[1], n[2], c[0], c[1], c[2], a])
            y.append(lbl)
    X = np.array(X)
    y = np.array(y)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    acc = clf.score(X, y)
    
    lines = ["# P16 — Features ML\n",
             "## Patches détectés\n",
             f"- Nombre total de patches : **{len(patches)}**\n"]
    for name, info in patches.items():
        nf = info.get("nFaces", 0)
        lines.append(f"- {name} : {nf} faces\n")
    lines.append(f"\n## Classification ML\n")
    lines.append(f"- Échantillons : **{len(X)}**\n")
    lines.append(f"- Accuracy RF : **{acc*100:.2f}%**\n")
    write_results(16, "results_P16.md", "".join(lines))
    save_matplotlib_image(16, "ml_features_P16.png")
    print("[P16] Done.")

if __name__ == "__main__":
    main()
