#!/usr/bin/env python3
"""
Test P03 — Angle normale / tangente locale (Section 3.5)
Méthode : angle entre normale de face et tangente locale de centerline
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA
from common import load_reader, compute_face_normal, compute_vessel_axis, get_boundary_surface, write_results, screenshot_4panel

def main():
    print("[P03] Normal vs local tangent angle")
    reader, mesh = load_reader()
    axis, centroids, _ = compute_vessel_axis(mesh)
    points = reader._points
    faces = reader._faces
    patches = reader.boundary_patches
    
    all_angles = []
    patch_stats = {}
    for name, info in patches.items():
        sf, nf = info.get("startFace",0), info.get("nFaces",0)
        face_ids = np.arange(sf, sf+nf)
        angles = []
        for fi in face_ids:
            n = compute_face_normal(faces[fi], points)
            fc = points[faces[fi]].mean(axis=0)
            d = np.linalg.norm(centroids - fc, axis=1)
            li = np.argmin(d)
            laxis = axis
            if d[li] < 0.05:
                local = centroids[max(0,li-2):li+3]
                if len(local) > 3:
                    pca = PCA(n_components=3); pca.fit(local)
                    laxis = pca.components_[0]
            cos = np.clip(np.abs(np.dot(n, laxis)), -1, 1)
            angles.append(np.degrees(np.arccos(cos)))
        angles = np.array(angles)
        all_angles.extend(angles)
        patch_stats[name] = {"mean": angles.mean(), "std": angles.std(), "min": angles.min(), "max": angles.max()}
    
    lines = ["# P03 — Angle normale / tangente locale\n"]
    for name, st in patch_stats.items():
        lines.append(f"## {name}\n")
        lines.append(f"- Angle moyen : {st['mean']:.2f}°\n")
        lines.append(f"- Écart-type : {st['std']:.2f}°\n")
        lines.append(f"- Min : {st['min']:.2f}°, Max : {st['max']:.2f}°\n")
    write_results(3, "results_P03.md", "".join(lines))
    
    boundary_poly = get_boundary_surface(reader)
    plotter = pv.Plotter(shape=(1,1), off_screen=True)
    plotter.set_background("white")
    plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)
    plotter.add_text("P03: Normal-tangent angles", position="upper_left")
    plotter.view_isometric()
    screenshot_4panel(3, "normal_tangent_P03.png", reader.boundary_patches, reader._faces, reader._points, axis)
    print("[P03] Done.")

if __name__ == "__main__":
    main()
