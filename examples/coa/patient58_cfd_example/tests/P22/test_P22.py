#!/usr/bin/env python3
"""
Test P22 — Critères numériques recommandés (Section 22)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.decomposition import PCA
import pyvista as pv
from common import load_reader, compute_face_normal, compute_vessel_axis, write_results, save_matplotlib_image

pv.OFF_SCREEN = True

def compute_opening_metrics(boundary_points, boundary_faces, axis):
    from vtk.util.numpy_support import numpy_to_vtk
    import vtk

    n_pts = len(boundary_points)
    faces_flat = []
    for f in boundary_faces:
        faces_flat.extend([len(f)] + list(f))
    faces_arr = np.array(faces_flat, dtype=np.int32)

    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    for p in boundary_points:
        pts.InsertNextPoint(p.tolist())
    pd.SetPoints(pts)

    polys = vtk.vtkCellArray()
    for f in boundary_faces:
        polys.InsertNextCell(len(f))
        for v in f:
            polys.InsertCellPoint(int(v))
    pd.SetPolys(polys)

    pd.Modified()

    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.Update()

    edge_pd = fe.GetOutput()
    n_edges = edge_pd.GetNumberOfCells()
    if n_edges == 0:
        return []

    loops = []
    used = np.zeros(n_edges, dtype=bool)
    edge_cells = {}
    for i in range(n_edges):
        c = edge_pd.GetCell(i)
        p1 = c.GetPointId(0)
        p2 = c.GetPointId(1)
        edge_cells.setdefault(p1, []).append((i, p2))
        edge_cells.setdefault(p2, []).append((i, p1))

    for start_edge in range(n_edges):
        if used[start_edge]:
            continue
        loop = []
        current_edge = start_edge
        prev_point = None
        while True:
            used[current_edge] = True
            c = edge_pd.GetCell(current_edge)
            p1 = c.GetPointId(0)
            p2 = c.GetPointId(1)
            if prev_point is None:
                loop.append(p1)
                loop.append(p2)
                prev_point = p2
            else:
                if p1 == prev_point:
                    loop.append(p2)
                    prev_point = p2
                else:
                    loop.append(p1)
                    prev_point = p1

            next_edges = [e for e, pid in edge_cells.get(prev_point, []) if not used[e]]
            if not next_edges:
                break
            current_edge = next_edges[0]
            if loop[0] == prev_point:
                break

        if len(loop) >= 3:
            loops.append(np.array(loop, dtype=int))

    metrics = []
    for loop in loops:
        pts = boundary_points[loop]
        n = len(pts)
        center = pts.mean(axis=0)

        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = normal / norm

        area = 0.0
        perim = 0.0
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            perim += np.linalg.norm(b - a)
            area += abs(np.dot(a, np.cross(b, normal)))

        area = 0.5 * area
        if area < 1e-12:
            continue

        req = np.sqrt(area / np.pi)
        circularity = (4.0 * np.pi * area) / (perim * perim) if perim > 0 else 0.0

        pca = PCA(n_components=3)
        pca.fit(pts - center)
        ev = pca.explained_variance_
        planarity = ev[2] / ev.sum() if ev.sum() > 0 else 1.0

        alignment = abs(np.dot(normal, axis))

        pts_2d = (pts - center) @ normal
        if n > 3:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts_2d)
                convexity = area / hull.volume if hull.volume > 0 else 1.0
            except Exception:
                convexity = 1.0
        else:
            convexity = 1.0

        compactness = (4.0 * np.pi * area) / (perim * perim) if perim > 0 else 0.0

        metrics.append({
            "center": center,
            "normal": normal,
            "area": area,
            "perimeter": perim,
            "radius_eq": req,
            "circularity": circularity,
            "planarity": planarity,
            "convexity": convexity,
            "compactness": compactness,
            "alignment": alignment,
            "n_points": n,
        })

    return metrics


def main():
    print("[P22] Numerical criteria")
    reader, mesh = load_reader()
    axis, centroids, _ = compute_vessel_axis(mesh)
    points = reader._points
    faces = reader._faces

    metrics_list = compute_opening_metrics(points, faces, axis)

    lines = ["# P22 — Critères numériques\n"]
    if not metrics_list:
        lines.append("Aucune ouverture détectée.\n")
    else:
        for i, m in enumerate(metrics_list):
            lines.append(f"## Ouverture {i}\n")
            lines.append(f"- Center : {m['center'].tolist()}\n")
            lines.append(f"- Area : {m['area']:.6f}\n")
            lines.append(f"- Perimeter : {m['perimeter']:.6f}\n")
            lines.append(f"- Radius_eq : {m['radius_eq']:.6f}\n")
            lines.append(f"- Circularity : {m['circularity']:.4f}\n")
            lines.append(f"- Planarity : {m['planarity']:.4f}\n")
            lines.append(f"- Convexity : {m['convexity']:.4f}\n")
            lines.append(f"- Compactness : {m['compactness']:.4f}\n")
            lines.append(f"- Alignment : {m['alignment']:.4f}\n")
            lines.append(f"- N points : {m['n_points']}\n\n")

        means = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0] if k not in ("center",)}
        stds = {k: np.std([m[k] for m in metrics_list]) for k in metrics_list[0] if k not in ("center",)}
        lines.append("## Moyennes\n")
        for k, v in means.items():
            lines.append(f"- {k} : {v:.4f}\n")
        lines.append("\n## Écarts-types\n")
        for k, v in stds.items():
            lines.append(f"- {k} : {v:.4f}\n")

    content = "".join(lines)
    write_results(22, "results_P22.md", content)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        names = ["circularity", "planarity", "compactness", "convexity", "alignment", "area"]
        titles = ["Circularity", "Planarity", "Compactness", "Convexity", "Alignment", "Area"]
        for idx, (name, title) in enumerate(zip(names, titles)):
            ax = axes[idx // 3, idx % 3]
            vals = [m[name] for m in metrics_list] if metrics_list else [0]
            ax.bar(range(len(vals)), vals, color="steelblue")
            ax.set_title(title)
            ax.set_ylim(0, max(vals) * 1.2 if vals and max(vals) > 0 else 1)
        plt.tight_layout()
        save_matplotlib_image(22, "criteria_P22.png")
    except Exception as e:
        print(f"[P22] Image generation error: {e}")

    print("[P22] Done.")


if __name__ == "__main__":
    main()
