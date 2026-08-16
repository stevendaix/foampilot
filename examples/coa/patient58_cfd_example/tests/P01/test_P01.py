#!/usr/bin/env python3
"""
Test P01 — Détection des boucles de bord (Section 1)
Méthode : vtkFeatureEdges -> boundary loops -> ouvertures
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA
from common import load_reader, get_boundary_surface, write_results, screenshot_4panel


def chain_boundary_edges(edges, points):
    edge_lines = edges.lines
    edge_pairs = []
    for i in range(0, len(edge_lines), 3):
        n = edge_lines[i]
        idx = edge_lines[i + 1:i + 1 + n]
        for j in range(n):
            a = int(idx[j])
            b = int(idx[(j + 1) % n])
            edge_pairs.append((a, b))

    adj = {}
    for a, b in edge_pairs:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    visited = set()
    loops = []
    for start in adj:
        if start in visited:
            continue
        cur = start
        prev = None
        loop = [start]
        while True:
            nxt = None
            for n in adj[cur]:
                if n == prev:
                    continue
                e = tuple(sorted((cur, n)))
                if e not in visited:
                    nxt = n
                    visited.add(e)
                    break
            if nxt is None:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
            if cur == start:
                break
        if len(loop) > 2 and loop[0] == loop[-1]:
            loops.append(loop[:-1])
    return loops


def compute_loop_properties(loop, edge_pts):
    pts = edge_pts[loop]
    center = pts.mean(axis=0)
    area = 0.5 * np.abs(np.sum(np.cross(pts[:-1], pts[1:])))
    perim = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    circ = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0
    pca = PCA(n_components=3)
    pca.fit(pts - center)
    plan = pca.explained_variance_[2] / pca.explained_variance_.sum()
    return {
        "center": center,
        "area": area,
        "perim": perim,
        "circ": circ,
        "plan": plan,
        "n": len(loop),
    }


def main():
    print("[P01] Boundary loops detection")
    reader, mesh = load_reader()
    boundary_poly = get_boundary_surface(reader)
    boundary_poly = boundary_poly.clean().triangulate()

    edges = boundary_poly.extract_all_edges()
    loops = chain_boundary_edges(edges, edges.points)
    print(f"Found {len(loops)} boundary loops")

    loop_props = [compute_loop_properties(loop, edges.points) for loop in loops]
    loop_props.sort(key=lambda x: x["area"], reverse=True)

    lines = ["# P01 — Détection boucles de bord\n",
             f"- Boucles détectées : **{len(loops)}**\n"]
    for i, lp in enumerate(loop_props):
        lines.append(f"## Boucle {i + 1}\n")
        lines.append(f"- Sommets : {lp['n']}\n")
        lines.append(f"- Aire : {lp['area']:.6f}\n")
        lines.append(f"- Circularité : {lp['circ']:.4f}\n")
        lines.append(f"- Planarité : {lp['plan']:.6f}\n")
        lines.append(
            f"- Centre : [{lp['center'][0]:.4f}, {lp['center'][1]:.4f}, {lp['center'][2]:.4f}]\n"
        )
    write_results(1, "results_P01.md", "".join(lines))

    plotter = pv.Plotter(shape=(1, 1), off_screen=True)
    plotter.set_background("white")
    plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)
    colors = ["red", "blue", "green", "orange", "purple", "cyan"]
    for i, lp in enumerate(loop_props[:6]):
        pts = edges.points[np.array(loops[i])]
        n = len(pts)
        f = np.hstack([[n], np.arange(n)])
        poly = pv.PolyData(pts, faces=f)
        plotter.add_mesh(
            poly, color=colors[i % 6], opacity=0.9, show_edges=True, line_width=3
        )
    plotter.add_text("P01: Boundary loops", position="upper_left")
    plotter.view_isometric()
    screenshot_4panel(1, "boundary_loops_P01.png", reader.boundary_patches, reader._faces, reader._points, np.array([0, 0, 1]))
    print("[P01] Done.")


if __name__ == "__main__":
    main()
