#!/usr/bin/env python3
"""
Test P20 — Pipeline recommandé (Section 20)
Pipeline VTK/VMTK pour la détection inlet/outlet.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA

from common import load_reader, compute_vessel_axis, write_results, save_matplotlib_image

pv.OFF_SCREEN = True


def extract_boundary_surface(reader):
    points = reader._points
    faces = reader._faces
    all_faces_list = []
    for name, info in reader.boundary_patches.items():
        start_face = info.get("startFace", 0)
        n_faces = info.get("nFaces", 0)
        for fi in range(start_face, start_face + n_faces):
            face = faces[fi]
            n_pts = len(face)
            all_faces_list.append(n_pts)
            all_faces_list.extend([int(v) for v in face])
    all_faces_arr = np.array(all_faces_list, dtype=int)
    surface = pv.PolyData(points, faces=all_faces_arr)
    surface = surface.clean().triangulate().compute_normals(
        point_normals=False, consistent_normals=True, auto_orient_normals=True
    )
    return surface


def detect_boundary_edges(surface):
    edges = surface.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    return edges


def loop_area_and_perimeter(loop_points):
    if len(loop_points) < 3:
        return 0.0, 0.0
    v1 = loop_points[1:] - loop_points[:-1]
    v2 = np.roll(loop_points, -1, axis=0)[:-1] - loop_points[:-1]
    cross = np.cross(v1, v2)
    area = 0.5 * np.sum(np.linalg.norm(cross, axis=1))
    perimeter = np.sum(np.linalg.norm(v1, axis=1))
    return area, perimeter


def loop_center_and_normal(loop_points):
    center = loop_points.mean(axis=0)
    normal = np.array([0.0, 0.0, 0.0])
    for i in range(len(loop_points)):
        p0 = loop_points[i]
        p1 = loop_points[(i + 1) % len(loop_points)]
        normal += np.cross(p0, p1)
    norm = np.linalg.norm(normal)
    if norm > 1e-12:
        normal = normal / norm
    return center, normal


def compute_planarity(loop_points):
    if len(loop_points) < 3:
        return 1.0
    centered = loop_points - loop_points.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]
    distances = np.abs(centered @ normal)
    planarity = np.std(distances) / (np.linalg.norm(centered, axis=1).mean() + 1e-12)
    return float(planarity)


def classify_faces_by_angle(surface, axis):
    face_centers = surface.cell_centers().points
    normals = surface.cell_normals
    tangents = np.tile(axis, (face_centers.shape[0], 1))
    dots = np.abs(np.sum(normals * tangents, axis=1))
    angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return angles, face_centers, normals


def filter_openings(surface, angles, face_centers, axis, threshold_angle=25.0):
    opening_mask = angles < threshold_angle
    projections = face_centers @ axis
    s_min = projections.min()
    s_max = projections.max()
    inlet_mask = opening_mask & (np.abs(projections - s_min) < 1e-6)
    outlet_mask = opening_mask & (np.abs(projections - s_max) < 1e-6)
    return opening_mask, inlet_mask, outlet_mask


def compute_opening_metrics(surface, mask):
    centers = surface.cell_centers().points[mask]
    if centers.shape[0] == 0:
        return {}
    _, normal = loop_center_and_normal(centers)
    area, perimeter = loop_area_and_perimeter(centers)
    eq_radius = np.sqrt(area / np.pi) if area > 1e-12 else 0.0
    circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 1e-12 else 0.0
    planarity = compute_planarity(centers)
    return {
        "n_faces": int(mask.sum()),
        "area": float(area),
        "perimeter": float(perimeter),
        "eq_radius": float(eq_radius),
        "circularity": float(circularity),
        "planarity": float(planarity),
        "normal": normal.tolist(),
    }


def plot_pipeline(surface, axis, angles, inlet_mask, outlet_mask, wall_mask, save_path):
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
    plotter.set_background("white")
    if wall_mask.any():
        plotter.add_mesh(surface.extract_cells(wall_mask), color="lightgray", opacity=0.6, label="wall")
    if inlet_mask.any():
        plotter.add_mesh(surface.extract_cells(inlet_mask), color="green", label="inlet")
    if outlet_mask.any():
        plotter.add_mesh(surface.extract_cells(outlet_mask), color="red", label="outlet")
    plotter.add_arrows(cent=np.array(surface.center), direction=axis, mag=0.15, color="blue", label="axis")
    plotter.add_legend(size=(0.15, 0.15), loc="upper left")
    plotter.view_isometric()
    plotter.screenshot(str(save_path))
    plotter.close()


def main():
    print("[P20] Recommended pipeline (Section 20)")
    reader, mesh = load_reader()
    surface = extract_boundary_surface(reader)
    edges = detect_boundary_edges(surface)
    axis, centroids, _ = compute_vessel_axis(mesh)
    angles, face_centers, normals = classify_faces_by_angle(surface, axis)
    opening_mask, inlet_mask, outlet_mask = filter_openings(surface, angles, face_centers, axis)
    wall_mask = ~opening_mask

    inlet_metrics = compute_opening_metrics(surface, inlet_mask)
    outlet_metrics = compute_opening_metrics(surface, outlet_mask)

    img_path = Path(__file__).resolve().parent / "pipeline_P20.png"
    plot_pipeline(surface, axis, angles, inlet_mask, outlet_mask, wall_mask, img_path)

    results = [
        "# P20 — Pipeline recommandé VTK/VMTK\n",
        "\n## Métriques\n",
        f"- Faces totales (surface) : {surface.n_cells}\n",
        f"- Arêtes de bord détectées : {edges.n_points}\n",
        f"- Axis : [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]\n",
        "\n## Inlet\n",
        f"- Faces : {inlet_metrics.get('n_faces', 0)}\n",
        f"- Aire : {inlet_metrics.get('area', 0):.4f}\n",
        f"- Périmètre : {inlet_metrics.get('perimeter', 0):.4f}\n",
        f"- Rayon équivalent : {inlet_metrics.get('eq_radius', 0):.4f}\n",
        f"- Circularité : {inlet_metrics.get('circularity', 0):.4f}\n",
        f"- Planarité : {inlet_metrics.get('planarity', 0):.4f}\n",
        "\n## Outlet\n",
        f"- Faces : {outlet_metrics.get('n_faces', 0)}\n",
        f"- Aire : {outlet_metrics.get('area', 0):.4f}\n",
        f"- Périmètre : {outlet_metrics.get('perimeter', 0):.4f}\n",
        f"- Rayon équivalent : {outlet_metrics.get('eq_radius', 0):.4f}\n",
        f"- Circularité : {outlet_metrics.get('circularity', 0):.4f}\n",
        f"- Planarité : {outlet_metrics.get('planarity', 0):.4f}\n",
        "\n## Convention\n",
        "- Inlet = extrémité s_min de la centerline (point[0])\n",
        "- Outlet = extrémité s_max de la centerline (point[-1])\n",
        "\n## Status\n",
        "- ✅ Pipeline exécuté avec succès\n",
    ]
    write_results(20, "results_P20.md", "".join(results))
    print(f"[P20] Done. Image -> {img_path}")


if __name__ == "__main__":
    main()
