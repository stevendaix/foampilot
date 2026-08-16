#!/usr/bin/env python3
"""
Test P05 — Détection de parois cylindriques (Section 5)
Méthodes : RANSAC cylinder, analyse des normales radiales, courbure cylindrique
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA
from common import load_reader, compute_face_normal, compute_vessel_axis, get_boundary_surface, write_results, screenshot_4panel


def compute_face_area(face_vertex_indices, points):
    pts = points[face_vertex_indices]
    if len(pts) < 3:
        return 0.0
    return 0.5 * np.abs(np.sum(np.cross(pts[:-1], pts[1:])))


def build_boundary_face_data(reader):
    points = reader._points
    faces = reader._faces
    patches = reader.boundary_patches

    centers = []
    normals = []
    areas = []
    patch_names = []
    face_indices = []

    for name, info in patches.items():
        sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        for fi in range(sf, sf + nf):
            face = faces[fi]
            pts = points[face]
            center = pts.mean(axis=0)
            normal = compute_face_normal(face, points)
            area = compute_face_area(face, points)
            centers.append(center)
            normals.append(normal)
            areas.append(area)
            patch_names.append(name)
            face_indices.append(fi)

    return {
        "centers": np.array(centers),
        "normals": np.array(normals),
        "areas": np.array(areas),
        "patch_names": patch_names,
        "face_indices": np.array(face_indices),
    }


def fit_cylinder_ransac(pts, n_iter=150, threshold=0.015, min_inliers=20):
    best_inliers = []
    best_model = None
    if len(pts) < 3:
        return best_model, best_inliers

    for _ in range(n_iter):
        sample = pts[np.random.choice(len(pts), 3, replace=False)]
        p1, p2, p3 = sample
        v1 = p2 - p1
        v2 = p3 - p1
        axis_dir = np.cross(v1, v2)
        axis_norm = np.linalg.norm(axis_dir)
        if axis_norm < 1e-9:
            continue
        axis_dir = axis_dir / axis_norm

        projections = (pts - p1) @ axis_dir
        proj_min, proj_max = projections.min(), projections.max()
        if proj_max - proj_min < 1e-6:
            continue

        radial_pts = pts - np.outer(projections, axis_dir) - p1
        radial_dists = np.linalg.norm(radial_pts, axis=1)
        radius = np.median(radial_dists)
        if radius < 1e-6:
            continue

        dists = np.abs(radial_dists - radius)
        inliers = np.where(dists < threshold)[0]
        if len(inliers) > len(best_inliers) and len(inliers) >= min_inliers:
            best_inliers = inliers
            center_base = p1
            best_model = {
                "axis": axis_dir,
                "center": center_base,
                "radius": radius,
                "proj_min": proj_min,
                "proj_max": proj_max,
            }

    return best_model, best_inliers


def classify_cylindrical_wall(face_data, axis, cylinder_model, angle_thresh_deg=35.0, radial_std_thresh=0.04):
    centers = face_data["centers"]
    normals = face_data["normals"]

    if cylinder_model is None:
        return np.zeros(len(centers), dtype=bool), np.zeros(len(centers))

    axis_dir = cylinder_model["axis"]
    center = cylinder_model["center"]
    radius = cylinder_model["radius"]

    projections = (centers - center) @ axis_dir
    radial_vec = centers - center - np.outer(projections, axis_dir)
    radial_dists = np.linalg.norm(radial_vec, axis=1)
    radial_dev = np.abs(radial_dists - radius)

    cos_angle = np.abs(np.dot(normals, axis_dir))
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    is_wall = (angle_deg < angle_thresh_deg) & (radial_dev < radial_std_thresh)
    score = (1.0 - angle_deg / 180.0) * 0.6 + (1.0 - np.clip(radial_dev / (radial_std_thresh + 1e-9), 0, 1)) * 0.4
    return is_wall, score


def compute_face_curvature(face_vertex_indices, points):
    pts = points[face_vertex_indices]
    if len(pts) < 3:
        return 0.0
    c = pts.mean(axis=0)
    pca = PCA(n_components=3)
    pca.fit(pts - c)
    lambdas = pca.explained_variance_
    if lambdas[2] < 1e-12:
        return 0.0
    return float(lambdas[0] / lambdas[2])


def main():
    print("[P05] Cylindrical wall detection (Section 5)")
    reader, mesh = load_reader()
    axis, centroids, _ = compute_vessel_axis(mesh)
    print(f"  Vessel axis: {axis}")

    face_data = build_boundary_face_data(reader)
    centers = face_data["centers"]
    normals = face_data["normals"]
    areas = face_data["areas"]
    face_indices = face_data["face_indices"]

    model, inliers = fit_cylinder_ransac(centers, n_iter=200, threshold=0.02, min_inliers=15)

    if model is not None:
        print(f"  Cylinder fitted: radius={model['radius']:.4f}, inliers={len(inliers)}")
        is_wall, wall_score = classify_cylindrical_wall(face_data, axis, model)
        wall_count = int(is_wall.sum())
        opening_candidates = int((~is_wall).sum())
        print(f"  Wall faces: {wall_count}, Opening candidates: {opening_candidates}")

        curvatures = np.array([compute_face_curvature(reader._faces[fi], reader._points) for fi in face_indices])
        mean_wall_curv = float(curvatures[is_wall].mean()) if wall_count > 0 else 0.0
        mean_open_curv = float(curvatures[~is_wall].mean()) if opening_candidates > 0 else 0.0

        boundary_poly = get_boundary_surface(reader).clean().triangulate()
        plotter = pv.Plotter(shape=(1, 1), off_screen=True)
        plotter.set_background("white")
        plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.35)

        if wall_count > 0:
            wall_pts = centers[is_wall]
            n_w = len(wall_pts)
            f_w = np.hstack([[n_w], np.arange(n_w)])
            wall_poly = pv.PolyData(wall_pts, faces=f_w)
            plotter.add_mesh(wall_poly, color="lightblue", opacity=0.85, show_edges=True, line_width=1)

        if opening_candidates > 0:
            open_pts = centers[~is_wall]
            n_o = len(open_pts)
            f_o = np.hstack([[n_o], np.arange(n_o)])
            open_poly = pv.PolyData(open_pts, faces=f_o)
            plotter.add_mesh(open_poly, color="salmon", opacity=0.9, show_edges=True, line_width=2)

        plotter.add_text("P05: Cylindrical wall detection", position="upper_left")
        plotter.view_isometric()
        screenshot_4panel(5, "cylinder_P05.png", reader.boundary_patches, reader._faces, reader._points, axis)

        lines = ["# P05 — Détection de paroi cylindrique\n\n"]
        lines.append(f"- Axe vaisseau : [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]\n")
        lines.append(f"- Cylindre RANSAC : rayon={model['radius']:.4f}, inliers={len(inliers)}\n")
        lines.append(f"- Faces paroi : **{wall_count}**\n")
        lines.append(f"- Faces ouverture candidates : **{opening_candidates}**\n")
        lines.append(f"- Courbure moyenne paroi : {mean_wall_curv:.4f}\n")
        lines.append(f"- Courbure moyenne ouvertures : {mean_open_curv:.4f}\n\n")
        lines.append("## Interprétation\n")
        lines.append("- Les faces classées `paroi` ont des normales quasi radiales et une courbure cylindrique.\n")
        lines.append("- Les faces restantes sont candidates comme caps/inlet-outlet.\n")
        write_results(5, "results_P05.md", "".join(lines))
    else:
        lines = ["# P05 — Détection de paroi cylindrique\n\n"]
        lines.append("- **Aucun cylindre RANSAC n'a pu être ajusté** aux faces frontières.\n")
        lines.append("- Vérifier la géométrie ou les paramètres de RANSAC.\n")
        write_results(5, "results_P05.md", "".join(lines))

        boundary_poly = get_boundary_surface(reader)
        plotter = pv.Plotter(shape=(1, 1), off_screen=True)
        plotter.set_background("white")
        plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)
        plotter.add_text("P05: No cylinder detected", position="upper_left")
        plotter.view_isometric()
        screenshot_4panel(5, "cylinder_P05.png", reader.boundary_patches, reader._faces, reader._points, axis)

    print("[P05] Done.")


if __name__ == "__main__":
    main()
