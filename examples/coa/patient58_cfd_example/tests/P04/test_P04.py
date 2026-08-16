#!/usr/bin/env python3
"""
Test P04 — Détection de caps plans (Section 4)
Méthodes : PCA planaire, RANSAC planar, compacité, région de croissance
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


def pca_planarity(pts):
    if len(pts) < 3:
        return 1.0
    c = pts.mean(axis=0)
    pca = PCA(n_components=3)
    pca.fit(pts - c)
    lambdas = pca.explained_variance_
    return float(lambdas[2] / lambdas.sum())


def normal_consistency(normals):
    if len(normals) == 0:
        return 0.0
    C = np.linalg.norm(normals.mean(axis=0))
    return float(C)


def compactness(area, perimeter):
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def region_growing_cap(seed_indices, centers, normals, adjacency, max_angle_deg=30.0, max_dist=0.08):
    max_angle = np.radians(max_angle_deg)
    region = set(seed_indices)
    queue = list(seed_indices)
    seed_normal = normals[list(seed_indices)].mean(axis=0)
    seed_normal = seed_normal / (np.linalg.norm(seed_normal) + 1e-9)

    while queue:
        cur = queue.pop(0)
        for nb in adjacency.get(cur, []):
            if nb in region:
                continue
            if np.linalg.norm(centers[cur] - centers[nb]) > max_dist:
                continue
            angle = np.arccos(np.clip(np.abs(np.dot(normals[cur], seed_normal)), -1, 1))
            if angle < max_angle:
                region.add(nb)
                queue.append(nb)
    return region


def build_face_adjacency(face_indices, faces, points, centers):
    pt_to_faces = {}
    for idx, fi in enumerate(face_indices):
        for pt in faces[fi]:
            pt_to_faces.setdefault(pt, []).append(idx)

    adjacency = {i: set() for i in range(len(face_indices))}
    for idx, fi in enumerate(face_indices):
        face_pts = set(faces[fi])
        for pt in face_pts:
            for nb_idx in pt_to_faces.get(pt, []):
                if nb_idx != idx and np.linalg.norm(centers[idx] - centers[nb_idx]) < 0.05:
                    adjacency[idx].add(nb_idx)
    return {k: list(v) for k, v in adjacency.items()}


def ransac_plane(pts, n_iter=200, threshold=0.02):
    best_inliers = []
    best_plane = None
    if len(pts) < 3:
        return best_plane, best_inliers

    for _ in range(n_iter):
        sample = pts[np.random.choice(len(pts), 3, replace=False)]
        p1, p2, p3 = sample
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -np.dot(normal, p1)
        dists = np.abs(pts @ normal + d)
        inliers = np.where(dists < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal, d)
    return best_plane, best_inliers


def main():
    print("[P04] Plane caps detection (Section 4)")
    reader, mesh = load_reader()
    axis, centroids, U_mag = compute_vessel_axis(mesh)
    print(f"  Axis: {axis}")

    face_data = build_boundary_face_data(reader)
    centers = face_data["centers"]
    normals = face_data["normals"]
    areas = face_data["areas"]
    patch_names = face_data["patch_names"]
    face_indices = face_data["face_indices"]

    adjacency = build_face_adjacency(face_indices, reader._faces, reader._points, centers)

    proj = centers @ axis
    sorted_idx = np.argsort(proj)
    min_seeds = sorted_idx[:5].tolist()
    max_seeds = sorted_idx[-5:].tolist()

    cap1_region = region_growing_cap(min_seeds, centers, normals, adjacency, max_angle_deg=30.0, max_dist=0.08)
    cap2_region = region_growing_cap(max_seeds, centers, normals, adjacency, max_angle_deg=30.0, max_dist=0.08)

    cap_regions = []
    for region in [cap1_region, cap2_region]:
        if len(region) < 3:
            continue
        ridx = list(region)
        rcenters = centers[ridx]
        rnormals = normals[ridx]
        rareas = areas[ridx]
        planarity = pca_planarity(rcenters)
        norm_cons = normal_consistency(rnormals)
        total_area = rareas.sum()
        rcenter = rcenters.mean(axis=0)
        perimeter = 0.0
        for i in ridx:
            for j in adjacency.get(i, []):
                if j in region and i < j:
                    perimeter += np.linalg.norm(centers[i] - centers[j])
        comp = compactness(total_area, perimeter)
        cap_regions.append({
            "region": region,
            "indices": ridx,
            "center": rcenter,
            "area": float(total_area),
            "planarity": planarity,
            "normal_consistency": norm_cons,
            "compactness": comp,
            "n_faces": len(ridx),
        })

    cap_regions.sort(key=lambda x: x["area"], reverse=True)

    boundary_poly = get_boundary_surface(reader)
    boundary_poly = boundary_poly.clean().triangulate()

    plotter = pv.Plotter(shape=(1, 1), off_screen=True)
    plotter.set_background("white")
    plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)

    colors = ["red", "blue"]
    labels = ["Inlet candidate", "Outlet candidate"]
    for i, cap in enumerate(cap_regions[:2]):
        cap_pts = centers[cap["indices"]]
        n = len(cap_pts)
        f = np.hstack([[n], np.arange(n)])
        cap_poly = pv.PolyData(cap_pts, faces=f)
        plotter.add_mesh(cap_poly, color=colors[i % 2], opacity=0.9, show_edges=True, line_width=3)
        plotter.add_text(f"{labels[i]} (area={cap['area']:.4f})", position="upper_left" if i == 0 else "upper_right")

    plotter.add_text("P04: Plane caps detection", position="lower_left")
    plotter.view_isometric()
    screenshot_4panel(4, "plane_caps_P04.png", reader.boundary_patches, reader._faces, reader._points, axis)
    print("[P04] Done.")


if __name__ == "__main__":
    main()
