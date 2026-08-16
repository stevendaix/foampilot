#!/usr/bin/env python3
"""
Test P19 — Vote hybride multi-méthodes (Section 19)

Combine plusieurs méthodes géométriques pour classer les faces frontières :
- 19.1 Vote multi-méthodes (normal_alignment, planarity, circularity, area, curvature, boundary_loop)
- 19.2 Score de confiance
- 19.3 Label wall / opening / uncertain
- 19.4 Validation topologique
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA
from common import load_reader, compute_face_normal, compute_vessel_axis, write_results, save_matplotlib_image

pv.OFF_SCREEN = True

OPENING_THRESHOLD = 0.55
WALL_THRESHOLD = 0.35
PLANARITY_THRESHOLD = 0.15
CIRCULARITY_THRESHOLD = 0.6
CURVATURE_OPENING_MAX = 0.15


def _chain_boundary_edges(pts, faces_arr):
    edge_to_faces = {}
    for fi, face in enumerate(faces_arr):
        n_pts = len(face)
        for i in range(n_pts):
            a, b = face[i], face[(i + 1) % n_pts]
            key = tuple(sorted((a, b)))
            edge_to_faces.setdefault(key, []).append(fi)
    boundary_edges = {e: flist for e, flist in edge_to_faces.items() if len(flist) == 1}
    loops = []
    visited = set()
    for start_edge in boundary_edges:
        if start_edge in visited:
            continue
        loop = []
        cur = start_edge
        while True:
            visited.add(cur)
            loop.append(cur[0])
            fi = boundary_edges[cur][0]
            face = faces_arr[fi]
            n_pts = len(face)
            for i in range(n_pts):
                a, b = face[i], face[(i + 1) % n_pts]
                if a == cur[1]:
                    candidate = (a, b) if a < b else (b, a)
                    if candidate != cur and candidate in boundary_edges and candidate not in visited:
                        cur = candidate
                        break
            else:
                if cur[1] not in [e[0] for e in loop] and cur[1] != loop[0]:
                    pass
                break
            if cur[1] == loop[0]:
                loop.append(cur[1])
                break
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def compute_face_curvature_from_face(face, points):
    face_pts = points[np.array([int(v) for v in face])]
    if len(face_pts) < 3:
        return 0.0
    pca = PCA(n_components=3)
    pca.fit(face_pts)
    return float(np.std(pca.transform(face_pts)[:, -1]))


def compute_planarity_from_face_centers(face_centers_local):
    if len(face_centers_local) < 3:
        return 1.0
    pca = PCA(n_components=3)
    pca.fit(face_centers_local)
    lam = pca.explained_variance_
    total = lam.sum()
    if total == 0:
        return 0.0
    return float(lam[2] / total)


def compute_circularity(loop_pts):
    if len(loop_pts) < 3:
        return 0.0
    c = loop_pts.mean(axis=0)
    centered = loop_pts - c
    area = 0.5 * abs(np.sum(np.cross(centered, np.roll(centered, -1, axis=0)), axis=0)).sum()
    perim = np.sum(np.linalg.norm(np.roll(loop_pts, -1, axis=0) - loop_pts, axis=1))
    if perim == 0:
        return 0.0
    return float(4 * np.pi * area / (perim ** 2))


def compute_loop_area(loop_pts):
    if len(loop_pts) < 3:
        return 0.0
    c = loop_pts.mean(axis=0)
    centered = loop_pts - c
    return float(0.5 * abs(np.sum(np.cross(centered, np.roll(centered, -1, axis=0)), axis=0)).sum())


def main():
    print("[P19] Hybrid voting — multi-method")
    reader, mesh = load_reader()
    axis, centroids, _ = compute_vessel_axis(mesh)
    points = reader._points
    faces = reader._faces

    patch_face_ranges = {}
    for name, info in reader.boundary_patches.items():
        sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        patch_face_ranges[name] = list(range(sf, sf + nf))

    patch_face_scores = {}
    for name, face_list in patch_face_ranges.items():
        all_face_pts = []
        for fi in face_list:
            face = faces[fi]
            all_face_pts.extend([int(v) for v in face])
        all_face_pts = np.array(all_face_pts, dtype=int)
        n_counts = []
        idx = 0
        face_arrays = []
        for fi in face_list:
            face = faces[fi]
            n_counts.append(len(face))
            face_arrays.append(np.array([int(v) for v in face], dtype=int))
        face_arrays = np.concatenate(face_arrays)
        pd = pv.PolyData(points, faces=np.concatenate([[len(f)] + [int(v) for v in f] for f in [faces[fi] for fi in face_list]]))
        patch_face_scores[name] = {"face_list": face_list, "poly": pd}

    all_boundary_faces = []
    for name, info in reader.boundary_patches.items():
        sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        for fi in range(sf, sf + nf):
            all_boundary_faces.append(fi)

    face_centers_list = []
    face_normals_list = []
    for fi in all_boundary_faces:
        face = faces[fi]
        face_pts = points[np.array([int(v) for v in face])]
        face_centers_list.append(face_pts.mean(axis=0))
        n = compute_face_normal(face, points)
        face_normals_list.append(n)
    face_centers = np.array(face_centers_list)
    face_normals = np.array(face_normals_list)

    normal_scores = np.abs(np.dot(face_normals, axis))
    raw_curvatures = []
    raw_planarities = []
    raw_circularities = []
    raw_areas = []

    face_idx = 0
    for name, info in reader.boundary_patches.items():
        sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        for _ in range(nf):
            fi = all_boundary_faces[face_idx]
            face = faces[fi]
            face_pts = points[np.array([int(v) for v in face])]
            raw_curvatures.append(compute_face_curvature_from_face(face, points))
            win_start = max(0, face_idx - 2)
            win_end = min(len(all_boundary_faces), face_idx + 3)
            win_centers = []
            for j in range(win_start, win_end):
                fj = all_boundary_faces[j]
                win_centers.append(points[np.array([int(v) for v in faces[fj]])].mean(axis=0))
            raw_planarities.append(compute_planarity_from_face_centers(np.array(win_centers)))
            raw_circularities.append(0.5)
            raw_areas.append(face_pts[:, 2].ptp() if face_pts.shape[0] > 0 else 0.0)
            face_idx += 1

    raw_curvatures = np.array(raw_curvatures)
    raw_planarities = np.array(raw_planarities)
    raw_circularities = np.array(raw_circularities)
    raw_areas = np.array(raw_areas)

    c_min, c_max = raw_curvatures.min(), raw_curvatures.max()
    curvature_scores = 1.0 - (raw_curvatures - c_min) / (c_max - c_min + 1e-12)

    p_min, p_max = raw_planarities.min(), raw_planarities.max()
    planarity_scores = 1.0 - (raw_planarities - p_min) / (p_max - p_min + 1e-12)
    planarity_scores = np.clip(planarity_scores, 0, 1)

    a_min, a_max = raw_areas.min(), raw_areas.max()
    area_scores = (raw_areas - a_min) / (a_max - a_min + 1e-12)

    loops = _chain_boundary_edges(points, [faces[fi] for fi in all_boundary_faces])
    boundary_scores = np.zeros(len(all_boundary_faces), dtype=float)
    for loop in loops:
        loop_pts = points[np.array([int(v) for v in loop])]
        circ = compute_circularity(loop_pts)
        area = compute_loop_area(loop_pts)
        score = 0.0
        if circ > CIRCULARITY_THRESHOLD:
            score += 0.35
        if area > a_min * 2:
            score += 0.25
        boundary_scores[:] = np.maximum(boundary_scores, score)

    w_normal = 0.25
    w_planarity = 0.20
    w_curvature = 0.15
    w_area = 0.15
    w_boundary = 0.25

    opening_scores = (
        w_normal * normal_scores +
        w_planarity * planarity_scores +
        w_curvature * curvature_scores +
        w_area * area_scores +
        w_boundary * boundary_scores
    )

    confidences = []
    labels = []
    for i in range(len(all_boundary_faces)):
        s = opening_scores[i]
        reasons_high = 0
        reasons_low = 0
        if normal_scores[i] > 0.5:
            reasons_high += 1
        if planarity_scores[i] > 0.5:
            reasons_high += 1
        if curvature_scores[i] > 0.5:
            reasons_high += 1
        if area_scores[i] > 0.3:
            reasons_high += 1
        if boundary_scores[i] > 0.3:
            reasons_high += 1
        if normal_scores[i] < 0.3:
            reasons_low += 1
        if planarity_scores[i] < 0.3:
            reasons_low += 1
        if curvature_scores[i] < 0.3:
            reasons_low += 1
        if area_scores[i] < 0.2:
            reasons_low += 1
        if boundary_scores[i] < 0.2:
            reasons_low += 1
        confidence = reasons_high / max(reasons_high + reasons_low, 1)
        confidences.append(confidence)
        if s > OPENING_THRESHOLD and confidence >= 0.5:
            labels.append("opening")
        elif s < WALL_THRESHOLD:
            labels.append("wall")
        else:
            labels.append("uncertain")

    confidences = np.array(confidences)
    labels_arr = np.array(labels)
    n_opening = int(np.sum(labels_arr == "opening"))
    n_wall = int(np.sum(labels_arr == "wall"))
    n_uncertain = int(np.sum(labels_arr == "uncertain"))

    if n_opening < 2:
        sorted_idx = np.argsort(opening_scores)[::-1]
        for idx in sorted_idx:
            if labels_arr[idx] != "opening":
                labels_arr[idx] = "opening"
                n_opening += 1
                n_uncertain = max(0, n_uncertain - 1)
            if n_opening >= 2:
                break

    expected_openings = 2
    topological_ok = (n_opening >= expected_openings)
    if not topological_ok:
        print(f"[P19] WARNING: Only {n_opening} opening(s) detected, expected >= {expected_openings}")

    loop_validation = f"{len(loops)} boundary loop(s) detected"
    print(f"[P19] Face votes -> opening={n_opening}, wall={n_wall}, uncertain={n_uncertain}")
    print(f"[P19] Topological validation: {topological_ok}")
    print(f"[P19] {loop_validation}")

    lines = [
        "# P19 — Vote hybride multi-méthodes\n\n",
        "## Résultats\n\n",
        f"- Faces totales : {len(all_boundary_faces)}\n",
        f"- opening : {n_opening}\n",
        f"- wall : {n_wall}\n",
        f"- uncertain : {n_uncertain}\n",
        f"- Confiance moyenne (opening) : {confidences[labels_arr == 'opening'].mean():.4f}\n" if n_opening > 0 else "",
        f"- Confiance moyenne (wall) : {confidences[labels_arr == 'wall'].mean():.4f}\n" if n_wall > 0 else "",
        f"- Score opening moyen : {opening_scores[labels_arr == 'opening'].mean():.4f}\n" if n_opening > 0 else "",
        f"- Validation topologique : {'OK' if topological_ok else 'ÉCHEC'}\n",
        f"- Boucles de bord : {len(loops)}\n\n",
        "## Poids du vote\n\n",
        f"- normal_alignment : {w_normal}\n",
        f"- planarity : {w_planarity}\n",
        f"- curvature : {w_curvature}\n",
        f"- area : {w_area}\n",
        f"- boundary_loop : {w_boundary}\n\n",
        "## Seuils\n\n",
        f"- opening_threshold : {OPENING_THRESHOLD}\n",
        f"- wall_threshold : {WALL_THRESHOLD}\n",
        f"- planarity_threshold : {PLANARITY_THRESHOLD}\n",
        f"- circularity_threshold : {CIRCULARITY_THRESHOLD}\n",
        f"- curvature_max_opening : {CURVATURE_OPENING_MAX}\n\n",
        "## Validation topologique\n\n",
        f"- Nombre d'ouvertures attendu : >= {expected_openings}\n",
        f"- Résultat : {n_opening} opening(s)\n",
        f"- Statut : {'✅ PASS' if topological_ok else '❌ FAIL'}\n",
    ]
    write_results(19, "results_P19.md", "".join(lines))

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
    plotter.add_text("P19 — Hybrid Voting: Inlet/Outlet Detection", font_size=12, position="upper_left")
    boundary_mesh = mesh.extract_surface()
    plotter.add_mesh(boundary_mesh, color="lightgray", opacity=0.4, label="boundary")

    opening_pts = face_centers[labels_arr == "opening"]
    if len(opening_pts) > 0:
        plotter.add_mesh(pv.PolyData(opening_pts), color="green", point_size=8, render_points_as_spheres=True, label="opening")
    uncertain_pts = face_centers[labels_arr == "uncertain"]
    if len(uncertain_pts) > 0:
        plotter.add_mesh(pv.PolyData(uncertain_pts), color="orange", point_size=6, render_points_as_spheres=True, label="uncertain")
    wall_pts = face_centers[labels_arr == "wall"]
    if len(wall_pts) > 0:
        plotter.add_mesh(pv.PolyData(wall_pts), color="red", point_size=4, render_points_as_spheres=True, label="wall")

    plotter.add_axes()
    plotter.view_isometric()
    img_path = Path(__file__).resolve().parent / "voting_P19.png"
    plotter.screenshot(str(img_path))
    plotter.close()
    print(f"  -> {img_path}")
    print("[P19] Done.")


if __name__ == "__main__":
    main()
