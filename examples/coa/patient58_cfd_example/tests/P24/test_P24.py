#!/usr/bin/env python3
"""
Test P24 — Conclusion (Section 24)
Combine les méthodes clés pour la détection inlet/outlet :
- topologie de surface (boucles de bord)
- centerline géométrique (PCA sur centroïdes actifs)
- normales locales
- angle normale / tangente locale
- forme des caps (aire, circularité, planarity)
- validation topologique
- convention inlet/outlet
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA
from common import load_reader, compute_face_normal, compute_vessel_axis, get_boundary_surface, write_results, save_matplotlib_image

pv.OFF_SCREEN = True


def chain_boundary_edges(edges):
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


def main():
    print("[P24] Conclusion — Pipeline combiné")

    reader, mesh = load_reader()
    axis, centroids, _ = compute_vessel_axis(mesh)
    points = reader._points
    faces = reader._faces
    patches = reader.boundary_patches

    lines = ["# P24 — Conclusion (Section 24)\n\n",
             "Pipeline combiné : topologie + centerline + normales + angle + forme + validation + convention\n\n"]

    # === Étape 1 : Topologie de surface (boundary loops) ===
    boundary_poly = get_boundary_surface(reader)
    boundary_poly = boundary_poly.clean().triangulate()
    edges = boundary_poly.extract_all_edges()
    loops = chain_boundary_edges(edges)
    loop_props = [compute_loop_properties(loop, edges.points) for loop in loops]
    loop_props.sort(key=lambda x: x["area"], reverse=True)
    n_loops = len(loops)

    lines.append(f"## 1. Topologie de surface\n")
    lines.append(f"- Boucles de bord détectées : **{n_loops}**\n")
    if loop_props:
        lp = loop_props[0]
        lines.append(f"- Plus grande boucle : aire={lp['area']:.6f}, circ={lp['circ']:.4f}, plan={lp['plan']:.4f}\n")
    topo_status = "✅" if n_loops >= 2 else "⚠️"
    lines.append(f"- Status : {topo_status}\n\n")

    # === Étape 2 : Centerline géométrique (PCA sur centroïdes actifs) ===
    s = np.dot(centroids - centroids.mean(axis=0), axis)
    s_min_idx = np.argmin(s)
    s_max_idx = np.argmax(s)
    end1 = centroids[s_min_idx]
    end2 = centroids[s_max_idx]
    cl = centroids[np.argsort(s)]

    lines.append(f"## 2. Centerline géométrique\n")
    lines.append(f"- Axe : [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]\n")
    lines.append(f"- Extrémité 1 (s_min) : [{end1[0]:.4f}, {end1[1]:.4f}, {end1[2]:.4f}]\n")
    lines.append(f"- Extrémité 2 (s_max) : [{end2[0]:.4f}, {end2[1]:.4f}, {end2[2]:.4f}]\n")
    lines.append(f"- Status : ✅\n\n")

    # === Étape 3 : Normales locales + angle normale / tangente ===
    face_data = build_boundary_face_data(reader)
    centers = face_data["centers"]
    normals = face_data["normals"]
    areas = face_data["areas"]
    face_indices = face_data["face_indices"]

    angles = []
    for fi in face_indices:
        n = compute_face_normal(faces[fi], points)
        fc = points[faces[fi]].mean(axis=0)
        d = np.linalg.norm(centroids - fc, axis=1)
        li = np.argmin(d)
        laxis = axis
        if d[li] < 0.05:
            local = centroids[max(0, li - 2):li + 3]
            if len(local) > 3:
                pca = PCA(n_components=3)
                pca.fit(local)
                laxis = pca.components_[0]
        cos = np.clip(np.abs(np.dot(n, laxis)), -1, 1)
        angles.append(np.degrees(np.arccos(cos)))
    angles = np.array(angles)

    lines.append(f"## 3. Normales locales + angle normale/tangente\n")
    lines.append(f"- Angle moyen : {angles.mean():.2f}°\n")
    lines.append(f"- Angle min : {angles.min():.2f}°\n")
    lines.append(f"- Angle max : {angles.max():.2f}°\n")
    opening_candidates = np.where(angles < 25)[0]
    wall_candidates = np.where(angles > 65)[0]
    lines.append(f"- Faces candidate opening (angle < 25°) : **{len(opening_candidates)}**\n")
    lines.append(f"- Faces candidate wall (angle > 65°) : **{len(wall_candidates)}**\n")
    lines.append(f"- Status : ✅\n\n")

    # === Étape 4 : Forme des caps (aire, circularité, planarity) ===
    adjacency = {}
    pt_to_faces = {}
    for idx, fi in enumerate(face_indices):
        for pt in faces[fi]:
            pt_to_faces.setdefault(pt, []).append(idx)

    for idx, fi in enumerate(face_indices):
        face_pts = set(faces[fi])
        for pt in face_pts:
            for nb_idx in pt_to_faces.get(pt, []):
                if nb_idx != idx and np.linalg.norm(centers[idx] - centers[nb_idx]) < 0.05:
                    adjacency.setdefault(idx, set()).add(nb_idx)

    proj = centers @ axis
    sorted_idx = np.argsort(proj)
    min_seeds = sorted_idx[:5].tolist()
    max_seeds = sorted_idx[-5:].tolist()

    def region_growing_cap(seed_indices, max_angle_deg=30.0, max_dist=0.08):
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

    cap1_region = region_growing_cap(min_seeds)
    cap2_region = region_growing_cap(max_seeds)

    cap_regions = []
    for region in [cap1_region, cap2_region]:
        if len(region) < 3:
            continue
        ridx = list(region)
        rcenters = centers[ridx]
        rnormals = normals[ridx]
        rareas = areas[ridx]
        planarity = pca_planarity(rcenters)
        norm_cons = float(np.linalg.norm(rnormals.mean(axis=0)))
        total_area = float(rareas.sum())
        rcenter = rcenters.mean(axis=0)
        perimeter = 0.0
        for i in ridx:
            for j in adjacency.get(i, []):
                if j in region and i < j:
                    perimeter += np.linalg.norm(centers[i] - centers[j])
        comp = (4.0 * np.pi * total_area / (perimeter ** 2)) if perimeter > 0 else 0.0
        cap_regions.append({
            "region": region,
            "indices": ridx,
            "center": rcenter,
            "area": total_area,
            "planarity": planarity,
            "normal_consistency": norm_cons,
            "compactness": comp,
            "n_faces": len(ridx),
        })

    cap_regions.sort(key=lambda x: x["area"], reverse=True)

    lines.append(f"## 4. Forme des caps\n")
    lines.append(f"- Caps détectés : **{len(cap_regions)}**\n")
    for i, cap in enumerate(cap_regions):
        lines.append(f"- Cap {i + 1} : aire={cap['area']:.4f}, "
                     f"planarité={cap['planarity']:.4f}, "
                     f"normal_cons={cap['normal_consistency']:.4f}, "
                     f"compacité={cap['compactness']:.4f}, "
                     f"n_faces={cap['n_faces']}\n")
    shape_status = "✅" if len(cap_regions) >= 2 else "⚠️"
    lines.append(f"- Status : {shape_status}\n\n")

    # === Étape 5 : Validation topologique ===
    expected_openings = 2
    n_openings = len(cap_regions)
    topo_ok = n_openings >= expected_openings

    lines.append(f"## 5. Validation topologique\n")
    lines.append(f"- Nombre d'ouvertures attendu : >= {expected_openings}\n")
    lines.append(f"- Nombre d'ouvertures détectées : {n_openings}\n")
    lines.append(f"- Validation : {'✅ PASS' if topo_ok else '❌ FAIL'}\n\n")

    # === Étape 6 : Convention inlet/outlet ===
    opening_labels = []
    for i, cap in enumerate(cap_regions):
        dist_to_end1 = np.linalg.norm(cap["center"] - end1)
        dist_to_end2 = np.linalg.norm(cap["center"] - end2)
        if dist_to_end1 < dist_to_end2:
            label = "inlet"
        else:
            label = "outlet"
        opening_labels.append({
            "label": label,
            "center": cap["center"],
            "area": cap["area"],
            "dist_to_end1": float(dist_to_end1),
            "dist_to_end2": float(dist_to_end2),
        })

    lines.append(f"## 6. Convention inlet/outlet\n")
    lines.append(f"- Convention : extrémité s_min → inlet, extrémité s_max → outlet\n")
    for item in opening_labels:
        lines.append(f"- {item['label'].upper()} : centre=[{item['center'][0]:.4f}, {item['center'][1]:.4f}, {item['center'][2]:.4f}], "
                     f"aire={item['area']:.4f}\n")
    conv_status = "✅" if len(opening_labels) >= 2 else "❌"
    lines.append(f"- Status : {conv_status}\n\n")

    # === Résumé global ===
    overall = "✅" if all(s in ["✅", "⚠️"] for s in [topo_status, shape_status, conv_status]) and topo_ok else "❌"
    lines.append(f"## Résumé global\n")
    lines.append(f"- P1 (topologie) : {topo_status}\n")
    lines.append(f"- P2 (centerline) : ✅\n")
    lines.append(f"- P3 (normales/angle) : ✅\n")
    lines.append(f"- P4 (forme caps) : {shape_status}\n")
    lines.append(f"- P5 (validation topo) : {'✅' if topo_ok else '❌'}\n")
    lines.append(f"- P6 (convention) : {conv_status}\n")
    lines.append(f"- **Global : {overall}**\n")
    lines.append(f"\n> La géométrie peut donner 'opening_0' et 'opening_1'. "
                 f"Elle ne peut pas donner avec certitude 'inlet' et 'outlet' sans information supplémentaire.\n")

    write_results(24, "results_P24.md", "".join(lines))

    # === Image PyVista off_screen ===
    plotter = pv.Plotter(shape=(2, 3), off_screen=True, window_size=(1800, 1200))
    plotter.set_background("white")

    # Row 0, col 0: boundary loops
    plotter.subplot(0, 0)
    plotter.add_text("P24-1: Boundary loops")
    plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)
    colors = ["red", "blue", "green", "orange", "purple", "cyan"]
    for i, lp in enumerate(loop_props[:6]):
        pts = edges.points[np.array(loops[i])]
        n = len(pts)
        f = np.hstack([[n], np.arange(n)])
        poly = pv.PolyData(pts, faces=f)
        plotter.add_mesh(poly, color=colors[i % 6], opacity=0.9, show_edges=True, line_width=3)

    # Row 0, col 1: centerline + endpoints
    plotter.subplot(0, 1)
    plotter.add_text("P24-2: Centerline + endpoints")
    plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)
    plotter.add_mesh(pv.PolyData(cl), color="black", point_size=4, render_points_as_spheres=True)
    plotter.add_arrows(end1, axis * 0.05, mag=0.05, color="red")
    plotter.add_arrows(end2, axis * 0.05, mag=0.05, color="blue")

    # Row 0, col 2: normal-tangent angles
    plotter.subplot(0, 2)
    plotter.add_text("P24-3: Normal-tangent angles")
    plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)
    face_angles = []
    for fi in face_indices:
        n = compute_face_normal(faces[fi], points)
        fc = points[faces[fi]].mean(axis=0)
        d = np.linalg.norm(centroids - fc, axis=1)
        li = np.argmin(d)
        laxis = axis
        if d[li] < 0.05:
            local = centroids[max(0, li - 2):li + 3]
            if len(local) > 3:
                pca = PCA(n_components=3)
                pca.fit(local)
                laxis = pca.components_[0]
        cos = np.clip(np.abs(np.dot(n, laxis)), -1, 1)
        face_angles.append(np.degrees(np.arccos(cos)))
    face_angles = np.array(face_angles)
    if len(face_angles) == boundary_poly.n_cells:
        boundary_poly.cell_data["angle"] = face_angles
        plotter.add_mesh(boundary_poly, scalars="angle", cmap="coolwarm", show_edges=False)

    # Row 1, col 0: planar caps
    plotter.subplot(1, 0)
    plotter.add_text("P24-4: Plane caps")
    plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)
    cap_colors = ["red", "blue"]
    for i, cap in enumerate(cap_regions[:2]):
        cap_pts = centers[cap["indices"]]
        n = len(cap_pts)
        f = np.hstack([[n], np.arange(n)])
        cap_poly = pv.PolyData(cap_pts, faces=f)
        plotter.add_mesh(cap_poly, color=cap_colors[i % 2], opacity=0.9, show_edges=True, line_width=3)

    # Row 1, col 1: inlet/outlet convention
    plotter.subplot(1, 1)
    plotter.add_text("P24-5: Inlet/Outlet convention")
    plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)
    for i, item in enumerate(opening_labels):
        cap = cap_regions[i]
        cap_pts = centers[cap["indices"]]
        n = len(cap_pts)
        f = np.hstack([[n], np.arange(n)])
        cap_poly = pv.PolyData(cap_pts, faces=f)
        color = "green" if item["label"] == "inlet" else "blue"
        plotter.add_mesh(cap_poly, color=color, opacity=0.9, show_edges=True, line_width=3)
        plotter.add_text(f"{item['label'].upper()}", position="upper_left" if i == 0 else "upper_right")

    # Row 1, col 2: summary text
    plotter.subplot(1, 2)
    plotter.add_text("P24 Summary")
    summary_text = (
        f"Loops: {n_loops}\n"
        f"Caps: {len(cap_regions)}\n"
        f"Mean angle: {angles.mean():.1f}°\n"
        f"Topo OK: {topo_ok}\n"
        f"Convention: {conv_status}\n"
        f"Overall: {overall}"
    )
    plotter.add_text(summary_text, position="upper_left")

    plotter.view_isometric()
    img_path = Path(__file__).resolve().parent / "conclusion_P24.png"
    plotter.screenshot(str(img_path))
    plotter.close()
    print(f"  -> {img_path}")
    print("[P24] Done.")


if __name__ == "__main__":
    main()
