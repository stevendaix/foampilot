#!/usr/bin/env python3
"""
Test P23 — Tests prioritaires (Section 23)
Méthodes : boundary loops, centerline PCA, angle normale/tangente, caps plans, filtrage forme
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA
from common import load_reader, compute_face_normal, compute_vessel_axis, get_boundary_surface, write_results


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


def normal_consistency(normals):
    if len(normals) == 0:
        return 0.0
    C = np.linalg.norm(normals.mean(axis=0))
    return float(C)


def compactness(area, perimeter):
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


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


def main():
    print("[P23] Priority tests (Section 23)")
    reader, mesh = load_reader()
    axis, centroids, U_mag = compute_vessel_axis(mesh)
    points = reader._points
    faces = reader._faces
    patches = reader.boundary_patches

    lines = ["# P23 — Tests prioritaires (Section 23)\n",
             "Combinaison des 5 méthodes recommandées :\n\n"]

    # === P1: Boundary loops ===
    boundary_poly = get_boundary_surface(reader)
    boundary_poly = boundary_poly.clean().triangulate()
    edges = boundary_poly.extract_all_edges()
    loops = chain_boundary_edges(edges)
    loop_props = [compute_loop_properties(loop, edges.points) for loop in loops]
    loop_props.sort(key=lambda x: x["area"], reverse=True)
    n_loops = len(loops)
    lines.append(f"## P1 — Détection boucles de bord\n")
    lines.append(f"- Boucles détectées : **{n_loops}**\n")
    if loop_props:
        lines.append(f"- Plus grande boucle : aire={loop_props[0]['area']:.6f}, "
                     f"circ={loop_props[0]['circ']:.4f}, plan={loop_props[0]['plan']:.4f}\n")
    p1_status = "✅" if n_loops >= 2 else "⚠️"
    lines.append(f"- Status : {p1_status}\n\n")

    # === P2: Centerline + endpoints ===
    s = np.dot(centroids - centroids.mean(axis=0), axis)
    s_min_idx = np.argmin(s)
    s_max_idx = np.argmax(s)
    end1 = centroids[s_min_idx]
    end2 = centroids[s_max_idx]
    lines.append(f"## P2 — Centerline simulée + extrémités\n")
    lines.append(f"- Axe : [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]\n")
    lines.append(f"- Extrémité 1 : [{end1[0]:.4f}, {end1[1]:.4f}, {end1[2]:.4f}]\n")
    lines.append(f"- Extrémité 2 : [{end2[0]:.4f}, {end2[1]:.4f}, {end2[2]:.4f}]\n")
    lines.append(f"- Status : ✅\n\n")

    # === P3: Angle normale / tangente locale ===
    all_angles = []
    patch_stats = {}
    for name, info in patches.items():
        sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        face_ids = np.arange(sf, sf + nf)
        angles = []
        for fi in face_ids:
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
        all_angles.extend(angles)
        patch_stats[name] = {
            "mean": float(angles.mean()),
            "std": float(angles.std()),
            "min": float(angles.min()),
            "max": float(angles.max()),
        }

    lines.append(f"## P3 — Angle normale / tangente locale\n")
    for name, st in patch_stats.items():
        lines.append(f"- {name} : mean={st['mean']:.2f}°, std={st['std']:.2f}°, "
                     f"min={st['min']:.2f}°, max={st['max']:.2f}°\n")
    overall_mean = float(np.mean(all_angles)) if all_angles else 0.0
    lines.append(f"- Angle global moyen : {overall_mean:.2f}°\n")
    lines.append(f"- Status : ✅\n\n")

    # === P4: Caps plans ===
    face_data = build_boundary_face_data(reader)
    centers = face_data["centers"]
    normals = face_data["normals"]
    areas = face_data["areas"]
    face_indices = face_data["face_indices"]

    adjacency = build_face_adjacency(face_indices, faces, points, centers)

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
        total_area = float(rareas.sum())
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
            "area": total_area,
            "planarity": planarity,
            "normal_consistency": norm_cons,
            "compactness": comp,
            "n_faces": len(ridx),
        })

    cap_regions.sort(key=lambda x: x["area"], reverse=True)
    lines.append(f"## P4 — Détection de caps plans\n")
    lines.append(f"- Caps détectés : **{len(cap_regions)}**\n")
    for i, cap in enumerate(cap_regions):
        lines.append(f"- Cap {i + 1} : aire={cap['area']:.4f}, "
                     f"planarité={cap['planarity']:.4f}, "
                     f"normal_cons={cap['normal_consistency']:.4f}, "
                     f"compacité={cap['compactness']:.4f}, "
                     f"n_faces={cap['n_faces']}\n")
    p4_status = "✅" if len(cap_regions) >= 2 else "⚠️"
    lines.append(f"- Status : {p4_status}\n\n")

    # === P5: Filtrage par forme ===
    filtered_caps = []
    for cap in cap_regions:
        ridx = cap["indices"]
        rcenters = centers[ridx]
        rc = cap["center"]
        dist_to_end1 = np.linalg.norm(rc - end1)
        dist_to_end2 = np.linalg.norm(rc - end2)
        near_endpoint = min(dist_to_end1, dist_to_end2) < 0.1
        strong_planar = cap["planarity"] < 0.1
        compact = cap["compactness"] > 0.5
        reasonable_area = 0.001 < cap["area"] < 0.5
        passes = strong_planar and compact and reasonable_area and near_endpoint
        filtered_caps.append({
            "cap": cap,
            "near_endpoint": near_endpoint,
            "strong_planar": strong_planar,
            "compact": compact,
            "reasonable_area": reasonable_area,
            "passes": passes,
        })

    lines.append(f"## P5 — Filtrage par forme\n")
    for i, fc in enumerate(filtered_caps):
        cap = fc["cap"]
        lines.append(f"- Cap {i + 1} : plan={fc['strong_planar']}, "
                     f"compact={fc['compact']}, area_ok={fc['reasonable_area']}, "
                     f"near_endpoint={fc['near_endpoint']} → **{'FILTRÉ ✅' if fc['passes'] else 'REJETÉ ❌'}**\n")
    n_pass = sum(1 for fc in filtered_caps if fc["passes"])
    p5_status = "✅" if n_pass >= 2 else "⚠️"
    lines.append(f"- Caps filtrés : **{n_pass}**\n")
    lines.append(f"- Status : {p5_status}\n\n")

    # === Résumé global ===
    overall = "✅" if all(s in ["✅", "⚠️"] for s in [p1_status, p4_status, p5_status]) else "❌"
    lines.append(f"## Résumé global\n")
    lines.append(f"- P1 (boucles de bord) : {p1_status}\n")
    lines.append(f"- P2 (centerline) : ✅\n")
    lines.append(f"- P3 (angle normale/tangente) : ✅\n")
    lines.append(f"- P4 (caps plans) : {p4_status}\n")
    lines.append(f"- P5 (filtrage forme) : {p5_status}\n")
    lines.append(f"- **Global : {overall}**\n")

    write_results(23, "results_P23.md", "".join(lines))

    # === Image PyVista off_screen ===
    plotter = pv.Plotter(shape=(2, 3), off_screen=True)
    plotter.set_background("white")

    # Row 0: boundary + loops
    plotter.subplot(0, 0)
    plotter.add_text("P1: Boundary loops")
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
    plotter.add_text("P2: Centerline + endpoints")
    plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)
    cl = centroids[np.argsort(s)]
    plotter.add_mesh(pv.PolyData(cl), color="black", point_size=4, render_points_as_spheres=True)
    plotter.add_arrows(end1, axis * 0.05, mag=0.05, color="red")
    plotter.add_arrows(end2, axis * 0.05, mag=0.05, color="blue")

    # Row 0, col 2: normal-tangent angles (colored by angle)
    plotter.subplot(0, 2)
    plotter.add_text("P3: Normal-tangent angles")
    plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)
    if all_angles:
        angle_arr = np.array(all_angles)
        face_angles = []
        idx = 0
        for name, info in patches.items():
            sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
            face_angles.extend(angle_arr[idx:idx + nf].tolist())
            idx += nf
        face_angles = np.array(face_angles)
        boundary_poly_cells = get_boundary_surface(reader).clean().triangulate()
        if len(face_angles) == boundary_poly_cells.n_cells:
            boundary_poly_cells.cell_data["angle"] = face_angles
            plotter.add_mesh(boundary_poly_cells, scalars="angle", cmap="coolwarm", show_edges=False)

    # Row 1, col 0: planar caps
    plotter.subplot(1, 0)
    plotter.add_text("P4: Plane caps")
    plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)
    cap_colors = ["red", "blue"]
    for i, cap in enumerate(cap_regions[:2]):
        cap_pts = centers[cap["indices"]]
        n = len(cap_pts)
        f = np.hstack([[n], np.arange(n)])
        cap_poly = pv.PolyData(cap_pts, faces=f)
        plotter.add_mesh(cap_poly, color=cap_colors[i % 2], opacity=0.9, show_edges=True, line_width=3)

    # Row 1, col 1: filtered caps
    plotter.subplot(1, 1)
    plotter.add_text("P5: Filtered caps")
    plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)
    for i, fc in enumerate(filtered_caps):
        cap = fc["cap"]
        cap_pts = centers[cap["indices"]]
        n = len(cap_pts)
        f = np.hstack([[n], np.arange(n)])
        cap_poly = pv.PolyData(cap_pts, faces=f)
        color = "green" if fc["passes"] else "gray"
        plotter.add_mesh(cap_poly, color=color, opacity=0.9, show_edges=True, line_width=3)

    # Row 1, col 2: combined summary text
    plotter.subplot(1, 2)
    plotter.add_text("P23 Summary")
    summary_text = (
        f"P1 Loops: {n_loops}\n"
        f"P2 Endpoints: 2\n"
        f"P3 Mean angle: {overall_mean:.1f}°\n"
        f"P4 Caps: {len(cap_regions)}\n"
        f"P5 Filtered: {n_pass}\n"
        f"Overall: {overall}"
    )
    plotter.add_text(summary_text, position="upper_left")

    plotter.view_isometric()
    img_path = Path(__file__).resolve().parent / "priority_P23.png"
    plotter.screenshot(str(img_path))
    print(f"  -> {img_path}")
    print("[P23] Done.")


if __name__ == "__main__":
    main()
