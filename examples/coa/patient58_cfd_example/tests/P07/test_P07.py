#!/usr/bin/env python3
"""
Test P07 — Region growing par similarité de normales (Section 7.1)
Plan : plan_test_inlet.md section 07
"""
import sys
from pathlib import Path
import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_reader, compute_vessel_axis, get_boundary_surface, write_results


def build_adjacency(boundary_poly):
    faces = boundary_poly.faces
    idx = 0
    edge_to_faces = {}
    for cell_id in range(boundary_poly.n_cells):
        n = faces[idx]
        pts = faces[idx + 1:idx + 1 + n]
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            edge = (min(a, b), max(a, b))
            edge_to_faces.setdefault(edge, []).append(cell_id)
        idx += 1 + n

    adjacency = {}
    for edge, cells in edge_to_faces.items():
        if len(cells) == 2:
            a, b = cells
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
    return adjacency


def region_growing_by_normals(normals, adjacency, angle_threshold_deg=25.0):
    angle_threshold = np.cos(np.deg2rad(angle_threshold_deg))
    n = len(normals)
    visited = np.zeros(n, dtype=bool)
    labels = np.full(n, -1, dtype=int)
    regions = []

    for seed in range(n):
        if visited[seed]:
            continue
        label = len(regions)
        queue = [seed]
        visited[seed] = True
        region = []

        while queue:
            face = queue.pop()
            labels[face] = label
            region.append(face)
            for neighbor in adjacency.get(face, []):
                if not visited[neighbor]:
                    cos_angle = np.dot(normals[face], normals[neighbor])
                    if cos_angle > angle_threshold:
                        visited[neighbor] = True
                        queue.append(neighbor)
        regions.append(region)

    return labels, regions


def analyze_regions(boundary_poly, labels, regions, axis):
    cell_sizes = boundary_poly.compute_cell_sizes()
    areas = cell_sizes.cell_data["Area"]
    centroids = boundary_poly.cell_centers().points

    region_info = []
    for label, region in enumerate(regions):
        region_areas = areas[region]
        region_centroids = centroids[region]
        region_normals = labels  # we'll get normals from the poly

        total_area = region_areas.sum()
        centroid = region_centroids.mean(axis=0)
        n_faces = len(region)

        # Position along axis
        axis_proj = np.dot(region_centroids - centroid, axis)
        extent_min = axis_proj.min()
        extent_max = axis_proj.max()
        center_proj = np.dot(centroid, axis)

        region_info.append({
            "label": label,
            "n_faces": n_faces,
            "area": total_area,
            "centroid": centroid,
            "axis_proj_center": center_proj,
            "axis_extent": extent_max - extent_min,
        })

    return region_info


def main():
    print(f"[P07] Starting region growing by normal similarity...")
    reader, mesh = load_reader()
    print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} cells")

    axis, centroids, U_mag = compute_vessel_axis(mesh)
    print(f"Axis: {axis}")

    boundary_poly = get_boundary_surface(reader)
    boundary_poly = boundary_poly.compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=True,
        non_manifold_traversal=False,
        inplace=True,
    )
    normals = boundary_poly.cell_data["Normals"]

    print(f"Boundary faces: {boundary_poly.n_cells}")
    adjacency = build_adjacency(boundary_poly)
    print(f"Adjacency built: {len(adjacency)} faces with neighbors")

    labels, regions = region_growing_by_normals(normals, adjacency, angle_threshold_deg=25.0)
    print(f"Regions found: {len(regions)}")

    region_info = analyze_regions(boundary_poly, labels, regions, axis)
    region_info.sort(key=lambda x: x["area"], reverse=True)

    # Identify candidate openings (caps) vs wall
    # Caps: small area, high position extent along axis (at extremes)
    # Wall: large area, low extent along axis, central position
    all_centroids = np.array([r["centroid"] for r in region_info])
    all_areas = np.array([r["area"] for r in region_info])
    all_proj = np.array([r["axis_proj_center"] for r in region_info])

    # Heuristic: openings are typically at extremes of axis projection
    if len(all_proj) > 0:
        proj_min, proj_max = all_proj.min(), all_proj.max()
        proj_range = proj_max - proj_min if proj_max > proj_min else 1.0
    else:
        proj_min = proj_max = 0.0
        proj_range = 1.0

    cap_candidates = []
    wall_candidates = []
    for info in region_info:
        rel_pos = (info["axis_proj_center"] - proj_min) / proj_range if proj_range > 0 else 0.5
        # Simple heuristic: small area + at extreme positions = cap
        if info["area"] < all_areas.mean() * 0.5 and (rel_pos < 0.2 or rel_pos > 0.8):
            info["type"] = "cap_candidate"
            cap_candidates.append(info)
        else:
            info["type"] = "wall_candidate"
            wall_candidates.append(info)

    print(f"Cap candidates: {len(cap_candidates)}, Wall candidates: {len(wall_candidates)}")

    # Write results
    lines = ["# P07 — Region growing par similarité de normales\n",
             f"- Angle seuil : 25°\n",
             f"- Régions détectées : **{len(regions)}**\n",
             f"- Candidats cap (ouverture) : **{len(cap_candidates)}**\n",
             f"- Candidats paroi : **{len(wall_candidates)}**\n\n"]
    
    lines.append("## Top régions par aire\n")
    for i, info in enumerate(region_info[:10]):
        lines.append(f"### Région {info['label']} ({info['type']})\n")
        lines.append(f"- Faces : {info['n_faces']}\n")
        lines.append(f"- Aire : {info['area']:.6f}\n")
        lines.append(f"- Centre : [{info['centroid'][0]:.4f}, {info['centroid'][1]:.4f}, {info['centroid'][2]:.4f}]\n")
        lines.append(f"- Projection axe : {info['axis_proj_center']:.4f}\n")
        lines.append(f"- Type : {info['type']}\n\n")

    write_results(7, "results_P07.md", "".join(lines))

    # Visualization
    try:
        boundary_poly.cell_data["Region"] = labels
        plotter = pv.Plotter(shape=(1, 1), off_screen=True)
        plotter.set_background("white")
        
        # Color regions
        n_regions = len(regions)
        if n_regions > 0:
            cmap = "tab20"
            plotter.add_mesh(
                boundary_poly,
                scalars="Region",
                cmap=cmap,
                show_edges=False,
                opacity=0.9,
            )
        else:
            plotter.add_mesh(boundary_poly, color="lightgray", opacity=0.4)

        plotter.add_text("P07: Region growing (normals)", position="upper_left")
        plotter.view_isometric()

        img_path = Path(__file__).resolve().parent / "region_growing_P07.png"
        plotter.screenshot(str(img_path), window_size=(1600, 1200))
        plotter.close()
        print(f"  -> {img_path}")
    except Exception as e:
        print(f"Plot error: {e}")
        import traceback
        traceback.print_exc()

    print(f"[P07] Done.")


if __name__ == "__main__":
    main()
