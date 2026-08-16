#!/usr/bin/env python3
"""
Test P08 — Clustering KMeans / DBSCAN sur features de faces (plan_test_inlet.md section 8)
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import load_reader, compute_face_normal, compute_vessel_axis, get_boundary_surface, write_results
import pyvista as pv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

pv.OFF_SCREEN = True

def compute_face_features(boundary_poly, axis):
    n_faces = boundary_poly.n_cells
    face_centers = boundary_poly.cell_centers().points

    face_normals = None
    try:
        normals_poly = boundary_poly.compute_normals(
            cell_normals=True,
            point_normals=False,
            consistent_normals=True,
            inplace=False
        )
        if "Normals" in normals_poly.cell_data:
            face_normals = normals_poly.cell_data["Normals"]
    except Exception:
        pass

    if face_normals is None or len(face_normals) != n_faces:
        face_normals = np.zeros((n_faces, 3))
        for i in range(n_faces):
            face_verts = boundary_poly.get_cell(i).points
            face_normals[i] = compute_face_normal(np.arange(len(face_verts)), face_verts)

    face_areas = None
    try:
        sizes_poly = boundary_poly.compute_cell_sizes(volume=False, area=True)
        if "Area" in sizes_poly.cell_data:
            face_areas = sizes_poly.cell_data["Area"]
    except Exception:
        pass

    if face_areas is None or len(face_areas) != n_faces:
        face_areas = np.zeros(n_faces)
        for i in range(n_faces):
            face_verts = boundary_poly.get_cell(i).points
            if len(face_verts) >= 3:
                v1 = face_verts[1] - face_verts[0]
                v2 = face_verts[2] - face_verts[0]
                face_areas[i] = 0.5 * np.linalg.norm(np.cross(v1, v2))

    face_dot_axis = np.abs(np.dot(face_normals, axis))

    curvatures = np.zeros(n_faces)
    try:
        tmp = boundary_poly.compute_curvature(curvature_type="Mean")
        curvatures = tmp.cell_data["Curvature"]
        curvatures = np.nan_to_num(curvatures, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        pass

    planarities = np.zeros(n_faces)
    for i in range(n_faces):
        pts = face_centers[i:i+1]
        if pts.shape[0] >= 3:
            pca = PCA(n_components=3)
            pca.fit(pts)
            lam = pca.explained_variance_
            if lam.sum() > 0:
                planarities[i] = lam[2] / lam.sum()

    pos_along_axis = np.dot(face_centers, axis)
    distances_along_axis = pos_along_axis.max() - pos_along_axis.min()
    if distances_along_axis > 0:
        pos_along_axis_norm = (pos_along_axis - pos_along_axis.min()) / distances_along_axis
    else:
        pos_along_axis_norm = np.zeros(n_faces)

    features = np.column_stack([
        face_dot_axis,
        np.log1p(face_areas),
        curvatures,
        planarities,
        pos_along_axis_norm,
    ])

    return features, face_centers, face_normals, face_areas

def main():
    print("[P08] Starting clustering test...")
    reader, mesh = load_reader()
    print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} cells")

    axis, centroids, U_mag = compute_vessel_axis(mesh)
    print(f"Axis: {axis}")

    boundary_poly = get_boundary_surface(reader)
    features, face_centers, face_normals, face_areas = compute_face_features(boundary_poly, axis)
    print(f"Boundary faces: {boundary_poly.n_cells}, features shape: {features.shape}")

    features_scaled = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(features_scaled)

    dbscan = DBSCAN(eps=1.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(features_scaled)

    def characterize_cluster(labels, centers):
        clusters = {}
        for lab in np.unique(labels):
            if lab == -1:
                continue
            mask = labels == lab
            if mask.sum() == 0:
                continue
            centers_cluster = face_centers[mask]
            normals_cluster = face_normals[mask]
            areas_cluster = face_areas[mask]
            dot_vals = features[mask, 0]
            clusters[int(lab)] = {
                "n_faces": int(mask.sum()),
                "total_area": float(areas_cluster.sum()),
                "mean_dot_axis": float(dot_vals.mean()),
                "std_dot_axis": float(dot_vals.std()),
                "bbox_min": centers_cluster.min(axis=0).tolist(),
                "bbox_max": centers_cluster.max(axis=0).tolist(),
            }
        return clusters

    kmeans_clusters = characterize_cluster(kmeans_labels, face_centers)
    dbscan_clusters = characterize_cluster(dbscan_labels, face_centers)

    print("KMeans clusters:")
    for k, v in kmeans_clusters.items():
        print(f"  {k}: n_faces={v['n_faces']}, area={v['total_area']:.4f}, dot_axis={v['mean_dot_axis']:.4f}")
    print("DBSCAN clusters:")
    for k, v in dbscan_clusters.items():
        print(f"  {k}: n_faces={v['n_faces']}, area={v['total_area']:.4f}, dot_axis={v['mean_dot_axis']:.4f}")

    def infer_labels(cluster_info):
        opening_label = None
        wall_label = None
        for lab, info in cluster_info.items():
            if info["mean_dot_axis"] > 0.7 and opening_label is None:
                opening_label = lab
            elif info["mean_dot_axis"] < 0.4 and wall_label is None:
                wall_label = lab
        if opening_label is None:
            for lab, info in cluster_info.items():
                if info["mean_dot_axis"] == max(v["mean_dot_axis"] for v in cluster_info.values()):
                    opening_label = lab
                    break
        if wall_label is None and opening_label is not None:
            for lab, info in cluster_info.items():
                if lab != opening_label:
                    wall_label = lab
                    break
        return opening_label, wall_label

    kmeans_opening, kmeans_wall = infer_labels(kmeans_clusters)
    dbscan_opening, dbscan_wall = infer_labels(dbscan_clusters)

    print(f"KMeans inferred: opening_label={kmeans_opening}, wall_label={kmeans_wall}")
    print(f"DBSCAN inferred: opening_label={dbscan_opening}, wall_label={dbscan_wall}")

    plotter = pv.Plotter(shape=(1, 2), off_screen=True)
    plotter.set_background("white")

    plotter.subplot(0, 0)
    plotter.add_text("KMeans (3 clusters)", position="upper_left")
    colors = {0: (1, 0, 0), 1: (0, 0, 1), 2: (0, 1, 0), -1: (0.5, 0.5, 0.5)}
    for lab in np.unique(kmeans_labels):
        mask = kmeans_labels == lab
        if mask.sum() == 0:
            continue
        pts = face_centers[mask]
        faces_list = []
        faces_list.append(len(pts))
        faces_list.extend(np.arange(len(pts)))
        poly = pv.PolyData(pts, faces=np.array(faces_list))
        plotter.add_mesh(poly, color=colors.get(lab, (0.5, 0.5, 0.5)), opacity=0.8)
    plotter.view_isometric()

    plotter.subplot(0, 1)
    plotter.add_text("DBSCAN", position="upper_left")
    for lab in np.unique(dbscan_labels):
        mask = dbscan_labels == lab
        if mask.sum() == 0:
            continue
        pts = face_centers[mask]
        faces_list = []
        faces_list.append(len(pts))
        faces_list.extend(np.arange(len(pts)))
        poly = pv.PolyData(pts, faces=np.array(faces_list))
        plotter.add_mesh(poly, color=colors.get(lab, (0.5, 0.5, 0.5)), opacity=0.8)
    plotter.view_isometric()

    img_path = Path(__file__).resolve().parent / "clustering_P08.png"
    plotter.screenshot(str(img_path), window_size=(1600, 600))
    plotter.close()
    print(f"  -> {img_path}")

    n_noise = int(np.sum(dbscan_labels == -1))
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

    result_md = f"""# P08 — Results — Clustering KMeans/DBSCAN (section 8)

## Vessel axis
{axis}

## KMeans (n_clusters=3)
| label | n_faces | total_area | mean_dot_axis | std_dot_axis |
|-------|---------|------------|---------------|--------------|
"""
    for lab, info in sorted(kmeans_clusters.items()):
        result_md += f"| {lab} | {info['n_faces']} | {info['total_area']:.4f} | {info['mean_dot_axis']:.4f} | {info['std_dot_axis']:.4f} |\n"
    result_md += f"\nInferred opening label: {kmeans_opening}\nInferred wall label: {kmeans_wall}\n"

    result_md += f"""
## DBSCAN (eps=1.5, min_samples=5)
| label | n_faces | total_area | mean_dot_axis | std_dot_axis |
|-------|---------|------------|---------------|--------------|
"""
    for lab, info in sorted(dbscan_clusters.items()):
        result_md += f"| {lab} | {info['n_faces']} | {info['total_area']:.4f} | {info['mean_dot_axis']:.4f} | {info['std_dot_axis']:.4f} |\n"
    result_md += f"\nNoise faces: {n_noise}\nInferred opening label: {dbscan_opening}\nInferred wall label: {dbscan_wall}\n"

    result_md += f"""
## Image
{img_path.name}

## Status
completed
"""
    write_results(8, "results_P08.md", result_md)
    print("[P08] Done.")

if __name__ == "__main__":
    main()
