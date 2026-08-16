#!/usr/bin/env python3
"""
Visualisation honnête : montrer exactement ce que chaque méthode trouve.
- P11, P13, P18 : patches OpenFOAM existants (376 inlet, 2512 outlet)
- P02 : extrémités centerline (pas de faces)
- P04, P23, P24 : caps par région croissante (50 + 12 faces)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from foampilot.postprocess import OpenFOAMDirectReader

CASE_DIR = Path(__file__).resolve().parent.parent
TIME_STEP = "500"
OUT_DIR = Path(__file__).resolve().parent / "method_images_true"
OUT_DIR.mkdir(exist_ok=True)


def load_reader():
    return OpenFOAMDirectReader(case_path=CASE_DIR)


def compute_vessel_axis(mesh):
    U = mesh.cell_data["U"]
    U_mag = np.linalg.norm(U, axis=1)
    active = U_mag > 0.01
    centroids = mesh.cell_centers().points[active]
    pca = PCA(n_components=3)
    pca.fit(centroids)
    axis = pca.components_[0]
    if axis[2] > 0:
        axis = -axis
    return axis / np.linalg.norm(axis), centroids


def build_boundary_data(reader):
    points = reader._points
    faces = reader._faces
    patches = reader.boundary_patches

    centers = []
    face_ids = []
    for name, info in patches.items():
        sf = info.get("startFace", 0)
        nf = info.get("nFaces", 0)
        for fi in range(sf, sf + nf):
            centers.append(points[faces[fi]].mean(axis=0))
            face_ids.append(fi)

    centers = np.array(centers, dtype=float)
    face_ids = np.array(face_ids, dtype=int)
    face_id_to_idx = {fid: i for i, fid in enumerate(face_ids)}
    return centers, face_ids, face_id_to_idx


def get_patch_face_ids(reader, patch_name):
    patches = reader.boundary_patches
    if patch_name not in patches:
        return []
    info = patches[patch_name]
    sf = info.get("startFace", 0)
    nf = info.get("nFaces", 0)
    return list(range(sf, sf + nf))


def compute_face_normal(face_vertex_indices, points):
    face = face_vertex_indices
    if len(face) < 3:
        return np.array([0.0, 0.0, 1.0])
    pts = points[face]
    normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return normal / norm


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
            face_indices.append(fi)

    return {
        "centers": np.array(centers),
        "normals": np.array(normals),
        "areas": np.array(areas),
        "face_indices": np.array(face_indices),
    }


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


def method_p04_region_growing(reader, mesh):
    axis, centroids = compute_vessel_axis(mesh)
    face_data = build_boundary_face_data(reader)
    centers = face_data["centers"]
    normals = face_data["normals"]
    face_indices = face_data["face_indices"]

    adjacency = build_face_adjacency(face_indices, reader._faces, reader._points, centers)

    proj = centers @ axis
    sorted_idx = np.argsort(proj)
    min_seeds = sorted_idx[:5].tolist()
    max_seeds = sorted_idx[-5:].tolist()

    cap1_region = region_growing_cap(min_seeds, centers, normals, adjacency, max_angle_deg=30.0, max_dist=0.08)
    cap2_region = region_growing_cap(max_seeds, centers, normals, adjacency, max_angle_deg=30.0, max_dist=0.08)

    cap1_faces = set(face_indices[list(cap1_region)].tolist()) if len(cap1_region) >= 3 else set()
    cap2_faces = set(face_indices[list(cap2_region)].tolist()) if len(cap2_region) >= 3 else set()

    s1 = np.dot(centers[list(cap1_region)].mean(axis=0) - centers.mean(axis=0), axis) if cap1_region else 0
    s2 = np.dot(centers[list(cap2_region)].mean(axis=0) - centers.mean(axis=0), axis) if cap2_region else 0

    if s1 < s2:
        inlet_faces, outlet_faces = cap1_faces, cap2_faces
    else:
        inlet_faces, outlet_faces = cap2_faces, cap1_faces

    return {
        "inlet_faces": inlet_faces,
        "outlet_faces": outlet_faces,
        "cap1_faces": cap1_faces,
        "cap2_faces": cap2_faces,
        "axis": axis,
    }


def save_method_image_2d(all_centers_2d, boundary_ids, face_id_to_idx, inlet_faces, outlet_faces, title, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor="white")
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    n_bg = len(all_centers_2d)
    boundary_idx = np.array(sorted(boundary_ids))
    boundary_idx = boundary_idx[boundary_idx < n_bg]
    if len(boundary_idx) > 0:
        ax.scatter(
            all_centers_2d[boundary_idx, 0],
            all_centers_2d[boundary_idx, 1],
            c="#cccccc",
            s=1.0,
            alpha=0.4,
            marker=".",
            linewidths=0,
        )

    inlet_idx = np.array(sorted([face_id_to_idx[fid] for fid in inlet_faces if fid in face_id_to_idx]))
    inlet_idx = inlet_idx[inlet_idx < n_bg]
    if len(inlet_idx) > 0:
        ax.scatter(
            all_centers_2d[inlet_idx, 0],
            all_centers_2d[inlet_idx, 1],
            c="#0066cc",
            s=45.0,
            alpha=0.95,
            marker="o",
            edgecolors="#003366",
            linewidths=0.8,
            label=f"Inlet ({len(inlet_idx)} faces)",
        )

    outlet_idx = np.array(sorted([face_id_to_idx[fid] for fid in outlet_faces if fid in face_id_to_idx]))
    outlet_idx = outlet_idx[outlet_idx < n_bg]
    if len(outlet_idx) > 0:
        ax.scatter(
            all_centers_2d[outlet_idx, 0],
            all_centers_2d[outlet_idx, 1],
            c="#cc2200",
            s=45.0,
            alpha=0.95,
            marker="o",
            edgecolors="#661100",
            linewidths=0.8,
            label=f"Outlet ({len(outlet_idx)} faces)",
        )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", fontsize=10, framealpha=0.9)

    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved {out_path}")


def save_p02_image_2d(all_centers_2d, boundary_ids, face_id_to_idx, inlet_pt, outlet_pt, pca2d, title, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor="white")
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    n_bg = len(all_centers_2d)
    boundary_idx = np.array(sorted(boundary_ids))
    boundary_idx = boundary_idx[boundary_idx < n_bg]
    if len(boundary_idx) > 0:
        ax.scatter(
            all_centers_2d[boundary_idx, 0],
            all_centers_2d[boundary_idx, 1],
            c="#cccccc",
            s=1.0,
            alpha=0.4,
            marker=".",
            linewidths=0,
        )

    inlet_2d = pca2d.transform(inlet_pt.reshape(1, -1))
    outlet_2d = pca2d.transform(outlet_pt.reshape(1, -1))

    ax.scatter(
        inlet_2d[:, 0], inlet_2d[:, 1],
        c="#0066cc",
        s=250.0,
        alpha=1.0,
        marker="o",
        edgecolors="#003366",
        linewidths=2.5,
        label=f"Inlet: [{inlet_pt[0]:.3f}, {inlet_pt[1]:.3f}, {inlet_pt[2]:.3f}]",
    )
    ax.scatter(
        outlet_2d[:, 0], outlet_2d[:, 1],
        c="#cc2200",
        s=250.0,
        alpha=1.0,
        marker="o",
        edgecolors="#661100",
        linewidths=2.5,
        label=f"Outlet: [{outlet_pt[0]:.3f}, {outlet_pt[1]:.3f}, {outlet_pt[2]:.3f}]",
    )

    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    print("Loading mesh and boundary data...")
    reader = load_reader()
    mesh = reader.to_pyvista(fields=["U", "p"], time_step=TIME_STEP, as_point_data=False)

    boundary_centers, boundary_ids, face_id_to_idx = build_boundary_data(reader)
    print(f"  Boundary faces: {len(boundary_ids)}")
    print(f"  Boundary center array shape: {boundary_centers.shape}")

    pca2d = PCA(n_components=2)
    all_centers_2d = pca2d.fit_transform(boundary_centers)

    axis, centroids = compute_vessel_axis(mesh)

    # P11 - EXISTING patches only
    p11_inlet = set(get_patch_face_ids(reader, "INLET"))
    p11_outlet = set(get_patch_face_ids(reader, "OUTLET"))
    save_method_image_2d(
        all_centers_2d,
        boundary_ids,
        face_id_to_idx,
        p11_inlet,
        p11_outlet,
        f"P11 — Patches OpenFOAM existants\nINLET: {len(p11_inlet)} faces | OUTLET: {len(p11_outlet)} faces",
        OUT_DIR / "P11_inlet_outlet.png",
    )

    # P13 - SAME as P11 (just reads existing patches)
    save_method_image_2d(
        all_centers_2d,
        boundary_ids,
        face_id_to_idx,
        p11_inlet,
        p11_outlet,
        f"P13 — Graphe topologique (lit les patches)\nINLET: {len(p11_inlet)} faces | OUTLET: {len(p11_outlet)} faces",
        OUT_DIR / "P13_inlet_outlet.png",
    )

    # P18 - SAME as P11 (just convention, no detection)
    save_method_image_2d(
        all_centers_2d,
        boundary_ids,
        face_id_to_idx,
        p11_inlet,
        p11_outlet,
        f"P18 — Convention s_min/s_max (lit les patches)\nINLET: {len(p11_inlet)} faces | OUTLET: {len(p11_outlet)} faces",
        OUT_DIR / "P18_inlet_outlet.png",
    )

    # P04 - Region growing caps
    p04 = method_p04_region_growing(reader, mesh)
    save_method_image_2d(
        all_centers_2d,
        boundary_ids,
        face_id_to_idx,
        p04["inlet_faces"],
        p04["outlet_faces"],
        f"P04 — Région croissante\nInlet: {len(p04['inlet_faces'])} faces | Outlet: {len(p04['outlet_faces'])} faces",
        OUT_DIR / "P04_inlet_outlet.png",
    )

    # P23 - SAME as P04 (same region growing)
    save_method_image_2d(
        all_centers_2d,
        boundary_ids,
        face_id_to_idx,
        p04["inlet_faces"],
        p04["outlet_faces"],
        f"P23 — Tests prioritaires (même détection que P04)\nInlet: {len(p04['inlet_faces'])} faces | Outlet: {len(p04['outlet_faces'])} faces",
        OUT_DIR / "P23_inlet_outlet.png",
    )

    # P24 - SAME as P04 (same region growing)
    save_method_image_2d(
        all_centers_2d,
        boundary_ids,
        face_id_to_idx,
        p04["inlet_faces"],
        p04["outlet_faces"],
        f"P24 — Conclusion (même détection que P04)\nInlet: {len(p04['inlet_faces'])} faces | Outlet: {len(p04['outlet_faces'])} faces",
        OUT_DIR / "P24_inlet_outlet.png",
    )

    # P02 - Centerline endpoints (no faces, just points)
    p02 = {"inlet": centroids[np.argmin(np.dot(centroids - centroids.mean(axis=0), axis))],
           "outlet": centroids[np.argmax(np.dot(centroids - centroids.mean(axis=0), axis))]}
    save_p02_image_2d(
        all_centers_2d,
        boundary_ids,
        face_id_to_idx,
        p02["inlet"],
        p02["outlet"],
        pca2d,
        "P02 — Centerline PCA\nExtrémités inlet/outlet (pas de faces)",
        OUT_DIR / "P02_inlet_outlet.png",
    )

    print("\nSummary:")
    print(f"  P11/P13/P18: {len(p11_inlet)} inlet + {len(p11_outlet)} outlet = {len(p11_inlet)+len(p11_outlet)} faces (patches existants)")
    print(f"  P04/P23/P24: {len(p04['inlet_faces'])} inlet + {len(p04['outlet_faces'])} outlet = {len(p04['inlet_faces'])+len(p04['outlet_faces'])} faces (région croissante)")
    print(f"  P02: 0 faces (extrémités centerline)")
    print("\nAll images saved to", OUT_DIR)


if __name__ == "__main__":
    main()
