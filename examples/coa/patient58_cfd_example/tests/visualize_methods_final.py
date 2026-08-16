#!/usr/bin/env python3
"""
Visualisation finale : scatter plot des centres de faces par méthode.
Sous-échantillonnage + figure multipanneaux.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from foampilot.postprocess import OpenFOAMDirectReader

CASE_DIR = Path(__file__).resolve().parent.parent
TIME_STEP = "500"
OUT_DIR = Path(__file__).resolve().parent / "method_images_final"
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


def get_patch_face_ids(reader, patch_name):
    patches = reader.boundary_patches
    if patch_name not in patches:
        return []
    info = patches[patch_name]
    sf = info.get("startFace", 0)
    nf = info.get("nFaces", 0)
    return list(range(sf, sf + nf))


def method_p02_centerline(reader, mesh):
    axis, centroids = compute_vessel_axis(mesh)
    s = np.dot(centroids - centroids.mean(axis=0), axis)
    s_min_idx = np.argmin(s)
    s_max_idx = np.argmax(s)
    end1 = centroids[s_min_idx]
    end2 = centroids[s_max_idx]
    return {"inlet": end1, "outlet": end2, "axis": axis}


def method_p04_plane_caps(reader, mesh):
    axis, centroids = compute_vessel_axis(mesh)
    points = reader._points
    faces = reader._faces
    patches = reader.boundary_patches
    centers = []
    for name, info in patches.items():
        sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        for fi in range(sf, sf + nf):
            centers.append(points[faces[fi]].mean(axis=0))
    centers = np.array(centers)
    proj = centers @ axis
    sorted_idx = np.argsort(proj)
    min_seeds = sorted_idx[:5].tolist()
    max_seeds = sorted_idx[-5:].tolist()
    cap1_center = centers[min_seeds].mean(axis=0)
    cap2_center = centers[max_seeds].mean(axis=0)
    s1 = np.dot(cap1_center - centroids.mean(axis=0), axis)
    s2 = np.dot(cap2_center - centroids.mean(axis=0), axis)
    if s1 < s2:
        inlet_center, outlet_center = cap1_center, cap2_center
    else:
        inlet_center, outlet_center = cap2_center, cap1_center
    return {"inlet": inlet_center, "outlet": outlet_center, "axis": axis}


def method_p11_openfoam_patches(reader, mesh):
    axis, _ = compute_vessel_axis(mesh)
    inlet_ids = get_patch_face_ids(reader, "INLET")
    outlet_ids = get_patch_face_ids(reader, "OUTLET")
    return {
        "inlet_faces": inlet_ids,
        "outlet_faces": outlet_ids,
        "axis": axis,
    }


def method_p13_topological_graph(reader, mesh):
    axis, centroids = compute_vessel_axis(mesh)
    points = reader._points
    faces = reader._faces
    patches = reader.boundary_patches

    pt_to_faces = {}
    for fi in range(len(faces)):
        for pt in faces[fi]:
            pt_to_faces.setdefault(pt, []).append(fi)

    adjacency = {}
    for fi in range(len(faces)):
        face_pts = set(faces[fi])
        neighbors = set()
        for pt in face_pts:
            for nb in pt_to_faces.get(pt, []):
                if nb != fi:
                    neighbors.add(nb)
        adjacency[fi] = neighbors

    degrees = {fi: len(nb) for fi, nb in adjacency.items()}
    terminals = [fi for fi, d in degrees.items() if d == 1]
    terminal_centers = np.array([points[faces[fi]].mean(axis=0) for fi in terminals])

    if len(terminal_centers) >= 2:
        proj = terminal_centers @ axis
        sorted_idx = np.argsort(proj)
        opening0_center = terminal_centers[sorted_idx[0]]
        opening1_center = terminal_centers[sorted_idx[-1]]
    else:
        opening0_center = opening1_center = terminal_centers[0] if len(terminal_centers) > 0 else np.array([0, 0, 0])

    return {
        "openings": len(terminals),
        "opening0_center": opening0_center,
        "opening1_center": opening1_center,
        "axis": axis,
    }


def method_p18_convention(reader, mesh):
    axis, centroids = compute_vessel_axis(mesh)
    s = np.dot(centroids - centroids.mean(axis=0), axis)
    s_min, s_max = np.percentile(s, 2), np.percentile(s, 98)
    return {"s_min": s_min, "s_max": s_max, "axis": axis}


def method_p23_priority_tests(reader, mesh):
    axis, centroids = compute_vessel_axis(mesh)
    points = reader._points
    faces = reader._faces
    patches = reader.boundary_patches
    centers = []
    for name, info in patches.items():
        sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        for fi in range(sf, sf + nf):
            centers.append(points[faces[fi]].mean(axis=0))
    centers = np.array(centers)
    proj = centers @ axis
    sorted_idx = np.argsort(proj)
    min_seeds = sorted_idx[:5].tolist()
    max_seeds = sorted_idx[-5:].tolist()
    cap1_center = centers[min_seeds].mean(axis=0)
    cap2_center = centers[max_seeds].mean(axis=0)
    s1 = np.dot(cap1_center - centroids.mean(axis=0), axis)
    s2 = np.dot(cap2_center - centroids.mean(axis=0), axis)
    if s1 < s2:
        inlet_center, outlet_center = cap1_center, cap2_center
    else:
        inlet_center, outlet_center = cap2_center, cap1_center
    return {
        "inlet": inlet_center,
        "outlet": outlet_center,
        "n_caps": 2,
        "axis": axis,
    }


def method_p24_conclusion(reader, mesh):
    axis, centroids = compute_vessel_axis(mesh)
    points = reader._points
    faces = reader._faces
    patches = reader.boundary_patches
    centers = []
    for name, info in patches.items():
        sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        for fi in range(sf, sf + nf):
            centers.append(points[faces[fi]].mean(axis=0))
    centers = np.array(centers)
    proj = centers @ axis
    sorted_idx = np.argsort(proj)
    min_seeds = sorted_idx[:5].tolist()
    max_seeds = sorted_idx[-5:].tolist()
    cap1_center = centers[min_seeds].mean(axis=0)
    cap2_center = centers[max_seeds].mean(axis=0)
    s1 = np.dot(cap1_center - centroids.mean(axis=0), axis)
    s2 = np.dot(cap2_center - centroids.mean(axis=0), axis)
    if s1 < s2:
        inlet_center, outlet_center = cap1_center, cap2_center
    else:
        inlet_center, outlet_center = cap2_center, cap1_center
    return {
        "inlet": inlet_center,
        "outlet": outlet_center,
        "n_caps": 2,
        "axis": axis,
    }


def assign_faces_to_caps(reader, cap1_center, cap2_center, axis, centroids):
    points = reader._points
    faces = reader._faces
    patches = reader.boundary_patches
    centers = []
    global_ids = []
    for name, info in patches.items():
        sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        for fi in range(sf, sf + nf):
            centers.append(points[faces[fi]].mean(axis=0))
            global_ids.append(fi)
    centers = np.array(centers)
    global_ids = np.array(global_ids)
    d1 = np.linalg.norm(centers - cap1_center, axis=1)
    d2 = np.linalg.norm(centers - cap2_center, axis=1)
    cap1_idx = np.where(d1 < d2)[0]
    cap2_idx = np.where(d2 < d1)[0]
    s1 = np.dot(cap1_center - centroids.mean(axis=0), axis)
    s2 = np.dot(cap2_center - centroids.mean(axis=0), axis)
    if s1 < s2:
        inlet_faces = set(global_ids[cap1_idx].tolist())
        outlet_faces = set(global_ids[cap2_idx].tolist())
    else:
        inlet_faces = set(global_ids[cap2_idx].tolist())
        outlet_faces = set(global_ids[cap1_idx].tolist())
    return inlet_faces, outlet_faces


def save_single_method_image(mesh, inlet_faces, outlet_faces, title, out_path):
    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    ax.set_axis_off()
    ax.set_title(title, fontsize=14, pad=12)

    all_centers = mesh.cell_centers().points
    rng = np.random.default_rng(42)
    
    n_total = mesh.n_cells
    n_plot = min(3000, n_total)
    idx_bg = rng.choice(n_total, size=n_plot, replace=False)
    idx_bg = np.sort(idx_bg)
    
    ax.scatter(
        all_centers[idx_bg, 0],
        all_centers[idx_bg, 1],
        all_centers[idx_bg, 2],
        c="#cccccc",
        s=0.5,
        alpha=0.3,
        marker=".",
        depthshade=False,
    )

    inlet_idx = np.array(sorted(inlet_faces))
    inlet_idx = inlet_idx[inlet_idx < n_total]
    if len(inlet_idx) > 0:
        ax.scatter(
            all_centers[inlet_idx, 0],
            all_centers[inlet_idx, 1],
            all_centers[inlet_idx, 2],
            c="#0066cc",
            s=8.0,
            alpha=0.9,
            marker="o",
            depthshade=False,
            label=f"Inlet ({len(inlet_idx)} faces)",
        )

    outlet_idx = np.array(sorted(outlet_faces))
    outlet_idx = outlet_idx[outlet_idx < n_total]
    if len(outlet_idx) > 0:
        ax.scatter(
            all_centers[outlet_idx, 0],
            all_centers[outlet_idx, 1],
            all_centers[outlet_idx, 2],
            c="#cc2200",
            s=8.0,
            alpha=0.9,
            marker="o",
            depthshade=False,
            label=f"Outlet ({len(outlet_idx)} faces)",
        )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", fontsize=10, framealpha=0.9)

    pts = all_centers[idx_bg]
    max_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2.0
    mid = pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved {out_path}")


def save_p02_image(mesh, inlet_pt, outlet_pt, title, out_path):
    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    ax.set_axis_off()
    ax.set_title(title, fontsize=14, pad=12)

    all_centers = mesh.cell_centers().points
    n_total = mesh.n_cells
    n_plot = min(3000, n_total)
    rng = np.random.default_rng(42)
    idx_bg = rng.choice(n_total, size=n_plot, replace=False)
    idx_bg = np.sort(idx_bg)

    ax.scatter(
        all_centers[idx_bg, 0],
        all_centers[idx_bg, 1],
        all_centers[idx_bg, 2],
        c="#cccccc",
        s=0.5,
        alpha=0.3,
        marker=".",
        depthshade=False,
    )

    ax.scatter(
        [inlet_pt[0]], [inlet_pt[1]], [inlet_pt[2]],
        c="#0066cc",
        s=150.0,
        alpha=1.0,
        marker="o",
        depthshade=False,
        label=f"Inlet: [{inlet_pt[0]:.3f}, {inlet_pt[1]:.3f}, {inlet_pt[2]:.3f}]",
    )
    ax.scatter(
        [outlet_pt[0]], [outlet_pt[1]], [outlet_pt[2]],
        c="#cc2200",
        s=150.0,
        alpha=1.0,
        marker="o",
        depthshade=False,
        label=f"Outlet: [{outlet_pt[0]:.3f}, {outlet_pt[1]:.3f}, {outlet_pt[2]:.3f}]",
    )

    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    pts = all_centers[idx_bg]
    max_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2.0
    mid = pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved {out_path}")


def save_comparison_grid(mesh, methods_data, out_path):
    fig = plt.figure(figsize=(24, 16), facecolor="white")
    
    all_centers = mesh.cell_centers().points
    n_total = mesh.n_cells
    n_plot = min(2000, n_total)
    rng = np.random.default_rng(42)
    idx_bg = rng.choice(n_total, size=n_plot, replace=False)
    idx_bg = np.sort(idx_bg)
    
    pts_bg = all_centers[idx_bg]
    max_range = (pts_bg.max(axis=0) - pts_bg.min(axis=0)).max() / 2.0
    mid = pts_bg.mean(axis=0)

    for i, (name, inlet_faces, outlet_faces) in enumerate(methods_data, 1):
        ax = fig.add_subplot(2, 4, i, projection="3d", facecolor="white")
        ax.set_axis_off()
        ax.set_title(name, fontsize=12, pad=8)

        ax.scatter(
            pts_bg[:, 0], pts_bg[:, 1], pts_bg[:, 2],
            c="#cccccc", s=0.3, alpha=0.2, marker=".", depthshade=False,
        )

        inlet_idx = np.array(sorted(inlet_faces))
        inlet_idx = inlet_idx[inlet_idx < n_total]
        if len(inlet_idx) > 0:
            ax.scatter(
                all_centers[inlet_idx, 0],
                all_centers[inlet_idx, 1],
                all_centers[inlet_idx, 2],
                c="#0066cc", s=4.0, alpha=0.9, marker="o", depthshade=False,
            )

        outlet_idx = np.array(sorted(outlet_faces))
        outlet_idx = outlet_idx[outlet_idx < n_total]
        if len(outlet_idx) > 0:
            ax.scatter(
                all_centers[outlet_idx, 0],
                all_centers[outlet_idx, 1],
                all_centers[outlet_idx, 2],
                c="#cc2200", s=4.0, alpha=0.9, marker="o", depthshade=False,
            )

        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.view_init(elev=20, azim=-60)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    print("Loading mesh...")
    reader = load_reader()
    mesh = reader.to_pyvista(fields=["U", "p"], time_step=TIME_STEP, as_point_data=False)
    axis, centroids = compute_vessel_axis(mesh)
    points = reader._points
    faces = reader._faces

    # P11
    p11 = method_p11_openfoam_patches(reader, mesh)
    p11_inlet_faces = set(p11["inlet_faces"])
    p11_outlet_faces = set(p11["outlet_faces"])
    save_single_method_image(
        mesh,
        p11_inlet_faces,
        p11_outlet_faces,
        "P11 — Patches OpenFOAM\nINLET: 376 faces | OUTLET: 2512 faces",
        OUT_DIR / "P11_inlet_outlet.png",
    )

    # P04
    p04 = method_p04_plane_caps(reader, mesh)
    p04_inlet_faces, p04_outlet_faces = assign_faces_to_caps(reader, p04["inlet"], p04["outlet"], axis, centroids)
    save_single_method_image(
        mesh,
        p04_inlet_faces,
        p04_outlet_faces,
        f"P04 — Caps plans\nInlet: {len(p04_inlet_faces)} faces | Outlet: {len(p04_outlet_faces)} faces",
        OUT_DIR / "P04_inlet_outlet.png",
    )

    # P23
    p23 = method_p23_priority_tests(reader, mesh)
    p23_inlet_faces, p23_outlet_faces = assign_faces_to_caps(reader, p23["inlet"], p23["outlet"], axis, centroids)
    save_single_method_image(
        mesh,
        p23_inlet_faces,
        p23_outlet_faces,
        f"P23 — Tests prioritaires\nInlet: {len(p23_inlet_faces)} faces | Outlet: {len(p23_outlet_faces)} faces",
        OUT_DIR / "P23_inlet_outlet.png",
    )

    # P24
    p24 = method_p24_conclusion(reader, mesh)
    p24_inlet_faces, p24_outlet_faces = assign_faces_to_caps(reader, p24["inlet"], p24["outlet"], axis, centroids)
    save_single_method_image(
        mesh,
        p24_inlet_faces,
        p24_outlet_faces,
        f"P24 — Conclusion\nInlet: {len(p24_inlet_faces)} faces | Outlet: {len(p24_outlet_faces)} faces",
        OUT_DIR / "P24_inlet_outlet.png",
    )

    # P02
    p02 = method_p02_centerline(reader, mesh)
    save_p02_image(
        mesh,
        p02["inlet"],
        p02["outlet"],
        "P02 — Centerline PCA\nExtrémités inlet/outlet",
        OUT_DIR / "P02_inlet_outlet.png",
    )

    # P13
    p13 = method_p13_topological_graph(reader, mesh)
    centers = []
    global_ids = []
    patches = reader.boundary_patches
    for name, info in patches.items():
        sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        for fi in range(sf, sf + nf):
            centers.append(points[faces[fi]].mean(axis=0))
            global_ids.append(fi)
    centers = np.array(centers)
    global_ids = np.array(global_ids)
    d0 = np.linalg.norm(centers - p13["opening0_center"], axis=1)
    d1 = np.linalg.norm(centers - p13["opening1_center"], axis=1)
    opening0_faces = set(global_ids[np.where(d0 < d1)[0]].tolist())
    opening1_faces = set(global_ids[np.where(d1 < d0)[0]].tolist())
    save_single_method_image(
        mesh,
        opening0_faces,
        opening1_faces,
        f"P13 — Graphe topologique\nopening0: {len(opening0_faces)} faces | opening1: {len(opening1_faces)} faces",
        OUT_DIR / "P13_inlet_outlet.png",
    )

    # P18
    p18 = method_p18_convention(reader, mesh)
    p18_inlet_faces, p18_outlet_faces = p04_inlet_faces, p04_outlet_faces
    save_single_method_image(
        mesh,
        p18_inlet_faces,
        p18_outlet_faces,
        f"P18 — Convention s_min/s_max\nInlet: {len(p18_inlet_faces)} faces | Outlet: {len(p18_outlet_faces)} faces",
        OUT_DIR / "P18_inlet_outlet.png",
    )

    # Comparison grid
    methods_grid = [
        ("P02", set(), set(), p02["inlet"], p02["outlet"]),
        ("P04", p04_inlet_faces, p04_outlet_faces, None, None),
        ("P11", p11_inlet_faces, p11_outlet_faces, None, None),
        ("P13", opening0_faces, opening1_faces, None, None),
        ("P18", p18_inlet_faces, p18_outlet_faces, None, None),
        ("P23", p23_inlet_faces, p23_outlet_faces, None, None),
        ("P24", p24_inlet_faces, p24_outlet_faces, None, None),
    ]
    
    print("\nAll images saved to", OUT_DIR)


if __name__ == "__main__":
    main()
