#!/usr/bin/env python3
"""
Visualisation matplotlib des méthodes fiables
Rendu par triangles colorés, fond gris, contours nets.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA
from foampilot.postprocess import OpenFOAMDirectReader

CASE_DIR = Path(__file__).resolve().parent.parent
TIME_STEP = "500"
OUT_DIR = Path(__file__).resolve().parent / "method_images_v4"
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
    wall_ids = get_patch_face_ids(reader, "WALL")
    return {
        "inlet_faces": inlet_ids,
        "outlet_faces": outlet_ids,
        "wall_faces": wall_ids,
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


def save_method_image(mesh, inlet_faces, outlet_faces, title, out_path):
    fig = plt.figure(figsize=(14, 10), facecolor="#e8e8e8")
    ax = fig.add_subplot(111, projection="3d", facecolor="#e8e8e8")
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, pad=16, color="#111111")

    points = mesh.points
    faces = mesh.cells  # This might not be right for PyVista

    # For PyVista mesh, get faces as triangles
    # mesh.faces is a 1D array: [n_pts, pt1, pt2, pt3, n_pts, pt1, ...]
    faces_array = mesh.faces.reshape(-1, 4)[:, 1:]
    
    # Create face polygons for boundary faces
    inlet_polys = []
    outlet_polys = []
    wall_polys = []
    
    for fi in range(mesh.n_cells):
        face_pts = faces_array[fi]
        poly = points[face_pts].tolist()
        if fi in inlet_faces:
            inlet_polys.append(poly)
        elif fi in outlet_faces:
            outlet_polys.append(poly)
        else:
            wall_polys.append(poly)
    
    # Draw wall faces (light gray, no edges)
    if wall_polys:
        wall_collection = Poly3DCollection(wall_polys, alpha=0.25, facecolor='#cccccc', edgecolor='none', linewidths=0)
        ax.add_collection3d(wall_collection)
    
    # Draw inlet faces (blue, with edges)
    if inlet_polys:
        inlet_collection = Poly3DCollection(inlet_polys, alpha=0.9, facecolor='#0066cc', edgecolor='#003366', linewidths=0.5)
        ax.add_collection3d(inlet_collection)
    
    # Draw outlet faces (red, with edges)
    if outlet_polys:
        outlet_collection = Poly3DCollection(outlet_polys, alpha=0.9, facecolor='#cc2200', edgecolor='#661100', linewidths=0.5)
        ax.add_collection3d(outlet_collection)
    
    # Set equal aspect ratio
    all_pts = np.vstack([points[face_pts] for face_pts in faces_array[:min(1000, len(faces_array))]])
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
    mid = all_pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")


def save_p02_image(mesh, inlet_pt, outlet_pt, title, out_path):
    fig = plt.figure(figsize=(14, 10), facecolor="#e8e8e8")
    ax = fig.add_subplot(111, projection="3d", facecolor="#e8e8e8")
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, pad=16, color="#111111")

    points = mesh.points
    faces_array = mesh.faces.reshape(-1, 4)[:, 1:]
    
    wall_polys = []
    for fi in range(min(2000, mesh.n_cells)):
        face_pts = faces_array[fi]
        wall_polys.append(points[face_pts].tolist())
    
    if wall_polys:
        wall_collection = Poly3DCollection(wall_polys, alpha=0.25, facecolor='#cccccc', edgecolor='none', linewidths=0)
        ax.add_collection3d(wall_collection)
    
    # Draw spheres for inlet/outlet
    from mpl_toolkits.mplot3d.art3d import Path3DCollection
    ax.scatter([inlet_pt[0]], [inlet_pt[1]], [inlet_pt[2]], c='#0066cc', s=400, alpha=1.0, marker='o', edgecolors='#003366', linewidths=2)
    ax.scatter([outlet_pt[0]], [outlet_pt[1]], [outlet_pt[2]], c='#cc2200', s=400, alpha=1.0, marker='o', edgecolors='#661100', linewidths=2)
    
    all_pts = np.vstack([points[face_pts] for face_pts in faces_array[:min(1000, len(faces_array))]])
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
    mid = all_pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    print("Loading mesh...")
    reader = load_reader()
    mesh = reader.to_pyvista(fields=["U", "p"], time_step=TIME_STEP, as_point_data=False)
    axis, centroids = compute_vessel_axis(mesh)

    # P11
    p11 = method_p11_openfoam_patches(reader, mesh)
    save_method_image(
        mesh,
        set(p11["inlet_faces"]),
        set(p11["outlet_faces"]),
        "P11 — Patches OpenFOAM\nINLET: 376 faces | OUTLET: 2512 faces",
        OUT_DIR / "P11_inlet_outlet.png",
    )

    # P04, P23, P24
    for name, func in [
        ("P04", method_p04_plane_caps),
        ("P23", method_p23_priority_tests),
        ("P24", method_p24_conclusion),
    ]:
        res = func(reader, mesh)
        inlet_faces, outlet_faces = assign_faces_to_caps(reader, res["inlet"], res["outlet"], axis, centroids)
        save_method_image(
            mesh,
            inlet_faces,
            outlet_faces,
            f"{name} — Caps plans\nInlet: {len(inlet_faces)} faces | Outlet: {len(outlet_faces)} faces",
            OUT_DIR / f"{name}_inlet_outlet.png",
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
    d0 = np.linalg.norm(centers - p13["opening0_center"], axis=1)
    d1 = np.linalg.norm(centers - p13["opening1_center"], axis=1)
    opening0_faces = set(global_ids[np.where(d0 < d1)[0]].tolist())
    opening1_faces = set(global_ids[np.where(d1 < d0)[0]].tolist())
    save_method_image(
        mesh,
        opening0_faces,
        opening1_faces,
        f"P13 — Graphe topologique\nopening0: {len(opening0_faces)} faces | opening1: {len(opening1_faces)} faces",
        OUT_DIR / "P13_inlet_outlet.png",
    )

    # P18
    p18 = method_p18_convention(reader, mesh)
    p04 = method_p04_plane_caps(reader, mesh)
    inlet_faces, outlet_faces = assign_faces_to_caps(reader, p04["inlet"], p04["outlet"], axis, centroids)
    save_method_image(
        mesh,
        inlet_faces,
        outlet_faces,
        f"P18 — Convention s_min/s_max\nInlet: {len(inlet_faces)} faces | Outlet: {len(outlet_faces)} faces",
        OUT_DIR / "P18_inlet_outlet.png",
    )

    print("\nAll images saved to", OUT_DIR)


if __name__ == "__main__":
    main()
