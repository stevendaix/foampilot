#!/usr/bin/env python3
"""
Visualisation améliorée des méthodes fiables
Meilleur rendu, angles de vue, transparence, contours.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA
from foampilot.postprocess import OpenFOAMDirectReader

CASE_DIR = Path(__file__).resolve().parent.parent
TIME_STEP = "500"
OUT_DIR = Path(__file__).resolve().parent / "method_images_v2"
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
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1000))
    
    # Mesh de base avec edges fines pour la structure
    plotter.add_mesh(
        mesh,
        color="#cccccc",
        show_edges=False,
        opacity=0.2,
        lighting=True,
        smooth_shading=True,
    )
    
    # Zones inlet/outlet avec extraction de surface
    if len(inlet_faces) > 0:
        inlet_cells = list(inlet_faces)
        inlet_mesh = mesh.extract_cells(inlet_cells)
        if inlet_mesh.n_points > 0:
            plotter.add_mesh(
                inlet_mesh,
                color="#0077bb",
                show_edges=True,
                edge_color="#003366",
                line_width=1.5,
                opacity=0.92,
                label=f"Inlet ({len(inlet_faces)} faces)",
            )
    
    if len(outlet_faces) > 0:
        outlet_cells = list(outlet_faces)
        outlet_mesh = mesh.extract_cells(outlet_cells)
        if outlet_mesh.n_points > 0:
            plotter.add_mesh(
                outlet_mesh,
                color="#cc3322",
                show_edges=True,
                edge_color="#661100",
                line_width=1.5,
                opacity=0.92,
                label=f"Outlet ({len(outlet_faces)} faces)",
            )
    
    plotter.enable_anti_aliasing()
    plotter.set_background("white")
    plotter.add_axes()
    plotter.add_title(title, font_size=14)
    plotter.view_isometric()
    plotter.camera.zoom(1.3)
    plotter.screenshot(str(out_path), scale=2)
    plotter.close()
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
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1000))
    plotter.add_mesh(mesh, color="#cccccc", show_edges=False, opacity=0.2, smooth_shading=True)
    
    inlet_sphere = pv.Sphere(radius=0.0025, center=p02["inlet"])
    outlet_sphere = pv.Sphere(radius=0.0025, center=p02["outlet"])
    plotter.add_mesh(inlet_sphere, color="#0077bb", label=f"Inlet: [{p02['inlet'][0]:.3f}, {p02['inlet'][1]:.3f}, {p02['inlet'][2]:.3f}]")
    plotter.add_mesh(outlet_sphere, color="#cc3322", label=f"Outlet: [{p02['outlet'][0]:.3f}, {p02['outlet'][1]:.3f}, {p02['outlet'][2]:.3f}]")
    
    arrow_start = p02["outlet"]
    arrow_dir = p02["inlet"] - p02["outlet"]
    arrow_dir = arrow_dir / np.linalg.norm(arrow_dir) * 0.025
    arrow = pv.Arrow(start=arrow_start, direction=arrow_dir, tip_length=0.5, tip_radius=0.15, shaft_radius=0.04)
    plotter.add_mesh(arrow, color="black")
    
    plotter.enable_anti_aliasing()
    plotter.set_background("white")
    plotter.add_axes()
    plotter.add_title("P02 — Centerline PCA\nExtrémités inlet/outlet", font_size=14)
    plotter.view_isometric()
    plotter.camera.zoom(1.3)
    plotter.screenshot(str(OUT_DIR / "P02_inlet_outlet.png"), scale=2)
    plotter.close()
    print(f"Saved {OUT_DIR / 'P02_inlet_outlet.png'}")

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
