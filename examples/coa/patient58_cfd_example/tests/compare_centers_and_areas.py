#!/usr/bin/env python3
"""
Compare P02 endpoints to other methods' face centers.
Compute inlet/outlet areas for each method.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.decomposition import PCA
from foampilot.postprocess import OpenFOAMDirectReader

CASE_DIR = Path(__file__).resolve().parent.parent
TIME_STEP = "500"


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


def face_area(reader, face_id):
    points = reader._points
    faces = reader._faces
    face = faces[face_id]
    if len(face) < 3:
        return 0.0
    pts = points[face]
    area = 0.0
    for i in range(1, len(pts) - 1):
        a = pts[0]
        b = pts[i]
        c = pts[i + 1]
        ab = b - a
        ac = c - a
        area += 0.5 * np.linalg.norm(np.cross(ab, ac))
    return area


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


def main():
    print("Loading mesh...")
    reader = load_reader()
    mesh = reader.to_pyvista(fields=["U", "p"], time_step=TIME_STEP, as_point_data=False)
    axis, centroids = compute_vessel_axis(mesh)

    points = reader._points
    faces = reader._faces

    # P02
    p02 = method_p02_centerline(reader, mesh)
    p02_inlet = p02["inlet"]
    p02_outlet = p02["outlet"]

    # P04
    p04 = method_p04_plane_caps(reader, mesh)
    inlet_faces_p04, outlet_faces_p04 = assign_faces_to_caps(reader, p04["inlet"], p04["outlet"], axis, centroids)

    # P11
    p11 = method_p11_openfoam_patches(reader, mesh)
    inlet_faces_p11 = set(p11["inlet_faces"])
    outlet_faces_p11 = set(p11["outlet_faces"])

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

    # Compute areas and centers
    methods = {
        "P04_inlet": {"faces": inlet_faces_p04, "center": p04["inlet"]},
        "P04_outlet": {"faces": outlet_faces_p04, "center": p04["outlet"]},
        "P11_inlet": {"faces": inlet_faces_p11, "center": None},
        "P11_outlet": {"faces": outlet_faces_p11, "center": None},
        "P13_opening0": {"faces": opening0_faces, "center": p13["opening0_center"]},
        "P13_opening1": {"faces": opening1_faces, "center": p13["opening1_center"]},
    }

    # Compute P11 centers from actual faces
    p11_inlet_center = np.mean([points[fi].mean(axis=0) for fi in inlet_faces_p11], axis=0) if inlet_faces_p11 else np.array([0, 0, 0])
    p11_outlet_center = np.mean([points[fi].mean(axis=0) for fi in outlet_faces_p11], axis=0) if outlet_faces_p11 else np.array([0, 0, 0])
    methods["P11_inlet"]["center"] = p11_inlet_center
    methods["P11_outlet"]["center"] = p11_outlet_center

    print("=" * 80)
    print("COMPARAISON DES CENTRES ET AIRES")
    print("=" * 80)

    for name, data in methods.items():
        face_list = list(data["faces"])
        center = data["center"]
        if face_list:
            areas = [face_area(reader, fi) for fi in face_list]
            total_area = sum(areas)
            mean_area = np.mean(areas)
            face_centers = np.array([points[fi].mean(axis=0) for fi in face_list])
            mean_center = face_centers.mean(axis=0)
            dist_to_mean = np.linalg.norm(center - mean_center) if center is not None else 0.0
            print(f"\n{name}:")
            print(f"  Nombre de faces: {len(face_list)}")
            print(f"  Aire totale: {total_area:.6f}")
            print(f"  Aire moyenne par face: {mean_area:.6f}")
            print(f"  Centre moyen des faces: [{mean_center[0]:.4f}, {mean_center[1]:.4f}, {mean_center[2]:.4f}]")
            if center is not None:
                print(f"  Centre méthode: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
                print(f"  Distance centre méthode -> mean: {dist_to_mean:.6f}")
        else:
            print(f"\n{name}:")
            print(f"  (pas de faces)")

    # Compare P02 to other methods
    print("\n" + "=" * 80)
    print("DISTANCES P02 -> CENTRES DES AUTRES MÉTHODES")
    print("=" * 80)

    for name, data in methods.items():
        if not data["faces"]:
            continue
        face_list = list(data["faces"])
        face_centers = np.array([points[fi].mean(axis=0) for fi in face_list])
        center = data["center"]
        
        p02_pt = p02_inlet if "inlet" in name or "opening0" in name else p02_outlet
        dists = np.linalg.norm(face_centers - p02_pt, axis=1)
        min_dist = dists.min()
        mean_dist = dists.mean()
        closest_idx = dists.argmin()
        print(f"\nP02 {'inlet' if 'inlet' in name or 'opening0' in name else 'outlet'} -> {name}:")
        print(f"  Distance min: {min_dist:.6f}")
        print(f"  Distance moyenne: {mean_dist:.6f}")
        print(f"  Face la plus proche: {face_list[closest_idx]}")
        print(f"  Centre face la plus proche: [{face_centers[closest_idx][0]:.4f}, {face_centers[closest_idx][1]:.4f}, {face_centers[closest_idx][2]:.4f}]")

    # Summary table
    print("\n" + "=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)
    print(f"{'Méthode':<20} {'# Inlet':>10} {'# Outlet':>10} {'Aire inlet':>12} {'Aire outlet':>12}")
    print("-" * 70)
    
    for prefix in ["P04", "P11", "P13"]:
        inlet_key = f"{prefix}_inlet"
        outlet_key = f"{prefix}_outlet"
        if inlet_key in methods and outlet_key in methods:
            inlet_faces = list(methods[inlet_key]["faces"])
            outlet_faces = list(methods[outlet_key]["faces"])
            inlet_area = sum(face_area(reader, fi) for fi in inlet_faces)
            outlet_area = sum(face_area(reader, fi) for fi in outlet_faces)
            print(f"{prefix:<20} {len(inlet_faces):>10} {len(outlet_faces):>10} {inlet_area:>12.6f} {outlet_area:>12.6f}")


if __name__ == "__main__":
    main()
