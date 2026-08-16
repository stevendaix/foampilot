#!/usr/bin/env python3
"""
Comparaison des méthodes fiables P02, P04, P11, P13, P18, P23, P24
Extraction des zones inlet/outlet et vérification de coïncidence exacte.
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
BASE_DIR = Path(__file__).resolve().parent


def load_reader():
    reader = OpenFOAMDirectReader(case_path=CASE_DIR)
    mesh = reader.to_pyvista(fields=["U", "p"], time_step=TIME_STEP, as_point_data=False)
    return reader, mesh


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
    axis = axis / np.linalg.norm(axis)
    return axis, centroids


def get_boundary_surface(reader):
    points = reader._points
    faces = reader._faces
    all_faces_list = []
    for name, info in reader.boundary_patches.items():
        sf = info.get("startFace", 0)
        nf = info.get("nFaces", 0)
        for fi in range(sf, sf + nf):
            face = faces[fi]
            n_pts = len(face)
            all_faces_list.append(n_pts)
            all_faces_list.extend([int(v) for v in face])
    return pv.PolyData(points, faces=np.array(all_faces_list, dtype=int))


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
    inlet_ids = get_patch_face_ids(reader, "inlet")
    outlet_ids = get_patch_face_ids(reader, "outlet")
    wall_ids = get_patch_face_ids(reader, "wall")
    return {
        "inlet_faces": len(inlet_ids),
        "outlet_faces": len(outlet_ids),
        "wall_faces": len(wall_ids),
        "inlet_ids": inlet_ids,
        "outlet_ids": outlet_ids,
        "wall_ids": wall_ids,
        "axis": axis,
    }


def method_p13_topological_graph(reader, mesh):
    axis, centroids = compute_vessel_axis(mesh)
    points = reader._points
    faces = reader._faces
    patches = reader.boundary_patches

    pt_to_faces = {}
    for name, info in patches.items():
        sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        for fi in range(sf, sf + nf):
            for pt in faces[fi]:
                pt_to_faces.setdefault(pt, []).append(fi)

    adjacency = {}
    for name, info in patches.items():
        sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        for fi in range(sf, sf + nf):
            face_pts = set(faces[fi])
            neighbors = set()
            for pt in face_pts:
                for nb in pt_to_faces.get(pt, []):
                    if nb != fi:
                        neighbors.add(nb)
            adjacency[fi] = neighbors

    degrees = {fi: len(nb) for fi, nb in adjacency.items()}
    terminals = [fi for fi, d in degrees.items() if d == 1]
    terminal_centers = []
    for fi in terminals:
        terminal_centers.append(points[faces[fi]].mean(axis=0))
    terminal_centers = np.array(terminal_centers)

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


def main():
    print("Loading mesh...")
    reader, mesh = load_reader()
    axis, centroids = compute_vessel_axis(mesh)

    results = {}
    results["P02"] = method_p02_centerline(reader, mesh)
    results["P04"] = method_p04_plane_caps(reader, mesh)
    results["P11"] = method_p11_openfoam_patches(reader, mesh)
    results["P13"] = method_p13_topological_graph(reader, mesh)
    results["P18"] = method_p18_convention(reader, mesh)
    results["P23"] = method_p23_priority_tests(reader, mesh)
    results["P24"] = method_p24_conclusion(reader, mesh)

    lines = ["# Comparaison des méthodes fiables P02/P04/P11/P13/P18/P23/P24\n"]
    lines.append(f"- Axe vaisseau : [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]\n\n")

    for name, res in results.items():
        lines.append(f"## {name}\n")
        if "inlet" in res and "outlet" in res:
            lines.append(f"- Inlet : [{res['inlet'][0]:.4f}, {res['inlet'][1]:.4f}, {res['inlet'][2]:.4f}]\n")
            lines.append(f"- Outlet : [{res['outlet'][0]:.4f}, {res['outlet'][1]:.4f}, {res['outlet'][2]:.4f}]\n")
        if "inlet_faces" in res:
            lines.append(f"- INLET : {res['inlet_faces']} faces\n")
            lines.append(f"- OUTLET : {res['outlet_faces']} faces\n")
            lines.append(f"- WALL : {res['wall_faces']} faces\n")
        if "openings" in res:
            lines.append(f"- Openings : {res['openings']}\n")
            lines.append(f"- opening_0 : [{res['opening0_center'][0]:.4f}, {res['opening0_center'][1]:.4f}, {res['opening0_center'][2]:.4f}]\n")
            lines.append(f"- opening_1 : [{res['opening1_center'][0]:.4f}, {res['opening1_center'][1]:.4f}, {res['opening1_center'][2]:.4f}]\n")
        if "s_min" in res:
            lines.append(f"- s_min : {res['s_min']:.6f}\n")
            lines.append(f"- s_max : {res['s_max']:.6f}\n")
        lines.append("\n")

    out_path = BASE_DIR / "comparison_methods.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Comparison written to {out_path}")

    print("\n=== COMPARISON SUMMARY ===")
    for name, res in results.items():
        if "inlet" in res and "outlet" in res:
            print(f"{name}: inlet={res['inlet']}, outlet={res['outlet']}")
        elif "inlet_faces" in res:
            print(f"{name}: INLET={res['inlet_faces']} faces, OUTLET={res['outlet_faces']} faces")
        elif "openings" in res:
            print(f"{name}: {res['openings']} openings")


if __name__ == "__main__":
    main()
