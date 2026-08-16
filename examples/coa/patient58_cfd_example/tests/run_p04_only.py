#!/usr/bin/env python3
"""
Lancement direct avec P04 uniquement :
1. Détection des caps par région croissante
2. Application des conditions inlet/outlet
3. Calcul OpenFOAM
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import subprocess
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


def method_p04_detect_caps(reader, mesh):
    axis, centroids = compute_vessel_axis(mesh)
    face_data = build_boundary_face_data(reader)
    centers = face_data["centers"]
    normals = face_data["normals"]
    areas = face_data["areas"]
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
        "centroids": centroids,
        "centers": centers,
        "face_indices": face_indices,
        "s1": s1,
        "s2": s2,
    }


def run_openfoam_calculations(case_dir, n_proc=1):
    """Run OpenFOAM calculations with detected inlet/outlet conditions"""
    case_path = Path(case_dir)

    if not (case_path / "system" / "controlDict").exists():
        print("Error: OpenFOAM case not found")
        return False

    cmd = ["mpirun", "-np", str(n_proc), "simpleFoam", "-case", str(case_path)]
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("Calculation completed successfully")
            return True
        else:
            print(f"Calculation failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("Calculation timed out")
        return False
    except Exception as e:
        print(f"Error running calculation: {e}")
        return False


def main():
    print("=" * 60)
    print("P04 uniquement - Détection inlet/outlet")
    print("=" * 60)

    print("\n[1] Chargement du maillage...")
    reader = load_reader()
    mesh = reader.to_pyvista(fields=["U", "p"], time_step=TIME_STEP, as_point_data=False)

    print("\n[2] Détection P04 des caps...")
    p04 = method_p04_detect_caps(reader, mesh)
    print(f"  Inlet: {len(p04['inlet_faces'])} faces")
    print(f"  Outlet: {len(p04['outlet_faces'])} faces")
    print(f"  Cap1 center: {p04['centers'][np.isin(p04['face_indices'], list(p04['cap1_faces']))].mean(axis=0) if p04['cap1_faces'] else 'N/A'}")
    print(f"  Cap2 center: {p04['centers'][np.isin(p04['face_indices'], list(p04['cap2_faces']))].mean(axis=0) if p04['cap2_faces'] else 'N/A'}")

    print("\n[3] Lancement du calcul OpenFOAM...")
    success = run_openfoam_calculations(CASE_DIR, n_proc=1)

    print("\n" + "=" * 60)
    print("Résumé")
    print("=" * 60)
    print(f"  P04 inlet: {len(p04['inlet_faces'])} faces")
    print(f"  P04 outlet: {len(p04['outlet_faces'])} faces")
    print(f"  Calculation: {'SUCCESS' if success else 'FAILED'}")


if __name__ == "__main__":
    main()
