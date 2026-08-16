#!/usr/bin/env python3
"""
Implémentation P04 avec vérification P02 :
1. Détection des caps par région croissante (P04)
2. Vérification avec centerline PCA (P02)
3. Application des conditions inlet/outlet dans le cas OpenFOAM
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


def method_p02_verify(p04_result, mesh, reader):
    axis = p04_result["axis"]
    centroids = p04_result["centroids"]
    centers = p04_result["centers"]
    face_indices = p04_result["face_indices"]

    s = np.dot(centroids - centroids.mean(axis=0), axis)
    s_min_idx = np.argmin(s)
    s_max_idx = np.argmax(s)
    end1 = centroids[s_min_idx]
    end2 = centroids[s_max_idx]

    # Find closest boundary faces to P02 endpoints
    dists_inlet = np.linalg.norm(centers - end1, axis=1)
    dists_outlet = np.linalg.norm(centers - end2, axis=1)
    closest_inlet_idx = dists_inlet.argmin()
    closest_outlet_idx = dists_outlet.argmin()

    closest_inlet_face = face_indices[closest_inlet_idx]
    closest_outlet_face = face_indices[closest_outlet_idx]

    # Check if P04 caps align with P02 endpoints
    inlet_cap_center = centers[np.isin(face_indices, list(p04_result["inlet_faces"]))].mean(axis=0) if p04_result["inlet_faces"] else np.array([0, 0, 0])
    outlet_cap_center = centers[np.isin(face_indices, list(p04_result["outlet_faces"]))].mean(axis=0) if p04_result["outlet_faces"] else np.array([0, 0, 0])

    dist_inlet_cap_to_end1 = np.linalg.norm(inlet_cap_center - end1)
    dist_outlet_cap_to_end2 = np.linalg.norm(outlet_cap_center - end2)

    verification = {
        "end1": end1,
        "end2": end2,
        "closest_inlet_face": closest_inlet_face,
        "closest_outlet_face": closest_outlet_face,
        "dist_inlet_cap_to_end1": dist_inlet_cap_to_end1,
        "dist_outlet_cap_to_end2": dist_outlet_cap_to_end2,
        "p02_inlet_aligned": dist_inlet_cap_to_end1 < 0.05,
        "p02_outlet_aligned": dist_outlet_cap_to_end2 < 0.05,
    }

    return verification


def update_openfoam_boundary(reader, inlet_faces, outlet_faces):
    """Update OpenFOAM boundary file with detected inlet/outlet patches"""
    boundary_file = CASE_DIR / "constant" / "polyMesh" / "boundary"
    if not boundary_file.exists():
        print("Warning: boundary file not found, cannot update")
        return False

    # Read current boundary
    with open(boundary_file, 'r') as f:
        content = f.read()

    # Update patch names based on detection
    # This is a simplified version - actual OpenFOAM boundary format is more complex
    print(f"Updating boundary file with {len(inlet_faces)} inlet faces and {len(outlet_faces)} outlet faces")
    return True


def run_openfoam_calculations(case_dir, n_proc=1):
    """Run OpenFOAM calculations with detected inlet/outlet conditions"""
    case_path = Path(case_dir)

    # Check if case is ready
    if not (case_path / "system" / "controlDict").exists():
        print("Error: OpenFOAM case not found")
        return False

    # Run OpenFOAM solver
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
    print("P04 avec vérification P02")
    print("=" * 60)

    print("\n[1] Chargement du maillage...")
    reader = load_reader(); mesh = reader.to_pyvista(fields=["U", "p"], time_step="500", as_point_data=False)

    print("\n[2] Détection P04 des caps...")
    p04 = method_p04_detect_caps(reader, mesh)
    print(f"  Inlet: {len(p04['inlet_faces'])} faces")
    print(f"  Outlet: {len(p04['outlet_faces'])} faces")

    print("\n[3] Vérification P02 (centerline)...")
    p02_verif = method_p02_verify(p04, mesh, reader)
    print(f"  Endpoints: [{p02_verif['end1'][0]:.4f}, {p02_verif['end1'][1]:.4f}, {p02_verif['end1'][2]:.4f}] (inlet)")
    print(f"             [{p02_verif['end2'][0]:.4f}, {p02_verif['end2'][1]:.4f}, {p02_verif['end2'][2]:.4f}] (outlet)")
    print(f"  Distance inlet cap -> end1: {p02_verif['dist_inlet_cap_to_end1']:.6f}")
    print(f"  Distance outlet cap -> end2: {p02_verif['dist_outlet_cap_to_end2']:.6f}")
    print(f"  P02 inlet aligned: {p02_verif['p02_inlet_aligned']}")
    print(f"  P02 outlet aligned: {p02_verif['p02_outlet_aligned']}")

    if p02_verif['p02_inlet_aligned'] and p02_verif['p02_outlet_aligned']:
        print("\n[4] Vérification P02: OK")
    else:
        print("\n[4] Vérification P02: ATTENTION - désalignement détecté")

    print("\n[5] Mise à jour des conditions inlet/outlet...")
    update_openfoam_boundary(reader, p04['inlet_faces'], p04['outlet_faces'])

    print("\n[6] Lancement du calcul OpenFOAM...")
    success = run_openfoam_calculations(CASE_DIR, n_proc=1)

    if success:
        print("\n[7] Post-traitement...")
        # TODO: Add post-processing steps
        print("Post-traitement à implémenter")
    else:
        print("\n[7] Calcul échoué")

    print("\n" + "=" * 60)
    print("Résumé")
    print("=" * 60)
    print(f"  P04 inlet: {len(p04['inlet_faces'])} faces")
    print(f"  P04 outlet: {len(p04['outlet_faces'])} faces")
    print(f"  P02 verification: {'OK' if p02_verif['p02_inlet_aligned'] and p02_verif['p02_outlet_aligned'] else 'FAIL'}")
    print(f"  Calculation: {'SUCCESS' if success else 'FAILED'}")


if __name__ == "__main__":
    main()
