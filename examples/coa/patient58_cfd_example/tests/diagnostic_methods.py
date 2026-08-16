#!/usr/bin/env python3
"""
Diagnostic des méthodes fiables : vérifier la cohérence des sélections
avant toute visualisation.
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


def face_area(points, faces, face_id):
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


def main():
    print("Loading mesh...")
    reader = load_reader()
    mesh = reader.to_pyvista(fields=["U", "p"], time_step=TIME_STEP, as_point_data=False)
    axis, centroids = compute_vessel_axis(mesh)
    points = reader._points
    faces = reader._faces
    patches = reader.boundary_patches

    print(f"Total faces in mesh: {len(faces)}")
    print(f"Total boundary faces: {sum(info['nFaces'] for info in patches.values())}")
    print(f"Patches: {list(patches.keys())}")
    for name, info in patches.items():
        print(f"  {name}: {info['nFaces']} faces")

    # P11 - Patches OpenFOAM (référence)
    print("\n" + "=" * 60)
    print("P11 - Patches OpenFOAM")
    print("=" * 60)
    p11_inlet = set(get_patch_face_ids(reader, "INLET"))
    p11_outlet = set(get_patch_face_ids(reader, "OUTLET"))
    p11_wall = set(get_patch_face_ids(reader, "WALL"))
    print(f"INLET: {len(p11_inlet)} faces")
    print(f"OUTLET: {len(p11_outlet)} faces")
    print(f"WALL: {len(p11_wall)} faces")
    print(f"Total boundary: {len(p11_inlet) + len(p11_outlet) + len(p11_wall)}")

    # P04 - Caps plans (seulement boundary faces)
    print("\n" + "=" * 60)
    print("P04 - Caps plans (boundary faces only)")
    print("=" * 60)
    boundary_centers = []
    boundary_ids = []
    for name, info in patches.items():
        sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        for fi in range(sf, sf + nf):
            boundary_centers.append(points[faces[fi]].mean(axis=0))
            boundary_ids.append(fi)
    boundary_centers = np.array(boundary_centers)
    boundary_ids = np.array(boundary_ids)
    
    proj = boundary_centers @ axis
    sorted_idx = np.argsort(proj)
    
    # Take extreme 10% as caps
    n_cap = max(1, int(len(boundary_ids) * 0.05))
    min_seeds = sorted_idx[:n_cap].tolist()
    max_seeds = sorted_idx[-n_cap:].tolist()
    
    cap1_center = boundary_centers[min_seeds].mean(axis=0)
    cap2_center = boundary_centers[max_seeds].mean(axis=0)
    
    s1 = np.dot(cap1_center - boundary_centers.mean(axis=0), axis)
    s2 = np.dot(cap2_center - boundary_centers.mean(axis=0), axis)
    
    if s1 < s2:
        inlet_center, outlet_center = cap1_center, cap2_center
        inlet_ids = set(boundary_ids[min_seeds].tolist())
        outlet_ids = set(boundary_ids[max_seeds].tolist())
    else:
        inlet_center, outlet_center = cap2_center, cap1_center
        inlet_ids = set(boundary_ids[max_seeds].tolist())
        outlet_ids = set(boundary_ids[min_seeds].tolist())
    
    print(f"Cap 1 (inlet?): {len(inlet_ids)} faces, center=[{inlet_center[0]:.4f}, {inlet_center[1]:.4f}, {inlet_center[2]:.4f}]")
    print(f"Cap 2 (outlet?): {len(outlet_ids)} faces, center=[{outlet_center[0]:.4f}, {outlet_center[1]:.4f}, {outlet_center[2]:.4f}]")
    
    # Check overlap with P11
    inlet_inter = len(inlet_ids & p11_inlet)
    outlet_inter = len(outlet_ids & p11_outlet)
    print(f"Overlap with P11 INLET: {inlet_inter} / {len(p11_inlet)}")
    print(f"Overlap with P11 OUTLET: {outlet_inter} / {len(p11_outlet)}")

    # P13 - Topological graph
    print("\n" + "=" * 60)
    print("P13 - Graphe topologique")
    print("=" * 60)
    
    # Build adjacency only on boundary faces
    pt_to_faces = {}
    for fi in boundary_ids:
        for pt in faces[fi]:
            pt_to_faces.setdefault(pt, []).append(fi)
    
    adjacency = {}
    for fi in boundary_ids:
        face_pts = set(faces[fi])
        neighbors = set()
        for pt in face_pts:
            for nb in pt_to_faces.get(pt, []):
                if nb != fi:
                    neighbors.add(nb)
        adjacency[fi] = neighbors
    
    degrees = {fi: len(nb) for fi, nb in adjacency.items()}
    terminals = [fi for fi, d in degrees.items() if d == 1]
    print(f"Terminal faces (degree=1): {len(terminals)}")
    
    if len(terminals) >= 2:
        terminal_centers = np.array([points[faces[fi]].mean(axis=0) for fi in terminals])
        proj_t = terminal_centers @ axis
        sorted_t = np.argsort(proj_t)
        
        # Take extreme terminals
        n_term = max(1, int(len(terminals) * 0.1))
        opening0_ids = set([terminals[i] for i in sorted_t[:n_term]])
        opening1_ids = set([terminals[i] for i in sorted_t[-n_term:]])
        
        opening0_center = points[faces[list(opening0_ids)[0]]].mean(axis=0)
        opening1_center = points[faces[list(opening1_ids)[0]]].mean(axis=0)
        
        print(f"opening0: {len(opening0_ids)} faces, center=[{opening0_center[0]:.4f}, {opening0_center[1]:.4f}, {opening0_center[2]:.4f}]")
        print(f"opening1: {len(opening1_ids)} faces, center=[{opening1_center[0]:.4f}, {opening1_center[1]:.4f}, {opening1_center[2]:.4f}]")
        
        # Check overlap with P11
        op0_inter_inlet = len(opening0_ids & p11_inlet)
        op0_inter_outlet = len(opening0_ids & p11_outlet)
        op1_inter_inlet = len(opening1_ids & p11_inlet)
        op1_inter_outlet = len(opening1_ids & p11_outlet)
        print(f"opening0 overlap INLET: {op0_inter_inlet}, OUTLET: {op0_inter_outlet}")
        print(f"opening1 overlap INLET: {op1_inter_inlet}, OUTLET: {op1_inter_outlet}")
    else:
        print("Not enough terminals found")
        opening0_ids = set()
        opening1_ids = set()

    # P02 - Centerline endpoints
    print("\n" + "=" * 60)
    print("P02 - Centerline PCA")
    print("=" * 60)
    s = np.dot(centroids - centroids.mean(axis=0), axis)
    s_min_idx = np.argmin(s)
    s_max_idx = np.argmax(s)
    p02_inlet = centroids[s_min_idx]
    p02_outlet = centroids[s_max_idx]
    print(f"Inlet: [{p02_inlet[0]:.4f}, {p02_inlet[1]:.4f}, {p02_inlet[2]:.4f}]")
    print(f"Outlet: [{p02_outlet[0]:.4f}, {p02_outlet[1]:.4f}, {p02_outlet[2]:.4f}]")
    
    # Find closest boundary faces to P02 endpoints
    dists_inlet = np.linalg.norm(boundary_centers - p02_inlet, axis=1)
    dists_outlet = np.linalg.norm(boundary_centers - p02_outlet, axis=1)
    closest_inlet_idx = dists_inlet.argmin()
    closest_outlet_idx = dists_outlet.argmin()
    print(f"Closest boundary face to inlet: {boundary_ids[closest_inlet_idx]}, dist={dists_inlet.min():.6f}")
    print(f"Closest boundary face to outlet: {boundary_ids[closest_outlet_idx]}, dist={dists_outlet.min():.6f}")

    # P18 - Convention s_min/s_max
    print("\n" + "=" * 60)
    print("P18 - Convention s_min/s_max")
    print("=" * 60)
    s_min, s_max = np.percentile(s, 2), np.percentile(s, 98)
    print(f"s_min: {s_min:.6f}, s_max: {s_max:.6f}")
    print(f"P02 s_min: {s.min():.6f}, s_max: {s.max():.6f}")
    
    # P23/P24 - Priority tests / Conclusion
    print("\n" + "=" * 60)
    print("P23/P24 - Tests prioritaires / Conclusion")
    print("=" * 60)
    print("Same as P04 + P13 + P18")
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    print(f"{'Méthode':<20} {'# Inlet':>10} {'# Outlet':>10}")
    print("-" * 50)
    print(f"{'P11 (patches)':<20} {len(p11_inlet):>10} {len(p11_outlet):>10}")
    print(f"{'P04 (caps 5%)':<20} {len(inlet_ids):>10} {len(outlet_ids):>10}")
    if len(terminals) >= 2:
        print(f"{'P13 (terminaux)':<20} {len(opening0_ids):>10} {len(opening1_ids):>10}")


if __name__ == "__main__":
    main()
