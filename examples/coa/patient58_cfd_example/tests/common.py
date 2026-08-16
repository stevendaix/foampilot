import os
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA
from pathlib import Path

from foampilot.postprocess import OpenFOAMDirectReader

pv.OFF_SCREEN = True

CASE_DIR = Path(__file__).resolve().parent.parent
TIME_STEP = "500"
BASE_DIR = Path(__file__).resolve().parent


def load_reader():
    reader = OpenFOAMDirectReader(case_path=CASE_DIR)
    mesh = reader.to_pyvista(fields=["U", "p"], time_step=TIME_STEP, as_point_data=False)
    return reader, mesh


def compute_face_normal(face_vertex_indices, points):
    pts = points[face_vertex_indices]
    if len(pts) < 3:
        return np.array([0.0, 0.0, 0.0])
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0])
    return normal / norm


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
    return axis, centroids, U_mag[active]


def get_boundary_surface(reader):
    points = reader._points
    faces = reader._faces
    all_faces_list = []
    for name, info in reader.boundary_patches.items():
        start_face = info.get("startFace", 0)
        n_faces = info.get("nFaces", 0)
        for fi in range(start_face, start_face + n_faces):
            face = faces[fi]
            n_pts = len(face)
            all_faces_list.append(n_pts)
            all_faces_list.extend([int(v) for v in face])
    all_faces_arr = np.array(all_faces_list, dtype=int)
    return pv.PolyData(points, faces=all_faces_arr)


def write_results(test_idx, filename, content):
    out_path = BASE_DIR / f"P{test_idx:02d}" / filename
    out_path.write_text(content, encoding="utf-8")
    print(f"  -> {out_path}")


def save_matplotlib_image(test_idx, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.5, f"Test P{test_idx:02d}\nResults saved in results_P{test_idx:02d}.md",
            ha="center", va="center", transform=ax.transAxes, fontsize=14,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))
    
    img_path = BASE_DIR / f"P{test_idx:02d}" / name
    fig.savefig(str(img_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {img_path}")
