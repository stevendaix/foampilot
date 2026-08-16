#!/usr/bin/env python3
"""
Comparaison STL vs build123d avec géométries corrigées.

Superpose le maillage STL original et les géométries CAD reconstruites
pour évaluer la qualité de la reconstruction aortique.

Métriques :
- Distance Hausdorff
- Erreur moyenne
- Volume et aire comparés
- Rapport de volume

Usage:
    python3 compare_stl_vs_build123d.py
"""

import os
import sys
from pathlib import Path

os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkEGLRenderWindow"

import json

import numpy as np
import trimesh
import pyvista as pv

import build123d as b123d

# PATHS
BASE_DIR = Path(__file__).resolve().parent
STL_PATH = BASE_DIR.parent / "patient58_cfd_example" / "constant" / "triSurface" / "tbad_TL_walls.stl"
CENTERLINE_PATH = BASE_DIR.parent / "patient58_cfd_example" / "centerline.npy"
CAD_FILES = {
    "sweep_direct": BASE_DIR / "aorta_sweep_direct.stl",
    "loft_oriented": BASE_DIR / "aorta_loft_oriented.stl",
    "sweep_vanilla": BASE_DIR / "aorta_sweep.step",
}
OUTPUT_DIR = BASE_DIR


def load_stl(path: Path) -> trimesh.Trimesh:
    """Load STL mesh."""
    mesh = trimesh.load(str(path), process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    return mesh


def load_step_as_mesh(step_path: Path) -> trimesh.Trimesh:
    """Load STEP or STL file and convert to mesh for comparison."""
    import tempfile

    if step_path.suffix.lower() == ".stl":
        mesh = trimesh.load(str(step_path), process=True)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        return mesh

    shape = b123d.import_step(str(step_path))

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        b123d.export_stl(shape, str(tmp_path))
        mesh = trimesh.load(str(tmp_path), process=True)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        return mesh
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def compute_hausdorff_distance(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh) -> tuple[float, float]:
    """Compute Hausdorff distance between two meshes."""
    from scipy.spatial import cKDTree

    pts1 = mesh1.sample(5000)
    pts2 = mesh2.sample(5000)

    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)

    dist1, _ = tree2.query(pts1, k=1)
    dist2, _ = tree1.query(pts2, k=1)

    hausdorff = float(max(dist1.max(), dist2.max()))
    mean_dist = float((dist1.mean() + dist2.mean()) / 2)
    return hausdorff, mean_dist


def compare_meshes(stl_mesh: trimesh.Trimesh, cad_mesh: trimesh.Trimesh) -> dict:
    """Compare STL and CAD meshes."""
    # Align CAD to STL for fair comparison
    stl_center = stl_mesh.centroid
    cad_center = cad_mesh.centroid
    translation = stl_center - cad_center

    cad_aligned = cad_mesh.copy()
    cad_aligned.vertices += translation

    result = {}
    result["stl_vertices"] = len(stl_mesh.vertices)
    result["stl_faces"] = len(stl_mesh.faces)
    result["stl_volume"] = float(stl_mesh.volume) if stl_mesh.is_watertight else None
    result["stl_area"] = float(stl_mesh.area)

    result["cad_vertices"] = len(cad_aligned.vertices)
    result["cad_faces"] = len(cad_aligned.faces)
    result["cad_volume"] = float(cad_aligned.volume) if cad_aligned.is_watertight else None
    result["cad_area"] = float(cad_aligned.area)

    try:
        hausdorff, mean_dist = compute_hausdorff_distance(stl_mesh, cad_aligned)
        result["hausdorff_distance"] = hausdorff
        result["mean_distance"] = mean_dist
    except Exception as exc:
        print(f"  Distance computation failed: {exc}")
        result["hausdorff_distance"] = None
        result["mean_distance"] = None

    if result["stl_volume"] and result["cad_volume"] and result["cad_volume"] > 0:
        result["volume_ratio"] = result["stl_volume"] / result["cad_volume"]
    else:
        result["volume_ratio"] = None

    if result["stl_area"] > 0 and result["cad_area"] > 0:
        result["area_ratio"] = result["stl_area"] / result["cad_area"]
    else:
        result["area_ratio"] = None

    return result, cad_aligned


def visualize_comparison(stl_mesh: trimesh.Trimesh, cad_mesh: trimesh.Trimesh,
                          centerline: np.ndarray, output_path: Path, title: str = "") -> None:
    """Create PyVista visualization comparing STL and CAD."""
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 900))

    stl_pv = pv.wrap(stl_mesh)
    cad_pv = pv.wrap(cad_mesh)

    plotter.add_mesh(
        stl_pv,
        color="lightblue",
        opacity=0.5,
        label="STL original",
        show_edges=True,
        edge_color="darkblue",
    )
    plotter.add_mesh(
        cad_pv,
        color="red",
        opacity=0.7,
        label="CAD build123d",
        show_edges=True,
        edge_color="darkred",
    )
    if centerline is not None:
        plotter.add_points(centerline, color="green", point_size=5, label="Centerline")

    plotter.add_legend()
    plotter.set_background("white")

    stl_bounds = stl_pv.bounds
    cad_bounds = cad_pv.bounds
    combined_center = (
        (stl_bounds[0] + stl_bounds[1] + cad_bounds[0] + cad_bounds[1]) / 4,
        (stl_bounds[2] + stl_bounds[3] + cad_bounds[2] + cad_bounds[3]) / 4,
        (stl_bounds[4] + stl_bounds[5] + cad_bounds[4] + cad_bounds[5]) / 4,
    )
    max_range = max(
        stl_bounds[1] - stl_bounds[0],
        stl_bounds[3] - stl_bounds[2],
        stl_bounds[5] - stl_bounds[4],
        cad_bounds[1] - cad_bounds[0],
        cad_bounds[3] - cad_bounds[2],
        cad_bounds[5] - cad_bounds[4],
    )
    plotter.camera.position = (
        combined_center[0] + max_range,
        combined_center[1] + max_range,
        combined_center[2] + max_range,
    )
    plotter.camera.focal_point = combined_center

    if title:
        plotter.add_text(title, position="upper_left", font_size=12)

    plotter.render()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_path))
    print(f"  Visualisation: {output_path}")


def main() -> int:
    print("=== Comparaison STL vs build123d ===")

    if not STL_PATH.exists():
        print(f"STL not found: {STL_PATH}")
        return 1
    stl_mesh = load_stl(STL_PATH)
    print(f"STL chargé: {len(stl_mesh.vertices)} vertices, {len(stl_mesh.faces)} faces")
    print(f"  Area: {stl_mesh.area:.6f} m²")
    print(f"  Watertight: {stl_mesh.is_watertight}")

    centerline = None
    if CENTERLINE_PATH.exists():
        centerline = np.load(str(CENTERLINE_PATH))
        print(f"Centerline chargée: {len(centerline)} points")

    all_metrics = {}
    for name, step_path in CAD_FILES.items():
        if not step_path.exists():
            print(f"\nSKIP {name}: {step_path} not found")
            continue

        print(f"\n--- {name} ---")
        try:
            cad_mesh = load_step_as_mesh(step_path)
            print(f"  CAD mesh: {len(cad_mesh.vertices)} vertices, {len(cad_mesh.faces)} faces")
        except Exception as exc:
            print(f"  Failed to load CAD: {exc}")
            continue

        metrics, cad_aligned = compare_meshes(stl_mesh, cad_mesh)
        all_metrics[name] = metrics

        for key, value in metrics.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")

        output_img = OUTPUT_DIR / f"compare_stl_vs_{name}.png"
        visualize_comparison(stl_mesh, cad_aligned, centerline, output_img, title=name)

    metrics_path = OUTPUT_DIR / "comparison_metrics_all.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMétriques: {metrics_path}")
    print("=== Terminé ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
