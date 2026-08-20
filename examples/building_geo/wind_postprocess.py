#!/usr/bin/env python3
"""
Post-process a single wind CFD case.

Usage:
    PYTHONPATH=../../src python3 wind_postprocess.py \
        --case cases/wind_270deg \
        --pedestrian-height 1.75
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pyvista as pv

from foampilot.postprocess.openfoam_direct import OpenFOAMDirectReader

RHO_AIR = 1.225


def load_case_speed(case_dir: Path) -> float:
    meta_path = case_dir / "case_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return float(meta.get("speed_10m", 10.0))
    return 10.0


def extract_boundary_mesh(reader: OpenFOAMDirectReader, patch_name: str) -> pv.UnstructuredGrid:
    """Extract a boundary patch mesh from OpenFOAMDirectReader."""
    patches = reader.boundary_patches
    if patch_name not in patches:
        return None

    patch_info = patches[patch_name]
    start_face = patch_info.get("startFace", 0)
    n_faces = patch_info.get("nFaces", 0)

    if n_faces == 0:
        return None

    surface = reader.mesh.extract_surface()
    if surface.n_points == 0:
        return None

    face_ids = np.arange(start_face, start_face + n_faces)
    if surface.n_cells < start_face + n_faces:
        return None

    try:
        boundary_mesh = surface.extract_cells(face_ids)
        return boundary_mesh
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Post-process a single wind CFD case")
    parser.add_argument("--case", required=True, help="Case directory")
    parser.add_argument("--pedestrian-height", type=float, default=1.75)
    parser.add_argument("--output-dir", default="post")
    args = parser.parse_args()

    case_dir = Path(args.case)
    output_dir = Path(args.output_dir)
    viz_dir = output_dir / "carto_par_cas"
    viz_dir.mkdir(parents=True, exist_ok=True)

    actual_speed = load_case_speed(case_dir)
    p_ref = 0.5 * RHO_AIR * actual_speed ** 2
    print(f"Case: {case_dir.name}")
    print(f"  u_ref = {actual_speed:.2f} m/s")
    print(f"  p_ref = {p_ref:.2f} Pa")

    reader = OpenFOAMDirectReader(case_path=case_dir)
    time_steps = reader.get_time_steps()
    if not time_steps:
        print("  No time steps found.")
        sys.exit(1)

    latest = time_steps[-1]
    print(f"  Latest time-step: {latest}")

    cell_mesh = reader.mesh
    reader.attach_field("U", time_step=latest)
    reader.attach_field("p", time_step=latest)
    reader.attach_field("k", time_step=latest)
    reader.attach_field("epsilon", time_step=latest)

    boundaries = {}
    for patch_name in reader.boundary_patches:
        boundary_mesh = extract_boundary_mesh(reader, patch_name)
        if boundary_mesh is not None:
            boundaries[patch_name] = boundary_mesh

    bounds = cell_mesh.bounds

    pv.set_jupyter_backend("none")
    pv.global_theme.background = "white"

    # 1. Horizontal slice at pedestrian height — |U|
    try:
        slice_mesh = cell_mesh.slice(normal="z", origin=(0, 0, args.pedestrian_height))
        if slice_mesh.n_points > 0:
            U = slice_mesh.cell_data.get("U")
            if U is not None:
                mag = np.linalg.norm(U, axis=1)
                slice_mesh.cell_data["velocity_magnitude"] = mag
            pl = pv.Plotter(off_screen=True)
            pl.set_background("white")
            pl.add_mesh(
                slice_mesh, scalars="velocity_magnitude",
                cmap="viridis", show_scalar_bar=True,
                scalar_bar_args={"title": "|U| (m/s)"},
            )
            pl.camera_position = "xy"
            pl.screenshot(str(viz_dir / f"{case_dir.name}_slice_pedestrian.png"))
            pl.close()
            print("  Saved: slice_pedestrian.png")
    except Exception as e:
        print(f"  slice_pedestrian error: {e}")

    # 2. Horizontal slice at 1m — |U|
    try:
        slice_1m = cell_mesh.slice(normal="z", origin=(0, 0, 1.0))
        if slice_1m.n_points > 0 and "U" in slice_1m.cell_data:
            U = slice_1m.cell_data["U"]
            u_mag = np.linalg.norm(U, axis=1)
            slice_1m.cell_data["velocity_magnitude"] = u_mag
            pl = pv.Plotter(off_screen=True)
            pl.set_background("white")
            pl.add_mesh(
                slice_1m, scalars="velocity_magnitude",
                cmap="viridis", show_scalar_bar=True,
                scalar_bar_args={"title": "|U| (m/s)"},
            )
            pl.camera_position = "xy"
            pl.screenshot(str(viz_dir / f"{case_dir.name}_horizontal_u_1m.png"))
            pl.close()
            print("  Saved: horizontal_u_1m.png")
    except Exception as e:
        print(f"  horizontal_u_1m error: {e}")

    # 3. Cp on buildings
    try:
        build_mesh = boundaries.get("buildings")
        if build_mesh is not None and "p" in build_mesh.cell_data:
            p = build_mesh.cell_data["p"]
            cp = p / p_ref
            build_mesh.cell_data["Cp"] = cp
            pl = pv.Plotter(off_screen=True)
            pl.set_background("white")
            pl.add_mesh(
                build_mesh, scalars="Cp", cmap="RdBu_r",
                show_scalar_bar=True, scalar_bar_args={"title": "Cp"},
                cpos="xy",
            )
            pl.camera_position = "xy"
            pl.screenshot(str(viz_dir / f"{case_dir.name}_cp_buildings.png"))
            pl.close()
            print("  Saved: cp_buildings.png")
    except Exception as e:
        print(f"  cp_buildings error: {e}")

    # 4. Velocity magnitude on full mesh slice
    try:
        slice_full = cell_mesh.slice(normal="z", origin=(0, 0, args.pedestrian_height))
        if slice_full.n_points > 0 and "U" in slice_full.cell_data:
            U = slice_full.cell_data["U"]
            mag = np.linalg.norm(U, axis=1)
            slice_full.cell_data["velocity_magnitude"] = mag
            pl = pv.Plotter(off_screen=True)
            pl.set_background("white")
            pl.add_mesh(
                slice_full, scalars="velocity_magnitude",
                cmap="viridis", show_scalar_bar=True,
                scalar_bar_args={"title": "|U| (m/s)"},
            )
            pl.camera_position = "xy"
            pl.screenshot(str(viz_dir / f"{case_dir.name}_contour_velocity_pedestrian.png"))
            pl.close()
            print("  Saved: contour_velocity_pedestrian.png")
    except Exception as e:
        print(f"  contour_velocity error: {e}")

    # 5. Vertical slice — pressure
    try:
        cy = (bounds[2] + bounds[3]) / 2
        slice_v = cell_mesh.slice(normal="y", origin=(0, cy, 0))
        if slice_v.n_points > 0:
            pl = pv.Plotter(off_screen=True)
            pl.set_background("white")
            pl.add_mesh(
                slice_v, scalars="p", cmap="RdBu_r",
                show_scalar_bar=True, scalar_bar_args={"title": "p (Pa)"},
            )
            pl.camera_position = "xz"
            pl.screenshot(str(viz_dir / f"{case_dir.name}_slice_vertical_p.png"))
            pl.close()
            print("  Saved: slice_vertical_p.png")
    except Exception as e:
        print(f"  slice_vertical_p error: {e}")

    # 6. Vertical slice — |U|
    try:
        if slice_v.n_points > 0 and "U" in slice_v.cell_data:
            U = slice_v.cell_data["U"]
            u_mag = np.linalg.norm(U, axis=1)
            slice_v.cell_data["velocity_magnitude"] = u_mag
            pl = pv.Plotter(off_screen=True)
            pl.set_background("white")
            pl.add_mesh(
                slice_v, scalars="velocity_magnitude",
                cmap="viridis", show_scalar_bar=True,
                scalar_bar_args={"title": "|U| (m/s)"},
            )
            pl.camera_position = "xz"
            pl.screenshot(str(viz_dir / f"{case_dir.name}_slice_vertical_u.png"))
            pl.close()
            print("  Saved: slice_vertical_u.png")
    except Exception as e:
        print(f"  slice_vertical_u error: {e}")

    print(f"\nPost-processing complete. Results in: {viz_dir}")


if __name__ == "__main__":
    main()
