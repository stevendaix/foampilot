#!/usr/bin/env python3
"""
Post-process a single VoxCity/OpenFOAM CFD case.

Pipeline:
  1. Convert OpenFOAM fields to VTK if needed (foamToVTK).
  2. Load the latest time-step with FoamPostProcessing.
  3. Compute derived fields: velocity magnitude, Cp on buildings.
  4. Generate per-case visualizations (slices, Cp, mesh, inlet profile).
  5. Export scalar statistics to JSON/CSV.

Usage:
    PYTHONPATH=src python3 voxcity_postprocess.py \
        --case /tmp/voxcity_vector_demo3 \
        --pedestrian-height 1.75 \
        --output-dir post
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing

RHO_AIR = 1.225
NU_AIR = 1.5e-5


def ensure_vtk(foam_post: FoamPostProcessing, fields=None):
    vtk_dir = foam_post.case_path / "VTK"
    if not vtk_dir.exists() or not list(vtk_dir.glob("*.vtk")):
        print("  Running foamToVTK...")
        foam_post.foamToVTK(fields=fields or ["U", "p", "k", "epsilon"])


def load_latest(foam_post: FoamPostProcessing):
    steps = foam_post.get_all_time_steps()
    if not steps:
        raise FileNotFoundError("No VTK time steps found.")
    latest = steps[-1]
    print(f"  Latest time-step: {latest} ({len(steps)} available)")
    return foam_post.load_time_step(latest)


def compute_velocity_magnitude(mesh: pv.DataSet):
    if "U" in mesh.point_data:
        U = mesh.point_data["U"]
        if U.ndim == 2 and U.shape[1] == 3:
            mesh.point_data["Umag"] = np.linalg.norm(U, axis=1)
        else:
            mesh.point_data["Umag"] = np.abs(U)
    return mesh


def compute_cp(mesh: pv.DataSet, u_ref: float):
    p_ref = 0.5 * RHO_AIR * u_ref**2
    if "p" in mesh.point_data and p_ref > 0:
        mesh.point_data["Cp"] = mesh.point_data["p"] / p_ref
    return mesh, p_ref


def generate_visualizations(case_dir: Path, structure, p_ref: float, pedestrian_height: float, output_dir: Path):
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    cell_mesh = structure["cell"]
    boundaries = structure.get("boundaries", {})
    bounds = cell_mesh.bounds
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cz = (bounds[4] + bounds[5]) / 2

    compute_velocity_magnitude(cell_mesh)

    pv.set_jupyter_backend("none")
    pv.global_theme.background = "white"

    # --- 1. Horizontal slice |U| at pedestrian height ---
    try:
        slice_mesh = cell_mesh.slice(normal="z", origin=(cx, cy, pedestrian_height))
        if slice_mesh.n_points > 0 and "Umag" in slice_mesh.point_data:
            pl = pv.Plotter(off_screen=True)
            pl.set_background("white")
            pl.add_mesh(
                slice_mesh,
                scalars="Umag",
                cmap="viridis",
                show_scalar_bar=True,
                scalar_bar_args={"title": "|U| (m/s)"},
            )
            pl.camera_position = "xy"
            pl.screenshot(str(viz_dir / "slice_pedestrian_velocity.png"))
            pl.close()
            print("  Saved: slice_pedestrian_velocity.png")
    except Exception as e:
        print(f"  slice_pedestrian_velocity error: {e}")

    # --- 2. Vertical slice p ---
    try:
        cy = (bounds[2] + bounds[3]) / 2
        slice_v = cell_mesh.slice(normal="y", origin=(cx, cy, cz))
        if slice_v.n_points > 0 and "p" in slice_v.point_data:
            pl = pv.Plotter(off_screen=True)
            pl.set_background("white")
            pl.add_mesh(
                slice_v,
                scalars="p",
                cmap="RdBu_r",
                show_scalar_bar=True,
                scalar_bar_args={"title": "p (Pa)"},
            )
            pl.camera_position = "xz"
            pl.screenshot(str(viz_dir / "slice_vertical_pressure.png"))
            pl.close()
            print("  Saved: slice_vertical_pressure.png")
    except Exception as e:
        print(f"  slice_vertical_pressure error: {e}")

    # --- 3. Vertical slice |U| ---
    try:
        if slice_v.n_points > 0 and "Umag" in slice_v.point_data:
            pl = pv.Plotter(off_screen=True)
            pl.set_background("white")
            pl.add_mesh(
                slice_v,
                scalars="Umag",
                cmap="viridis",
                show_scalar_bar=True,
                scalar_bar_args={"title": "|U| (m/s)"},
            )
            pl.camera_position = "xz"
            pl.screenshot(str(viz_dir / "slice_vertical_velocity.png"))
            pl.close()
            print("  Saved: slice_vertical_velocity.png")
    except Exception as e:
        print(f"  slice_vertical_velocity error: {e}")

    # --- 4. Cp on buildings ---
    try:
        build_mesh = boundaries.get("buildings")
        if build_mesh is not None:
            if "p" in build_mesh.point_data:
                build_mesh, _ = compute_cp(build_mesh, u_ref=np.sqrt(2 * p_ref / RHO_AIR))
            if "Cp" in build_mesh.point_data:
                pl = pv.Plotter(off_screen=True)
                pl.set_background("white")
                pl.add_mesh(
                    build_mesh,
                    scalars="Cp",
                    cmap="RdBu_r",
                    show_scalar_bar=True,
                    scalar_bar_args={"title": "Cp"},
                )
                pl.camera_position = "xy"
                pl.screenshot(str(viz_dir / "buildings_cp.png"))
                pl.close()
                print("  Saved: buildings_cp.png")
    except Exception as e:
        print(f"  buildings_cp error: {e}")

    # --- 4b. Buildings only ---
    try:
        build_mesh = boundaries.get("buildings")
        if build_mesh is not None:
            pl = pv.Plotter(off_screen=True)
            pl.set_background("white")
            pl.add_mesh(
                build_mesh,
                color="lightblue",
                show_edges=True,
                edge_color="black",
                line_width=0.5,
                opacity=0.95,
            )
            pl.camera_position = "xy"
            pl.screenshot(str(viz_dir / "buildings_only.png"))
            pl.close()
            print("  Saved: buildings_only.png")
    except Exception as e:
        print(f"  buildings_only error: {e}")

    # --- 5. Horizontal |U| with buildings overlay ---
    try:
        ped_slice = cell_mesh.slice(normal="z", origin=(cx, cy, pedestrian_height))
        if ped_slice.n_points > 0 and "Umag" in ped_slice.point_data:
            pl = pv.Plotter(off_screen=True)
            pl.set_background("white")
            pl.add_mesh(
                ped_slice,
                scalars="Umag",
                cmap="viridis",
                show_scalar_bar=True,
                scalar_bar_args={"title": "|U| (m/s)"},
            )
            build_mesh = boundaries.get("buildings")
            if build_mesh is not None:
                pl.add_mesh(build_mesh, color="black", opacity=0.4)
            pl.camera_position = "xy"
            pl.screenshot(str(viz_dir / "pedestrian_velocity_overlay.png"))
            pl.close()
            print("  Saved: pedestrian_velocity_overlay.png")
    except Exception as e:
        print(f"  pedestrian_velocity_overlay error: {e}")

    # --- 6. Mesh wireframe ---
    try:
        pl = pv.Plotter(off_screen=True)
        pl.set_background("black")
        pl.add_mesh(
            cell_mesh,
            style="wireframe",
            color="white",
            line_width=0.3,
            opacity=0.7,
        )
        pl.camera_position = "xy"
        pl.screenshot(str(viz_dir / "mesh_wireframe.png"))
        pl.close()
        print("  Saved: mesh_wireframe.png")
    except Exception as e:
        print(f"  mesh_wireframe error: {e}")


def export_statistics(case_dir: Path, structure, output_dir: Path, u_ref: float):
    stats_dir = output_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    cell_mesh = structure["cell"]
    boundaries = structure.get("boundaries", {})

    stats = {
        "case": str(case_dir),
        "u_ref_m_s": float(u_ref),
        "p_ref_Pa": float(0.5 * RHO_AIR * u_ref**2),
        "num_points": int(cell_mesh.n_points),
        "num_cells": int(cell_mesh.n_cells),
        "bounds": [float(b) for b in cell_mesh.bounds],
    }

    # Scalar stats for point data
    for field in ["Umag", "p", "k", "epsilon"]:
        if field in cell_mesh.point_data:
            data = cell_mesh.point_data[field]
            stats[f"{field}_mean"] = float(np.mean(data))
            stats[f"{field}_std"] = float(np.std(data))
            stats[f"{field}_min"] = float(np.min(data))
            stats[f"{field}_max"] = float(np.max(data))

    # Cp stats on buildings
    build_mesh = boundaries.get("buildings")
    if build_mesh is not None and "Cp" in build_mesh.point_data:
        cp = build_mesh.point_data["Cp"]
        stats["cp_mean"] = float(np.mean(cp))
        stats["cp_std"] = float(np.std(cp))
        stats["cp_min"] = float(np.min(cp))
        stats["cp_max"] = float(np.max(cp))

    stats_path = stats_dir / "case_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    # CSV export for cell centers
    try:
        centers = cell_mesh.cell_centers()
        data_to_export = {
            "X": centers.points[:, 0],
            "Y": centers.points[:, 1],
            "Z": centers.points[:, 2],
        }
        for field in ["Umag", "p"]:
            if field in cell_mesh.point_data:
                cell_data = centers.point_data_to_cell_data(pass_point_data=False)
                if field in cell_data.cell_data:
                    data_to_export[field] = cell_data.cell_data[field]

        csv_path = stats_dir / "cell_data.csv"
        import pandas as pd
        pd.DataFrame(data_to_export).to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
    except Exception as e:
        print(f"  CSV export error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Post-process a VoxCity OpenFOAM case")
    parser.add_argument("--case", required=True, help="Path to OpenFOAM case directory")
    parser.add_argument("--pedestrian-height", type=float, default=1.75, help="Pedestrian height (m)")
    parser.add_argument("--output-dir", default="post", help="Output directory")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualizations")
    args = parser.parse_args()

    case_dir = Path(args.case)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VoxCity OpenFOAM post-processing")
    print("=" * 60)
    print(f"Case: {case_dir}")

    foam_post = FoamPostProcessing(case_path=case_dir)
    ensure_vtk(foam_post)
    structure = load_latest(foam_post)

    # Reference speed from metadata or fallback
    u_ref = 10.0
    meta_path = case_dir / "case_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        u_ref = float(meta.get("speed_10m", u_ref))

    print(f"  u_ref = {u_ref:.2f} m/s, p_ref = {0.5 * RHO_AIR * u_ref**2:.2f} Pa")

    # Compute Cp on buildings if pressure available
    build_mesh = structure.get("boundaries", {}).get("buildings")
    if build_mesh is not None and "p" in build_mesh.point_data:
        compute_cp(build_mesh, u_ref=u_ref)

    if not args.skip_viz:
        print("\nGenerating visualizations...")
        generate_visualizations(case_dir, structure, p_ref=0.5 * RHO_AIR * u_ref**2,
                                pedestrian_height=args.pedestrian_height, output_dir=output_dir)

    print("\nExporting statistics...")
    export_statistics(case_dir, structure, output_dir, u_ref=u_ref)

    print("\n" + "=" * 60)
    print(f"Post-processing complete. Results in: {output_dir}")
    print(f"  {output_dir}/visualizations/ — PNG plots")
    print(f"  {output_dir}/statistics/    — JSON + CSV data")
    print("=" * 60)


if __name__ == "__main__":
    main()
