#!/usr/bin/env python3
"""
VoxCity-aware dedicated post-processing for OpenFOAM urban CFD cases.

This post-processor enriches the standard CFD post-processing with
VoxCity-specific analysis:
    - Compare CFD mesh buildings against original VoxCity footprints
    - Building-by-building statistics (Cp, velocity, turbulence)
    - Pedestrian-level comfort maps (speed, turbulence intensity)
    - Wind comfort classification ( Lawson / NEN )
    - VoxCity metadata integration

Usage:
    PYTHONPATH=../../foampilot/src python3 voxcity_dedicated_postprocess.py \
        --case neighborhood_case \
        --hdf5 output/voxcity.h5 \
        --pedestrian-height 1.75
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing

RHO_AIR = 1.225
NU_AIR = 1.5e-5


def load_voxcity_buildings(hdf5_path: str) -> dict:
    """Load VoxCity building metadata from HDF5."""
    buildings = {}
    try:
        with h5py.File(hdf5_path, "r") as f:
            vox = f.get("voxcity", {})
            if "building_height" not in vox or "building_id" not in vox:
                return buildings

            heights = vox["building_height"][:]
            ids = vox["building_id"][:]
            unique_ids = np.unique(ids[ids > 0])

            for bid in unique_ids:
                mask = ids == bid
                if not np.any(mask):
                    continue
                rows, cols = np.where(mask)
                if len(rows) < 4:
                    continue
                h = float(heights[mask].mean())
                buildings[int(bid)] = {
                    "voxcity_id": int(bid),
                    "height_m": h if np.isfinite(h) else 9.0,
                    "grid_area_m2": float(len(rows)),
                    "rows": rows.tolist(),
                    "cols": cols.tolist(),
                }
    except Exception as e:
        print(f"  WARNING: Could not load VoxCity buildings: {e}")
    return buildings


def compute_velocity_magnitude(mesh: pv.DataSet):
    if "U" in mesh.point_data:
        U = mesh.point_data["U"]
        if U.ndim == 2 and U.shape[1] == 3:
            mesh.point_data["Umag"] = np.linalg.norm(U, axis=1)
        else:
            mesh.point_data["Umag"] = np.abs(U)
    return mesh


def compute_turbulence_intensity(mesh: pv.DataSet):
    if "k" in mesh.point_data and "Umag" in mesh.point_data:
        k = mesh.point_data["k"]
        U = mesh.point_data["Umag"]
        if k.ndim == 1 and U.ndim == 1:
            I = np.sqrt(2.0 * k / 3.0) / np.maximum(U, 1e-6)
            mesh.point_data["TI"] = np.clip(I, 0.0, 1.0)
    return mesh


def classify_wind_comfort(ti: float, u: float) -> str:
    """Simple wind comfort classification based on NEN."""
    if u < 2.0:
        return "calm"
    elif u < 5.0:
        if ti < 0.1:
            return "comfortable"
        elif ti < 0.2:
            return "moderate"
        else:
            return "uncomfortable"
    elif u < 10.0:
        if ti < 0.15:
            return "moderate"
        else:
            return "uncomfortable"
    else:
        return "dangerous"


def generate_visualizations(case_dir: Path, structure: dict, p_ref: float,
                            pedestrian_height: float, output_dir: Path,
                            voxcity_buildings: dict, domain_bounds: tuple = None,
                            voxcity_gdf = None):
    """Generate VoxCity-aware visualizations."""
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    cell_mesh = structure["cell"]
    boundaries = structure.get("boundaries", {})

    compute_velocity_magnitude(cell_mesh)
    compute_turbulence_intensity(cell_mesh)

    pv.set_jupyter_backend("none")
    pv.global_theme.background = "white"

    if domain_bounds is not None:
        xmin, ymin, zmin, xmax, ymax, zmax = domain_bounds
    else:
        bounds = cell_mesh.bounds
        xmin, ymin, zmin, xmax, ymax, zmax = bounds

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    # --- 0. Map view with folium ---
    try:
        import folium
        from folium import FeatureGroup, LayerControl
        import json

        m = folium.Map(location=[(ymin + ymax) / 2, (xmin + xmax) / 2], zoom_start=16, tiles="CartoDB positron")
        buildings_layer = FeatureGroup(name="Buildings")
        if voxcity_gdf is not None:
            try:
                geojson_data = None
                if hasattr(voxcity_gdf, "__geo_interface__"):
                    geojson_data = json.loads(json.dumps(voxcity_gdf.__geo_interface__, default=str))
                elif hasattr(voxcity_gdf, "to_json"):
                    geojson_data = json.loads(voxcity_gdf.to_json())
                if geojson_data is not None:
                    folium.GeoJson(
                        geojson_data,
                        style_function=lambda x: {"fillColor": "#888888", "color": "#333333", "weight": 1, "fillOpacity": 0.6},
                        name="Buildings",
                    ).add_to(buildings_layer)
            except Exception as e:
                print(f"  map_view buildings layer error: {e}")
        buildings_layer.add_to(m)
        folium.Rectangle(
            bounds=[[ymin, xmin], [ymax, xmax]],
            color="red", weight=2, fill=False, popup="CFD Domain"
        ).add_to(m)
        folium.LayerControl().add_to(m)
        map_path = viz_dir / "map_view.html"
        m.save(str(map_path))
        print(f"  Saved: map_view.html")
    except Exception as e:
        print(f"  map_view error: {e}")

    # --- 1. Horizontal slice |U| at pedestrian height (focused on domain) ---
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

    # --- 2. Wind comfort map at pedestrian height ---
    try:
        slice_mesh = cell_mesh.slice(normal="z", origin=(cx, cy, pedestrian_height))
        if slice_mesh.n_points > 0 and "Umag" in slice_mesh.point_data and "TI" in slice_mesh.point_data:
            U = slice_mesh.point_data["Umag"]
            TI = slice_mesh.point_data["TI"]
            n = len(U)
            comfort = np.array([classify_wind_comfort(ti, u) for ti, u in zip(TI, U)])
            slice_mesh.point_data["comfort"] = comfort

            colors = {
                "calm": "#2196F3",
                "comfortable": "#4CAF50",
                "moderate": "#FFC107",
                "uncomfortable": "#FF5722",
                "dangerous": "#B71C1C",
            }
            pl = pv.Plotter(off_screen=True)
            pl.set_background("white")
            for label in ["calm", "comfortable", "moderate", "uncomfortable", "dangerous"]:
                mask = comfort == label
                if np.any(mask):
                    sub = slice_mesh.extract_points(mask)
                    pl.add_mesh(sub, color=colors[label], label=label)
            pl.add_legend()
            pl.camera_position = "xy"
            pl.screenshot(str(viz_dir / "wind_comfort_map.png"))
            pl.close()
            print("  Saved: wind_comfort_map.png")
    except Exception as e:
        print(f"  wind_comfort_map error: {e}")

    # --- 3. Cp on buildings ---
    try:
        build_mesh = boundaries.get("buildings")
        if build_mesh is not None and "p" in build_mesh.point_data:
            u_ref = 10.0
            p_ref = 0.5 * RHO_AIR * u_ref**2
            if p_ref > 0:
                build_mesh.point_data["Cp"] = build_mesh.point_data["p"] / p_ref
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

    # --- 4. Turbulence intensity at pedestrian height ---
    try:
        slice_mesh = cell_mesh.slice(normal="z", origin=(cx, cy, pedestrian_height))
        if slice_mesh.n_points > 0 and "TI" in slice_mesh.point_data:
            pl = pv.Plotter(off_screen=True)
            pl.set_background("white")
            pl.add_mesh(
                slice_mesh,
                scalars="TI",
                cmap="plasma",
                show_scalar_bar=True,
                scalar_bar_args={"title": "Turbulence Intensity"},
                clim=[0, 0.5],
            )
            pl.camera_position = "xy"
            pl.screenshot(str(viz_dir / "slice_pedestrian_ti.png"))
            pl.close()
            print("  Saved: slice_pedestrian_ti.png")
    except Exception as e:
        print(f"  slice_pedestrian_ti error: {e}")

    # --- 5. Vertical slice |U| through domain center ---
    try:
        slice_v = cell_mesh.slice(normal="y", origin=(cx, cy, 0))
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


def export_voxcity_statistics(case_dir: Path, structure: dict, output_dir: Path,
                              u_ref: float, voxcity_buildings: dict):
    """Export VoxCity-specific statistics."""
    stats_dir = output_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    cell_mesh = structure["cell"]
    boundaries = structure.get("boundaries", {})

    compute_velocity_magnitude(cell_mesh)
    compute_turbulence_intensity(cell_mesh)

    stats = {
        "case": str(case_dir),
        "u_ref_m_s": float(u_ref),
        "p_ref_Pa": float(0.5 * RHO_AIR * u_ref**2),
        "num_points": int(cell_mesh.n_points),
        "num_cells": int(cell_mesh.n_cells),
        "bounds": [float(b) for b in cell_mesh.bounds],
        "voxcity_buildings_in_hdf5": len(voxcity_buildings),
    }

    # Scalar stats for point data
    for field in ["Umag", "p", "k", "epsilon", "TI"]:
        if field in cell_mesh.point_data:
            data = cell_mesh.point_data[field]
            stats[f"{field}_mean"] = float(np.mean(data))
            stats[f"{field}_std"] = float(np.std(data))
            stats[f"{field}_min"] = float(np.min(data))
            stats[f"{field}_max"] = float(np.max(data))

    # Cp stats on buildings
    build_mesh = boundaries.get("buildings")
    if build_mesh is not None and "p" in build_mesh.point_data:
        p_ref = 0.5 * RHO_AIR * u_ref**2
        if p_ref > 0:
            build_mesh.point_data["Cp"] = build_mesh.point_data["p"] / p_ref
        if "Cp" in build_mesh.point_data:
            cp = build_mesh.point_data["Cp"]
            stats["cp_mean"] = float(np.mean(cp))
            stats["cp_std"] = float(np.std(cp))
            stats["cp_min"] = float(np.min(cp))
            stats["cp_max"] = float(np.max(cp))

    # Wind comfort distribution
    if "Umag" in cell_mesh.point_data and "TI" in cell_mesh.point_data:
        U = cell_mesh.point_data["Umag"]
        TI = cell_mesh.point_data["TI"]
        comfort_counts = {}
        for label in ["calm", "comfortable", "moderate", "uncomfortable", "dangerous"]:
            comfort_counts[label] = 0
        for u, ti in zip(U, TI):
            label = classify_wind_comfort(float(ti), float(u))
            comfort_counts[label] += 1
        stats["wind_comfort_distribution"] = comfort_counts

    # Building-specific stats if available
    if voxcity_buildings and build_mesh is not None:
        building_stats = []
        if "Cp" in build_mesh.point_data:
            cp_data = build_mesh.point_data["Cp"]
            for bid, bmeta in voxcity_buildings.items():
                building_stats.append({
                    "voxcity_id": bid,
                    "height_m": bmeta["height_m"],
                    "cp_mean": float(np.mean(cp_data)) if len(cp_data) > 0 else None,
                })
        stats["building_stats"] = building_stats

    stats_path = stats_dir / "voxcity_case_statistics.json"
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
        for field in ["Umag", "p", "TI"]:
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


def run_voxcity_postprocessing(case_dir: Path, output_dir: Path, hdf5_path: str = None,
                                speed: float = 10.0, pedestrian_height: float = 1.75,
                                domain_bounds: tuple = None, voxcity_gdf = None):
    """Run the complete VoxCity-aware post-processing pipeline."""
    case_dir = Path(case_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VoxCity-aware OpenFOAM post-processing")
    print("=" * 60)
    print(f"Case: {case_dir}")

    # Load VoxCity metadata
    voxcity_buildings = {}
    if hdf5_path and Path(hdf5_path).exists():
        print(f"\nLoading VoxCity metadata from {hdf5_path}...")
        voxcity_buildings = load_voxcity_buildings(hdf5_path)
        print(f"  VoxCity buildings: {len(voxcity_buildings)}")
        try:
            from voxcity.io import load_voxcity
            vox = load_voxcity(hdf5_path)
            gdf = getattr(vox, "extras", {}).get("building_gdf")
            if gdf is not None and len(gdf) > 0:
                voxcity_gdf = gdf
        except Exception:
            pass
    else:
        print(f"\nNo VoxCity HDF5 provided, running standard post-processing")

    # Load OpenFOAM results
    foam_post = FoamPostProcessing(case_path=case_dir)
    vtk_dir = case_dir / "VTK"
    if not vtk_dir.exists() or not list(vtk_dir.glob("*.vtk")):
        print("  Running foamToVTK...")
        foam_post.foamToVTK(fields=["U", "p", "k", "epsilon"])

    steps = foam_post.get_all_time_steps()
    if not steps:
        raise FileNotFoundError("No VTK time steps found.")
    latest = steps[-1]
    print(f"  Latest time-step: {latest} ({len(steps)} available)")

    structure = foam_post.load_time_step(latest)
    p_ref = 0.5 * RHO_AIR * speed**2

    if domain_bounds is None:
        try:
            boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
            content = boundary_file.read_text()
            import re
            xmin = ymin = zmin = float('inf')
            xmax = ymax = zmax = float('-inf')
            for patch in re.findall(r'(\w+)\s*\{[^}]*startFace\s+(\d+)', content):
                pass
            cell_mesh = structure["cell"]
            bounds = cell_mesh.bounds
            xmin, ymin, zmin, xmax, ymax, zmax = bounds
        except Exception:
            bounds = structure["cell"].bounds
            xmin, ymin, zmin, xmax, ymax, zmax = bounds
        domain_bounds = (xmin, ymin, zmin, xmax, ymax, zmax)

    # Compute derived fields
    cell_mesh = structure["cell"]
    boundaries = structure.get("boundaries", {})
    compute_velocity_magnitude(cell_mesh)
    compute_turbulence_intensity(cell_mesh)

    build_mesh = boundaries.get("buildings")
    if build_mesh is not None and "p" in build_mesh.point_data and p_ref > 0:
        build_mesh.point_data["Cp"] = build_mesh.point_data["p"] / p_ref

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(case_dir, structure, p_ref, pedestrian_height, output_dir,
                            voxcity_buildings, domain_bounds=domain_bounds, voxcity_gdf=voxcity_gdf)

    # Export statistics
    print("\nExporting statistics...")
    export_voxcity_statistics(case_dir, structure, output_dir, speed, voxcity_buildings)

    print("\n" + "=" * 60)
    print(f"Post-processing complete. Results in: {output_dir}")
    print(f"  {output_dir}/visualizations/ — PNG plots + map")
    print(f"  {output_dir}/statistics/    — JSON + CSV data")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="VoxCity-aware OpenFOAM post-processing")
    parser.add_argument("--case", required=True, help="Path to OpenFOAM case directory")
    parser.add_argument("--hdf5", default=None, help="Path to VoxCity HDF5 file")
    parser.add_argument("--pedestrian-height", type=float, default=1.75, help="Pedestrian height (m)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--speed", type=float, default=10.0, help="Reference wind speed (m/s)")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualizations")
    args = parser.parse_args()

    case_dir = Path(args.case)
    output_dir = Path(args.output_dir) if args.output_dir else case_dir / "post"
    run_voxcity_postprocessing(
        case_dir, output_dir,
        hdf5_path=args.hdf5,
        speed=args.speed,
        pedestrian_height=args.pedestrian_height,
    )


if __name__ == "__main__":
    main()
