#!/usr/bin/env python3
"""Verify and visualize building geometry from VoxCity HDF5.

Compares raw footprints from HDF5 with processed individual footprints
after cleaning, simplification, and filtering.

Usage:
    PYTHONPATH=../../../src:. python3 verify_geometry.py \
        --hdf5 output/voxcity.h5 \
        --output geometry_verification.png \
        --mesh-size 5.0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "voxcity_export_work" / "src"))

import numpy as np
from foampilot.urban import Building, UrbanModel
from foampilot.urban.model.terrain import CFDTerrain


def load_raw_buildings_from_hdf5(hdf5_path: str, mesh_size: float = 5.0) -> UrbanModel:
    """Load raw building footprints from VoxCity HDF5 using the proper GeoDataFrame."""
    import shapely.ops
    from voxcity.io import load_voxcity

    h5_path = Path(hdf5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"VoxCity HDF5 not found: {hdf5_path}")

    print(f"  Loading VoxCity from HDF5: {hdf5_path}")
    voxcity = load_voxcity(str(h5_path))

    gdf = getattr(voxcity, "extras", {}).get("building_gdf")
    if gdf is None and hasattr(voxcity, "building_gdf"):
        gdf = voxcity.building_gdf
    if gdf is None or len(gdf) == 0:
        print("  WARNING: No building_gdf found in VoxCity HDF5")
        return UrbanModel()

    print(f"  Found {len(gdf)} buildings in building_gdf")

    try:
        from voxcity.geoprocessor.overlap import process_building_footprints_by_overlap
        gdf = process_building_footprints_by_overlap(gdf, overlap_threshold=0.5)
        print("  VoxCity overlap processing: merged buildings with >50% overlap")
    except Exception as e:
        print(f"  WARNING: VoxCity overlap processing failed ({e}), using raw GDF")

    urban = UrbanModel()
    count = 0

    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)

        def project(geom):
            if geom is None:
                return None
            return shapely.ops.transform(lambda x, y: transformer.transform(x, y), geom)
    except Exception:
        project = None

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "Polygon":
            footprints = [geom]
        elif geom.geom_type == "MultiPolygon":
            footprints = list(geom.geoms)
        else:
            continue

        for footprint_idx, footprint in enumerate(footprints):
            if project is not None:
                try:
                    footprint = project(footprint)
                except Exception:
                    pass

            if not footprint.is_valid:
                try:
                    footprint = footprint.buffer(0)
                except Exception:
                    print(f"  WARNING: Invalid footprint for building {idx}_{footprint_idx}, skipping")
                    continue

            if footprint.is_empty:
                continue

            height = float(getattr(row, "height", 9.0) or 9.0)
            building_id = f"vox_{idx}_{footprint_idx}" if len(footprints) > 1 else f"vox_{idx}"

            try:
                urban.add_building(Building(
                    id=building_id,
                    footprint=footprint,
                    ground_z=0.0,
                    roof_z=height,
                    source="voxcity",
                    confidence=0.7,
                ))
                count += 1
            except ValueError as e:
                print(f"  WARNING: Could not add building {building_id}: {e}")
                continue

    print(f"  Loaded {count} raw buildings from HDF5")
    return urban


def preprocess_individual(urban: UrbanModel, mesh_size: float = 5.0) -> UrbanModel:
    """Apply individual footprint preprocessing (clean + simplify) without merging."""
    import sys
    from pathlib import Path
    vector_builder_path = Path(__file__).resolve().parent.parent / "voxcity_export_work" / "src"
    sys.path.insert(0, str(vector_builder_path))
    from vector_builder import VectorGmshBuilder

    builder = VectorGmshBuilder(urban, CFDTerrain.flat(z=0.0), mesh_size=mesh_size)
    builder._preprocess_geometry()
    return builder.urban


def verify_geometry(raw_urban: UrbanModel, processed_urban: UrbanModel, output_path: Path, mesh_size: float):
    """Generate verification plots comparing raw and processed geometry."""
    import matplotlib.pyplot as plt

    raw_buildings = list(raw_urban.buildings())
    proc_buildings = list(processed_urban.buildings())

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Geometry Verification — {len(raw_buildings)} raw → {len(proc_buildings)} processed (mesh_size={mesh_size}m)", fontsize=14)

    ax1 = axes[0, 0]
    ax1.set_title("Raw Footprints")
    for b in raw_buildings:
        x, y = b.footprint.exterior.xy
        ax1.fill(x, y, alpha=0.5, edgecolor="black", linewidth=0.5)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.set_title("Processed Footprints (individual)")
    for b in proc_buildings:
        x, y = b.footprint.exterior.xy
        ax2.fill(x, y, alpha=0.5, edgecolor="black", linewidth=0.5)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    heights_raw = [b.roof_z - b.ground_z for b in raw_buildings]
    heights_proc = [b.roof_z - b.ground_z for b in proc_buildings]
    bins = min(20, max(max(len(heights_raw), len(heights_proc)), 1))
    ax3.hist(heights_raw, bins=bins, alpha=0.6, label=f"Raw ({len(heights_raw)})", color="steelblue")
    ax3.hist(heights_proc, bins=bins, alpha=0.6, label=f"Processed ({len(heights_proc)})", color="darkorange")
    ax3.set_xlabel("Building Height (m)")
    ax3.set_ylabel("Count")
    ax3.set_title("Height Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    areas_raw = [b.footprint.area for b in raw_buildings]
    areas_proc = [b.footprint.area for b in proc_buildings]
    ax4.hist(areas_raw, bins=bins, alpha=0.6, label=f"Raw ({len(areas_raw)})", color="steelblue")
    ax4.hist(areas_proc, bins=bins, alpha=0.6, label=f"Processed ({len(areas_proc)})", color="darkorange")
    ax4.set_xlabel("Footprint Area (m²)")
    ax4.set_ylabel("Count")
    ax4.set_title("Area Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Verification plot saved: {output_path}")


def print_statistics(raw_urban: UrbanModel, processed_urban: UrbanModel):
    """Print statistics comparing raw and processed geometry."""
    raw_buildings = list(raw_urban.buildings())
    proc_buildings = list(processed_urban.buildings())

    print("\n" + "=" * 60)
    print("Geometry Statistics")
    print("=" * 60)

    print(f"  Raw buildings:      {len(raw_buildings)}")
    print(f"  Processed buildings: {len(proc_buildings)}")
    print(f"  Filtered:           {len(raw_buildings) - len(proc_buildings)}")

    if raw_buildings:
        raw_heights = [b.roof_z - b.ground_z for b in raw_buildings]
        raw_areas = [b.footprint.area for b in raw_buildings]
        print("\n  Raw:")
        print(f"    Height: min={min(raw_heights):.1f}, max={max(raw_heights):.1f}, mean={np.mean(raw_heights):.1f}")
        print(f"    Area:   min={min(raw_areas):.1f}, max={max(raw_areas):.1f}, mean={np.mean(raw_areas):.1f}")
        print(f"    Total area: {sum(raw_areas):.1f} m²")

    if proc_buildings:
        proc_heights = [b.roof_z - b.ground_z for b in proc_buildings]
        proc_areas = [b.footprint.area for b in proc_buildings]
        print("\n  Processed:")
        print(f"    Height: min={min(proc_heights):.1f}, max={max(proc_heights):.1f}, mean={np.mean(proc_heights):.1f}")
        print(f"    Area:   min={min(proc_areas):.1f}, max={max(proc_areas):.1f}, mean={np.mean(proc_areas):.1f}")
        print(f"    Total area: {sum(proc_areas):.1f} m²")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Verify building geometry from VoxCity HDF5")
    parser.add_argument("--hdf5", required=True, help="Path to VoxCity HDF5 file")
    parser.add_argument("--output", default="geometry_verification.png", help="Output verification plot")
    parser.add_argument("--mesh-size", type=float, default=5.0, help="Mesh size for footprint simplification")
    args = parser.parse_args()

    print("=" * 60)
    print("Geometry Verification")
    print("=" * 60)

    raw_urban = load_raw_buildings_from_hdf5(args.hdf5, mesh_size=args.mesh_size)
    processed_urban = preprocess_individual(raw_urban, mesh_size=args.mesh_size)

    print_statistics(raw_urban, processed_urban)

    output_path = Path(args.output)
    verify_geometry(raw_urban, processed_urban, output_path, args.mesh_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
