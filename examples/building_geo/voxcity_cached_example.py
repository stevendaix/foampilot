#!/usr/bin/env python3
"""
VoxCity -> OpenFOAM example using cached VoxCity data.

This script loads a pre-downloaded VoxCity HDF5 model and converts it
to OpenFOAM using the vector Gmsh path (no STL).

Usage:
    PYTHONPATH=../../src python3 voxcity_cached_example.py \
        --hdf5 /home/steven/foampilot/output/voxcity_test/voxcity.h5 \
        --output cases/voxcity_cached_demo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

import h5py
import numpy as np
from foampilot.urban import Building, UrbanModel
from foampilot.urban.model.terrain import CFDTerrain
from foampilot.urban.readers.voxcity_reader import VoxCityReader
from shapely.geometry import Polygon

# Add the voxcity_export_work/src to path for vector builder
sys.path.insert(0, str(Path(__file__).resolve().parent / "voxcity_export_work" / "src"))
from vector_builder import VectorGmshBuilder


def load_voxcity_hdf5(hdf5_path: str) -> tuple[UrbanModel, CFDTerrain]:
    """Load UrbanModel and CFDTerrain from a VoxCity HDF5 file."""
    urban = UrbanModel()
    terrain = CFDTerrain()

    with h5py.File(hdf5_path, 'r') as f:
        voxcity_group = f['voxcity']
        
        # Load DEM
        dem_data = voxcity_group['dem'][:]
        if 'x' in voxcity_group:
            xs = voxcity_group['x'][:]
            ys = voxcity_group['y'][:]
        else:
            # Infer from shape
            ny, nx = dem_data.shape
            xs = np.arange(nx, dtype=float)
            ys = np.arange(ny, dtype=float)
        
        for i in range(ny):
            for j in range(nx):
                x = float(xs[j]) if hasattr(xs, '__len__') else float(j)
                y = float(ys[i]) if hasattr(ys, '__len__') else float(i)
                z = float(dem_data[i, j])
                if np.isfinite(z):
                    terrain.add_point(x, y, z)

        # Load building footprints and heights
        if 'extras_gdf' in voxcity_group:
            extras_gdf = voxcity_group['extras_gdf']
            # Try to reconstruct GeoDataFrame-like structure
            building_count = 0
            for key in extras_gdf:
                if key.startswith('building_'):
                    building_count += 1
            
            print(f"  Found {building_count} building entries in HDF5")
            
            # Alternative: use building_height and building_id grids
            if 'building_height' in voxcity_group and 'building_id' in voxcity_group:
                heights = voxcity_group['building_height'][:]
                ids = voxcity_group['building_id'][:]
                
                # Find unique buildings
                unique_ids = np.unique(ids[ids > 0])
                print(f"  Unique building IDs: {len(unique_ids)}")
                
                for bid in unique_ids[:20]:  # Limit to first 20 for testing
                    mask = ids == bid
                    if not np.any(mask):
                        continue
                    
                    # Get footprint from mask
                    rows, cols = np.where(mask)
                    if len(rows) < 4:
                        continue
                    
                    min_row, max_row = rows.min(), rows.max()
                    min_col, max_col = cols.min(), cols.max()
                    
                    # Create simple rectangular footprint
                    x0 = float(min_col)
                    x1 = float(max_col + 1)
                    y0 = float(min_row)
                    y1 = float(max_row + 1)
                    
                    footprint = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                    
                    # Get height
                    height = float(heights[mask].mean())
                    if height <= 0 or np.isnan(height):
                        height = 9.0
                    
                    building_id = f"b_{int(bid)}"
                    urban.add_building(Building(
                        id=building_id,
                        footprint=footprint,
                        ground_z=0.0,
                        roof_z=height,
                        source="voxcity_hdf5",
                        confidence=0.8,
                    ))

    print(f"  Loaded {urban.building_count()} buildings from HDF5")
    return urban, terrain


def main():
    parser = argparse.ArgumentParser(description="VoxCity cached -> vector Gmsh -> OpenFOAM")
    parser.add_argument("--hdf5", required=True, help="Path to VoxCity HDF5 file")
    parser.add_argument("--output", default="cases/voxcity_cached_demo", help="Output case directory")
    parser.add_argument("--mesh-size", type=float, default=5.0, help="Mesh size for Gmsh")
    parser.add_argument("--nb-proc", type=int, default=1, help="Number of processes")
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5)
    if not hdf5_path.exists():
        print(f"ERROR: HDF5 file not found: {hdf5_path}")
        sys.exit(1)

    case_dir = Path(args.output)
    case_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VoxCity cached -> vector Gmsh -> OpenFOAM")
    print("=" * 60)
    print(f"HDF5 file: {hdf5_path}")
    print(f"Output: {case_dir}")

    # Step 1: Load VoxCity data
    print("\n[1/4] Loading VoxCity data from HDF5...")
    urban, terrain = load_voxcity_hdf5(str(hdf5_path))
    print(f"  Buildings: {urban.building_count()}")
    print(f"  Terrain points: {len(terrain.points)}")

    if urban.building_count() == 0:
        print("WARNING: No buildings loaded, using synthetic fallback")
        from foampilot.urban import Building, UrbanModel
        from shapely.geometry import Polygon
        urban = UrbanModel()
        urban.add_building(Building(
            id="B1",
            footprint=Polygon([(0, 0), (20, 0), (20, 12), (0, 12)]),
            ground_z=0.0,
            roof_z=15.0,
            source="synthetic",
        ))

    # Step 2: Build Gmsh geometry and export to OpenFOAM
    print(f"\n[2/4] Building Gmsh geometry (meshsize={args.mesh_size} m)...")
    builder = VectorGmshBuilder(urban, terrain, mesh_size=args.mesh_size)
    builder.build(margin=50.0, bottom_offset=5.0)
    builder.assign_patches()
    builder.build_mesh(mesh_size=args.mesh_size)

    print(f"\n[3/4] Exporting to OpenFOAM polyMesh...")
    builder.export_openfoam(case_dir)
    builder.finalize()

    # Step 4: Set up OpenFOAM case files
    print(f"\n[4/4] Setting up OpenFOAM case files...")
    from foampilot.solver import Solver

    solver = Solver(case_dir)
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = "kEpsilon"
    solver.transient = False

    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = 1.0
    solver.system.controlDict.deltaT = 1.0
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 1
    solver.system.controlDict.purgeWrite = 1

    solver.system.ensure_decomposeParDict(args.nb_proc)
    solver.system.write()
    solver.constant.write()

    solver.boundary.initialize_boundary()
    solver.boundary.set_raw_condition("inlet", "U", {
        "type": "fixedValue",
        "value": "uniform (10 0 0)",
    })
    solver.boundary.set_raw_condition("outlet", "U", {
        "type": "pressureInletOutletVelocity",
        "value": "uniform (0 0 0)",
    })
    solver.boundary.apply_condition_with_wildcard("side.*", "noFrictionWall")
    solver.boundary.set_condition("top", "symmetry")
    solver.boundary.set_condition("ground", "noFrictionWall")
    solver.boundary.write_boundary_conditions(internal_field_overrides={"U": "uniform (10 0 0)"})

    # Summary
    print("\n" + "=" * 60)
    print("Case generated successfully!")
    print("=" * 60)
    print(f"Case directory: {case_dir}")
    print(f"Buildings: {urban.building_count()}")
    print(f"Terrain points: {len(terrain.points)}")
    print("\nFiles written:")
    for path in sorted(case_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(case_dir)}")

    print("\nNext steps:")
    print(f"  checkMesh -case {case_dir}")
    print(f"  blockMesh -case {case_dir}")
    print(f"  snappyHexMesh -overwrite -case {case_dir}")


if __name__ == "__main__":
    main()
