#!/usr/bin/env python3
"""
Test VoxCity -> OpenFOAM snappyHexMesh on a small residential area.

This script:
  1. Downloads a VoxCity model for a small residential area via Earth Engine.
  2. Converts it to UrbanModel + CFDTerrain.
  3. Generates terrain.stl and buildings.stl.
  4. Writes an OpenFOAM case ready for blockMesh + snappyHexMesh.

WARNING: This uses Google Earth Engine and may incur data/usage costs.
The downloaded model is cached locally to avoid repeated downloads.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.urban import Building, UrbanModel
from foampilot.urban.model.terrain import CFDTerrain
from foampilot.urban.snappy_config import (
    DomainConfig,
    TerrainConfig,
    BuildingConfig,
    SnappyMeshConfig,
)
from foampilot.openfoam.snappy_case_builder import SnappyCaseBuilder
from foampilot.solver import Solver
from foampilot.urban.readers.voxcity_reader import VoxCityReader


# Small residential area in Paris 15e (around rue de Vaugirard)
# ~200m x 150m, low-rise buildings, good for a quick test.
DEFAULT_RECTANGLE_VERTICES = [
    (2.3225, 48.8515),
    (2.3225, 48.8528),
    (2.3240, 48.8528),
    (2.3240, 48.8515),
]

DEFAULT_OUTPUT = Path("/tmp/voxcity_residential_test")


def parse_args():
    parser = argparse.ArgumentParser(description="VoxCity residential test -> snappyHexMesh")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output case directory")
    parser.add_argument("--meshsize", type=float, default=5.0, help="VoxCity mesh size in meters")
    parser.add_argument("--run-mesh", action="store_true", help="Run blockMesh + snappyHexMesh")
    parser.add_argument("--nb-proc", type=int, default=1, help="Number of processes for decomposePar")
    parser.add_argument(
        "--rectangle-vertices",
        nargs="+",
        type=float,
        default=None,
        help="Lon/lat vertices for VoxCity, e.g. 2.3225 48.8515 2.3225 48.8528 ...",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    case_dir = Path(args.output)
    case_dir.mkdir(parents=True, exist_ok=True)

    if args.rectangle_vertices is None or len(args.rectangle_vertices) < 6:
        rectangle_vertices = DEFAULT_RECTANGLE_VERTICES
        print("Using default residential area (Paris 15e).")
    else:
        coords = args.rectangle_vertices
        rectangle_vertices = [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]

    print("=" * 60)
    print("VoxCity residential test")
    print("=" * 60)
    print(f"Output case : {case_dir}")
    print(f"Mesh size   : {args.meshsize} m")
    print(f"Rectangle   : {rectangle_vertices}")
    print("WARNING: This will use Google Earth Engine and may incur costs.")
    print("=" * 60)

    print("\n[1/4] Initializing Earth Engine...")
    try:
        import ee
        ee.Initialize(project="openfoam-project")
        print("  EE initialized.")
    except Exception as exc:
        print(f"  EE init failed: {exc}")
        print("  Run: ee.Authenticate()")
        sys.exit(1)

    print("\n[2/4] Downloading VoxCity model...")
    reader = VoxCityReader(meshsize=args.meshsize)
    urban, terrain = reader.read(rectangle_vertices)
    print(f"  Buildings : {urban.building_count()}")
    print(f"  Terrain   : {terrain.source}")

    if urban.building_count() == 0:
        print("ERROR: VoxCity returned 0 buildings. Try a larger area or different location.")
        sys.exit(1)

    print("\n[3/4] Building OpenFOAM case...")
    domain_config = DomainConfig(
        margin_x=30.0,
        margin_y=30.0,
        top_margin=30.0,
        bottom_margin=5.0,
    )
    terrain_config = TerrainConfig(
        dem_resolution=2.0,
        horizontal_extension=10.0,
        bottom_offset=3.0,
    )
    building_config = BuildingConfig(
        min_area=1.0,
        default_height=9.0,
        foundation_depth=0.5,
    )
    mesh_config = SnappyMeshConfig(
        base_cell_size=3.0,
        terrain_refinement_level=2,
        building_refinement_level=3,
        n_cells_between_walls=3,
        max_global_cells=2_000_000,
        add_layers=False,
    )

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
    solver.boundary.set_raw_condition("inlet", "U", {"type": "fixedValue", "value": "uniform (10 0 0)"})
    solver.boundary.set_raw_condition("outlet", "U", {"type": "pressureInletOutletVelocity", "value": "uniform (0 0 0)"})
    solver.boundary.apply_condition_with_wildcard("side.*", "noFrictionWall")
    solver.boundary.set_condition("top", "symmetry")
    solver.boundary.set_condition("ground", "noFrictionWall")
    solver.boundary.write_boundary_conditions(internal_field_overrides={"U": "uniform (10 0 0)"})

    builder = SnappyCaseBuilder(
        case_dir=case_dir,
        urban=urban,
        terrain=terrain,
        solver=solver,
        domain_config=domain_config,
        terrain_config=terrain_config,
        building_config=building_config,
        mesh_config=mesh_config,
    )

    print("  Writing STL and dictionaries...")
    builder.write(run_mesh=False)

    print("\n[4/4] Summary")
    print(f"  Case directory : {case_dir}")
    print(f"  Buildings      : {urban.building_count()}")
    print(f"  Terrain source : {terrain.source}")
    print("  Files written:")
    for path in sorted(case_dir.rglob("*")):
        if path.is_file():
            print(f"    {path.relative_to(case_dir)}")
    print("\nNext steps:")
    print(f"  checkMesh -case {case_dir}")
    print(f"  blockMesh -case {case_dir}")
    print(f"  surfaceFeatures -case {case_dir}")
    print(f"  snappyHexMesh -overwrite -case {case_dir}")


if __name__ == "__main__":
    main()
