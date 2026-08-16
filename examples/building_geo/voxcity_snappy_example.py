#!/usr/bin/env python3
"""
VoxCity -> OpenFOAM snappyHexMesh example.

Default: synthetic neighborhood (no VoxCity/EE cost).
Use --use-voxcity to fetch real data via VoxCity + Earth Engine.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

from foampilot.urban import (
    Building,
    UrbanModel,
)
from foampilot.urban.model.terrain import CFDTerrain
from foampilot.urban.snappy_config import (
    DomainConfig,
    TerrainConfig,
    BuildingConfig,
    SnappyMeshConfig,
)
from foampilot.openfoam.snappy_case_builder import SnappyCaseBuilder
from foampilot.solver import Solver
from shapely.geometry import Polygon


def build_synthetic_urban() -> tuple[UrbanModel, CFDTerrain]:
    """Build a small synthetic neighborhood with 3 buildings and flat terrain."""
    urban = UrbanModel()
    urban.add_building(Building(
        id="B1",
        footprint=Polygon([(0, 0), (20, 0), (20, 12), (0, 12)]),
        ground_z=0.0,
        roof_z=15.0,
        source="synthetic",
        confidence=1.0,
    ))
    urban.add_building(Building(
        id="B2",
        footprint=Polygon([(28, 5), (45, 5), (45, 18), (28, 18)]),
        ground_z=0.0,
        roof_z=22.0,
        source="synthetic",
        confidence=1.0,
    ))
    urban.add_building(Building(
        id="B3",
        footprint=Polygon([(10, 25), (30, 25), (30, 40), (10, 40)]),
        ground_z=0.0,
        roof_z=10.0,
        source="synthetic",
        confidence=1.0,
    ))
    terrain = CFDTerrain.flat(z=0.0)
    return urban, terrain


def build_from_voxcity(rectangle_vertices, meshsize=5.0):
    """Read from VoxCity/EE and convert to UrbanModel + CFDTerrain."""
    try:
        import ee
        from voxcity.generator import get_voxcity
        from foampilot.urban.readers.voxcity_reader import VoxCityReader
    except ImportError as exc:
        raise RuntimeError(
            "VoxCity and Google Earth Engine are required for --use-voxcity. "
            "Install them with: pip install voxcity && earthengine authenticate"
        ) from exc

    try:
        ee.Initialize(project="openfoam-project")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project="openfoam-project")

    reader = VoxCityReader(meshsize=meshsize)
    urban, terrain = reader.read(rectangle_vertices)
    if urban.building_count() == 0:
        raise RuntimeError("VoxCity returned 0 buildings for the requested area.")
    return urban, terrain


def main():
    parser = argparse.ArgumentParser(description="VoxCity -> snappyHexMesh OpenFOAM case")
    parser.add_argument("--output", default="cases/voxcity_snappy_demo", help="Output case directory")
    parser.add_argument("--mesh-only", action="store_true", help="Only write mesh files, skip solver")
    parser.add_argument("--run-mesh", action="store_true", help="Run blockMesh + snappyHexMesh")
    parser.add_argument("--nb-proc", type=int, default=1, help="Number of processes for decomposePar")
    parser.add_argument("--use-voxcity", action="store_true", help="Use VoxCity/EE instead of synthetic data")
    parser.add_argument("--voxcity-meshsize", type=float, default=5.0, help="VoxCity mesh size in meters")
    parser.add_argument(
        "--rectangle-vertices",
        nargs="+",
        type=float,
        default=None,
        help="Lon/lat vertices for VoxCity, e.g. 2.3522 48.8566 2.3522 48.8576 ...",
    )
    args = parser.parse_args()

    if args.use_voxcity:
        if args.rectangle_vertices is None or len(args.rectangle_vertices) < 6:
            parser.error("--use-voxcity requires --rectangle-vertices with at least 3 lon/lat pairs")

        coords = args.rectangle_vertices
        rectangle_vertices = [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]
        print("Using VoxCity/EE (this may take time and may incur data/usage costs)...")
        urban, terrain = build_from_voxcity(rectangle_vertices, meshsize=args.voxcity_meshsize)
    else:
        print("Using synthetic urban model (no VoxCity/EE cost).")
        urban, terrain = build_synthetic_urban()

    case_dir = Path(args.output)
    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Buildings: {urban.building_count()}")
    print(f"  Terrain: {terrain.source}")

    domain_config = DomainConfig(
        margin_x=50.0,
        margin_y=50.0,
        top_margin=50.0,
        bottom_margin=10.0,
    )
    terrain_config = TerrainConfig(
        dem_resolution=2.0,
        horizontal_extension=20.0,
        bottom_offset=5.0,
    )
    building_config = BuildingConfig(
        min_area=1.0,
        default_height=10.0,
        foundation_depth=0.5,
    )
    mesh_config = SnappyMeshConfig(
        base_cell_size=4.0,
        terrain_refinement_level=2,
        building_refinement_level=3,
        n_cells_between_walls=3,
        max_global_cells=2_000_000,
        add_layers=False,
    )

    solver = None
    if not args.mesh_only:
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

    print("Writing STL and OpenFOAM dictionaries...")
    builder.write(run_mesh=args.run_mesh)

    print(f"\nCase generated: {case_dir}")
    print("Files:")
    for path in sorted(case_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(case_dir)}")
    print("\nNext steps:")
    print(f"  checkMesh -case {case_dir}")
    if args.run_mesh:
        print("  snappyHexMesh -overwrite -case", case_dir)
    else:
        print("  blockMesh -case", case_dir)
        print("  surfaceFeatures -case", case_dir)
        print("  snappyHexMesh -overwrite -case", case_dir)


if __name__ == "__main__":
    main()
