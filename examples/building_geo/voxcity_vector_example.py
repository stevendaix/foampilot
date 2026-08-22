#!/usr/bin/env python3
"""
VoxCity -> OpenFOAM example using the vector Gmsh path (no STL).

This script demonstrates the complete pipeline:
   1. Read VoxCity data (real or synthetic fallback)
   2. Build Gmsh geometry directly from vectors (building_gdf + DEM)
   3. Export to OpenFOAM polyMesh via DirectOpenFOAMExporter
   4. Set up OpenFOAM case with boundary conditions following generate_wind_cases.py
   5. Validate with checkMesh

Usage:
    # Synthetic data (no VoxCity/EE cost)
    PYTHONPATH=../../src python3 voxcity_vector_example.py \
        --output cases/voxcity_vector_demo

    # Real VoxCity data (requires EE auth)
    PYTHONPATH=../../src python3 voxcity_vector_example.py \
        --use-voxcity \
        --rectangle-vertices 2.3225 48.8515 2.3225 48.8528 2.3240 48.8528 2.3240 48.8515 \
        --output cases/voxcity_vector_demo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "voxcity_export_work" / "src"))

from foampilot import FluidMechanics, ValueWithUnit
from foampilot.urban import Building, UrbanModel
from foampilot.urban.model.terrain import CFDTerrain
from foampilot.urban.snappy_config import DomainConfig, TerrainConfig, BuildingConfig, SnappyMeshConfig
from foampilot.solver import Solver
from foampilot.urban.readers.voxcity_reader import VoxCityReader
from vector_builder import VectorGmshBuilder
from wind_profile import KAPPA, Z_REF
from shapely.geometry import Polygon
import re


def build_synthetic_urban():
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


def get_air_properties():
    available_fluids = FluidMechanics.get_available_fluids()
    fluid_mech = FluidMechanics(
        available_fluids["Air"],
        temperature=ValueWithUnit(293.15, "K"),
        pressure=ValueWithUnit(101325, "Pa"),
    )
    return fluid_mech.get_fluid_properties()


def setup_openfoam_case(case_dir: Path, nb_proc: int = 1):
    """Set up OpenFOAM case files following generate_wind_cases.py reference."""
    fluid_props = get_air_properties()
    nu = fluid_props["kinematic_viscosity"]

    solver = Solver(case_dir)
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = "kEpsilon"
    solver.transient = False

    solver.constant.transportProperties.nu = nu

    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = 2000
    solver.system.controlDict.deltaT = 1.0
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 1
    solver.system.controlDict.purgeWrite = 1

    solver.system.fvSolution.SIMPLE["nNonOrthogonalCorrectors"] = "3"
    solver.system.fvSolution.SIMPLE["pRefCell"] = "0"
    solver.system.fvSolution.SIMPLE["pRefValue"] = "0"
    solver.system.fvSolution.SIMPLE["residualControl"] = {
        "p": "1e-4",
        "U": "1e-4",
        "(k|epsilon|omega)": "1e-4",
    }
    solver.system.fvSolution.relaxationFactors = {
        "fields": {"p": "0.2"},
        "equations": {"U": "0.5", "(k|epsilon|omega).*": "0.5"},
    }

    solver.system.ensure_decomposeParDict(nb_proc)
    solver.system.write()
    solver.constant.write()

    boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
    content = boundary_file.read_text()
    wall_patches = ["ground", "side_left", "side_right", "buildings"]
    for patch_name in wall_patches:
        pattern = rf'({patch_name}\s*\{{\s*type\s+)patch(;)'
        content = re.sub(pattern, r'\1wall\2', content)

    content = re.sub(
        r'(top\s*\{\s*type\s+)patch(;)',
        r'\1symmetry\2',
        content
    )
    boundary_file.write_text(content)

    slip_fields = {
        "U": "slip",
        "p": "zeroGradient",
        "k": "zeroGradient",
        "epsilon": "zeroGradient",
        "nut": "zeroGradient",
    }
    for field_file in (case_dir / "0").glob("*"):
        if not field_file.is_file():
            continue
        field_name = field_file.name
        if field_name not in slip_fields:
            continue
        field_content = field_file.read_text()
        field_content = re.sub(
            r'("top"\s*\{\s*type\s+)\w+(;\s*[^}]*\})',
            r'\1' + slip_fields[field_name] + r'\2',
            field_content
        )
        field_file.write_text(field_content)

    solver.boundary.initialize_boundary()

    z0 = 0.3
    z_ref = Z_REF
    speed = 10.0
    intensity = 0.1
    kappa = KAPPA

    u_code = (
        "const vectorField& cf = this->patch().Cf();\n"
        "vectorField vel(cf.size());\n"
        "forAll(cf, i)\n"
        "{\n"
        "    scalar z = cf[i].z();\n"
        f"    scalar z0 = {z0};\n"
        f"    scalar u_ref = {speed};\n"
        f"    scalar z_ref = {z_ref};\n"
        f"    scalar kappa = {kappa};\n"
        "    scalar u_star = u_ref * kappa / Foam::log(z_ref / z0);\n"
        "    scalar u_mag = u_star / kappa * Foam::log(Foam::max(z / z0, 1.0 + SMALL));\n"
        "    vel[i] = vector(u_mag, 0, 0);\n"
        "}\n"
        "operator==(vel);"
    )

    solver.boundary.set_raw_condition("inlet", "U", {
        "type": "codedFixedValue",
        "name": "inletVelocityProfile",
        "value": "uniform (0 0 0)",
        "code": "#{\n" + u_code + "\n#};",
    })

    k_code = (
        "const vectorField& cf = this->patch().Cf();\n"
        "scalarField kval(cf.size());\n"
        "forAll(cf, i)\n"
        "{\n"
        "    scalar z = cf[i].z();\n"
        f"    scalar z0 = {z0};\n"
        f"    scalar u_ref = {speed};\n"
        f"    scalar z_ref = {z_ref};\n"
        f"    scalar kappa = {kappa};\n"
        f"    scalar I = {intensity};\n"
        "    scalar u_star = u_ref * kappa / Foam::log(z_ref / z0);\n"
        "    scalar u_mag = u_star / kappa * Foam::log(Foam::max(z / z0, 1.0 + SMALL));\n"
        "    kval[i] = 1.5 * pow(I * u_mag, 2);\n"
        "}\n"
        "operator==(kval);"
    )

    solver.boundary.set_raw_condition("inlet", "k", {
        "type": "codedFixedValue",
        "name": "inletTkeProfile",
        "value": f"uniform {1.5 * (intensity * speed) ** 2}",
        "code": "#{\n" + k_code + "\n#};",
    })

    eps_code = (
        "const vectorField& cf = this->patch().Cf();\n"
        "scalarField eps(cf.size());\n"
        "forAll(cf, i)\n"
        "{\n"
        "    scalar z = cf[i].z();\n"
        f"    scalar z0 = {z0};\n"
        f"    scalar u_ref = {speed};\n"
        f"    scalar z_ref = {z_ref};\n"
        f"    scalar kappa = {kappa};\n"
        "    scalar u_star = u_ref * kappa / Foam::log(z_ref / z0);\n"
        "    eps[i] = pow(u_star, 3) / (kappa * Foam::max(z, z0));\n"
        "}\n"
        "operator==(eps);"
    )
    solver.boundary.set_raw_condition("inlet", "epsilon", {
        "type": "codedFixedValue",
        "name": "inletEpsilonProfile",
        "value": "uniform 0.1",
        "code": "#{\n" + eps_code + "\n#};",
    })

    solver.boundary.apply_condition_with_wildcard(
        pattern="outlet",
        condition_type="pressureOutlet"
    )

    solver.boundary.set_raw_condition("inlet", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("inlet", "nut", {"type": "zeroGradient"})

    solver.boundary.set_condition("top", "symmetry")
    solver.boundary.set_condition("side_left", "noFrictionWall")
    solver.boundary.set_condition("side_right", "noFrictionWall")

    solver.boundary.write_boundary_conditions(
        internal_field_overrides={"U": f"uniform ({speed} 0 0)"}
    )

    return solver


def main():
    parser = argparse.ArgumentParser(description="VoxCity -> vector Gmsh -> OpenFOAM example")
    parser.add_argument("--output", default="cases/voxcity_vector_demo", help="Output case directory")
    parser.add_argument("--mesh-size", type=float, default=5.0, help="Mesh size for Gmsh")
    parser.add_argument("--nb-proc", type=int, default=1, help="Number of processes")
    parser.add_argument("--use-voxcity", action="store_true", help="Use VoxCity/EE instead of synthetic")
    parser.add_argument("--voxcity-meshsize", type=float, default=5.0, help="VoxCity mesh size")
    parser.add_argument(
        "--rectangle-vertices",
        nargs="+",
        type=float,
        default=None,
        help="Lon/lat vertices for VoxCity",
    )
    args = parser.parse_args()

    case_dir = Path(args.output)
    case_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VoxCity -> vector Gmsh -> OpenFOAM example")
    print("=" * 60)

    # Step 1: Get urban model
    if args.use_voxcity:
        if args.rectangle_vertices is None or len(args.rectangle_vertices) < 6:
            print("ERROR: --use-voxcity requires --rectangle-vertices")
            sys.exit(1)

        coords = args.rectangle_vertices
        rectangle_vertices = [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]
        print(f"\n[1/4] Downloading VoxCity model (meshsize={args.voxcity_meshsize} m)...")
        urban, terrain = build_from_voxcity(rectangle_vertices, meshsize=args.voxcity_meshsize)
    else:
        print("\n[1/4] Building synthetic urban model...")
        urban, terrain = build_synthetic_urban()

    print(f"  Buildings: {urban.building_count()}")
    print(f"  Terrain: {terrain.source}")

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
    solver = setup_openfoam_case(case_dir, nb_proc=args.nb_proc)

    # Run simulation using foampilot
    print(f"\n[5/5] Running simulation...")
    solver.run_simulation(nb_proc=args.nb_proc)

    # Summary
    print("\n" + "=" * 60)
    print("Case generated successfully!")
    print("=" * 60)
    print(f"Case directory: {case_dir}")
    print(f"Buildings: {urban.building_count()}")
    print(f"Terrain: {terrain.source}")
    print("\nFiles written:")
    for path in sorted(case_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(case_dir)}")

    print("\nNext steps:")
    print(f"  checkMesh -case {case_dir}")
    print(f"  Or rerun simulation: python3 {Path(__file__).name} --output {case_dir}")


if __name__ == "__main__":
    main()
