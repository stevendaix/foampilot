#!/usr/bin/env python3
"""
Phase 3 — Real neighborhood from OpenStreetMap via osmnx.

Downloads building footprints from OSM for a given place name,
builds an UrbanModel, and generates a CFD case with foampilot.urban.

Usage:
    PYTHONPATH=../../src python3 osm_neighborhood_example.py \
        --place "Paris, France" \
        --distance 500 \
        --direction 270 \
        --speed 10.0 \
        --z0 0.3 \
        --output cases/osm_neighborhood
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot import Meshing, FluidMechanics, ValueWithUnit
from foampilot.solver import Solver
from foampilot.urban import (
    Building,
    UrbanModel,
    CFDDomain,
    WindFrame,
    CFDLOD,
    CFDSimplifier,
    GmshQuarterBuilder,
    MeshConfig,
)
from foampilot.urban.model.terrain import CFDTerrain
from shapely.geometry import Polygon


def download_osm_buildings(place: str, tags: dict, distance: int):
    """Download building footprints from OSM using foampilot OSMReader."""
    from foampilot.urban import OSMReader

    reader = OSMReader(distance=distance, tags=tags)
    urban = reader.read(place)
    print(f"  Found {urban.building_count()} buildings")
    return urban


def load_terrain(terrain_path: str = None) -> CFDTerrain:
    """Load terrain from GeoTIFF or return flat terrain."""
    if terrain_path is None:
        return CFDTerrain.flat(z=0.0)
    
    path = Path(terrain_path)
    if not path.exists():
        print(f"  Warning: terrain file not found: {terrain_path}")
        return CFDTerrain.flat(z=0.0)
    
    if path.suffix.lower() in (".tif", ".tiff"):
        print(f"  Loading terrain from GeoTIFF: {terrain_path}")
        return CFDTerrain.from_geotiff(path)
    else:
        print(f"  Warning: unsupported terrain format: {path.suffix}")
        return CFDTerrain.flat(z=0.0)


def simplify_buildings(urban: UrbanModel, max_vertices: int = 8) -> UrbanModel:
    """Simplify building footprints to reduce Gmsh meshing complexity.
    
    Converts all buildings to axis-aligned bounding boxes for robust meshing.
    """
    from shapely.geometry import Polygon
    
    simplified = UrbanModel()
    count = 0
    
    for b in urban.buildings():
        footprint = b.footprint
        
        # Use axis-aligned bounding box - most robust for Gmsh
        if footprint.geom_type == "Polygon" and footprint.area > 0:
            minx, miny, maxx, maxy = footprint.bounds
            footprint = Polygon([
                (minx, miny),
                (maxx, miny),
                (maxx, maxy),
                (minx, maxy),
            ])
        
        # Ensure valid geometry
        if not footprint.is_valid:
            try:
                footprint = footprint.buffer(0)
            except Exception:
                continue
        
        if footprint.is_empty or footprint.area < 1.0:
            continue
        
        simplified.add_building(Building(
            id=b.id,
            footprint=footprint,
            ground_z=b.ground_z,
            roof_z=b.roof_z,
            source=b.source,
            confidence=b.confidence,
            attributes=b.attributes,
        ))
        count += 1
    
    print(f"  Simplified {count} buildings")
    return simplified


def generate_case(
    urban: UrbanModel,
    direction_deg: float,
    speed: float,
    z0: float,
    z_ref: float,
    intensity: float,
    turbulence_model: str,
    output_dir: Path,
    terrain: CFDTerrain = None,
    nb_proc: int = 2,
    no_run: bool = False,
    sigfpe: bool = False,
):
    """Generate a single CFD case from an UrbanModel."""
    case_name = f"wind_{int(direction_deg)}deg"
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Generating case: {case_name}")
    print(f"  Wind direction: {direction_deg}°")
    print(f"  Speed (10 m): {speed:.2f} m/s")
    print(f"{'=' * 60}")

    wind_frame = WindFrame(direction_deg=direction_deg, origin=urban.center_xy())
    domain = CFDDomain(
        upstream=8.0,
        downstream=15.0,
        lateral=4.0,
        top=2.5,
        extent_units="href",
        reference_height_method="Hmax",
    )

    geometry = CFDSimplifier(urban, lod=CFDLOD.LOD1).simplify(
        wind_frame=wind_frame,
        domain=domain,
        terrain=terrain,
    )

    print(f"  Domain box: {geometry.domain_box}")
    print(f"  Buildings: {len(geometry.buildings)}")

    n_buildings = len(geometry.buildings)
    if n_buildings <= 5:
        global_size, building_size, max_size = 10.0, 2.0, 40.0
    elif n_buildings <= 20:
        global_size, building_size, max_size = 10.0, 2.0, 40.0
    elif n_buildings <= 50:
        global_size, building_size, max_size = 10.0, 2.0, 40.0
    else:
        global_size, building_size, max_size = 15.0, 3.0, 60.0

    print(f"  Mesh sizing: global={global_size}, building={building_size}, max={max_size}")

    builder = GmshQuarterBuilder(case_dir, geometry)
    builder.build()
    builder.assign_patches()
    builder.build_mesh(MeshConfig(
        global_size=global_size,
        building_size=building_size,
        max_size=max_size,
        min_size=global_size,
        ground_size=3.0,
        algorithm_3d=1,
    ))
    builder.export_openfoam()

    xmin, ymin, zmin, xmax, ymax, zmax = geometry.domain_box
    Dx = xmax - xmin
    Dy = ymax - ymin
    Dz = zmax - zmin
    building_heights = [b.height for b in urban.buildings()]

    # --- Patch types: walls → wall, sides → noFrictionWall, outlet → pressureOutlet ---
    boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
    content = boundary_file.read_text()
    import re
    wall_patches = ["ground", "side_left", "side_right",
                    "buildings"]
    for patch_name in wall_patches:
        pattern = rf'({patch_name}\s*\{{\s*type\s+)patch(;)'
        content = re.sub(pattern, r'\1wall\2', content)

    content = re.sub(
        r'(outlet\s*\{\s*type\s+)patch(;)',
        r'\1pressureOutlet\2',
        content
    )
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

    # --- Solver setup ---
    available_fluids = FluidMechanics.get_available_fluids()
    fluid_mech = FluidMechanics(
        available_fluids["Air"],
        temperature=ValueWithUnit(293.15, "K"),
        pressure=ValueWithUnit(101325, "Pa"),
    )
    fluid_props = fluid_mech.get_fluid_properties()
    nu = fluid_props["kinematic_viscosity"]

    solver = Solver(case_dir)
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = turbulence_model
    solver.transient = False

    solver.constant.transportProperties.nu = nu

    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.deltaT = 1.0
    solver.system.controlDict.endTime = 2000
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 1
    solver.system.controlDict.purgeWrite = 1

    solver.system.fvSolution.SIMPLE["nNonOrthogonalCorrectors"] = "2"
    solver.system.fvSolution.SIMPLE["pRefCell"] = "0"
    solver.system.fvSolution.SIMPLE["pRefValue"] = "0"
    solver.system.fvSolution.SIMPLE["residualControl"] = {
        "p": "1e-4",
        "U": "1e-4",
        "(k|epsilon|omega)": "1e-4",
    }
    solver.system.fvSolution.relaxationFactors = {
        "fields": {"p": "0.3"},
        "equations": {"U": "0.7", "(k|epsilon|omega).*": "0.7"},
    }

    solver.system.ensure_decomposeParDict(nb_proc)
    solver.system.write()
    solver.constant.write()

    # --- Boundary conditions ---
    solver.boundary.initialize_boundary()

    from wind_profile import friction_velocity, KAPPA
    u_star = friction_velocity(speed, z0, z_ref)

    u_code = (
        "const vectorField& cf = this->patch().Cf();\n"
        "vectorField vel(cf.size());\n"
        "forAll(cf, i)\n"
        "{\n"
        "    scalar z = cf[i].z();\n"
        f"    scalar z0 = {z0};\n"
        f"    scalar u_ref = {speed};\n"
        f"    scalar z_ref = {z_ref};\n"
        f"    scalar kappa = {KAPPA};\n"
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
        f"    scalar kappa = {KAPPA};\n"
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

    if turbulence_model == "kEpsilon":
        eps_code = (
            "const vectorField& cf = this->patch().Cf();\n"
            "scalarField eps(cf.size());\n"
            "forAll(cf, i)\n"
            "{\n"
            "    scalar z = cf[i].z();\n"
            f"    scalar z0 = {z0};\n"
            f"    scalar u_ref = {speed};\n"
            f"    scalar z_ref = {z_ref};\n"
            f"    scalar kappa = {KAPPA};\n"
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

    if turbulence_model == "kOmegaSST":
        omega_code = (
            "const vectorField& cf = this->patch().Cf();\n"
            "scalarField om(cf.size());\n"
            "forAll(cf, i)\n"
            "{\n"
            "    scalar z = cf[i].z();\n"
            f"    scalar z0 = {z0};\n"
            f"    scalar u_ref = {speed};\n"
            f"    scalar z_ref = {z_ref};\n"
            f"    scalar kappa = {KAPPA};\n"
            "    scalar u_star = u_ref * kappa / Foam::log(z_ref / z0);\n"
            "    om[i] = u_star / (kappa * Foam::max(z, z0));\n"
            "}\n"
            "operator==(om);"
        )
        solver.boundary.set_raw_condition("inlet", "omega", {
            "type": "codedFixedValue",
            "name": "inletOmegaProfile",
            "value": "uniform 0",
            "code": "#{\n" + omega_code + "\n#};",
        })

    solver.boundary.set_raw_condition("inlet", "nut", {"type": "zeroGradient"})

    solver.boundary.set_condition("top", "symmetry")

    solver.boundary.set_condition("side_left", "noFrictionWall")
    solver.boundary.set_condition("side_right", "noFrictionWall")

    solver.boundary.write_boundary_conditions(
        internal_field_overrides={"U": f"uniform ({speed} 0 0)"}
    )

    metadata = {
        "direction_deg": direction_deg,
        "speed_10m": speed,
        "z0": z0,
        "z_ref": z_ref,
        "intensity": intensity,
        "rotation_angle": wind_frame.rotation_angle,
        "u_star": u_star,
        "nu": float(nu.get_in("m^2/s") if hasattr(nu, 'get_in') else nu),
        "turbulence_model": turbulence_model,
        "n_buildings": len(urban.buildings()),
        "building_heights": building_heights,
        "domain_dims": {"Dx": Dx, "Dy": Dy, "Dz": Dz, "xmin": xmin, "ymin": ymin, "zmin": zmin},
        "terrain_source": terrain.source if terrain else None,
    }
    with open(case_dir / "case_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Case generated: {case_dir}")
    return case_dir


def main():
    parser = argparse.ArgumentParser(description="OSM neighborhood CFD example")
    parser.add_argument("--place", required=True, help="Place name (e.g. 'Paris, France')")
    parser.add_argument("--distance", type=int, default=100, help="Search radius in meters")
    parser.add_argument("--direction", type=float, default=270.0, help="Wind direction in degrees")
    parser.add_argument("--speed", type=float, default=10.0, help="Wind speed at reference height (m/s)")
    parser.add_argument("--z0", type=float, default=0.3, help="Surface roughness length (m)")
    parser.add_argument("--z-ref", type=float, default=10.0, help="Reference height (m)")
    parser.add_argument("--intensity", type=float, default=0.1, help="Turbulence intensity")
    parser.add_argument("--turbulence-model", default="kEpsilon",
                        choices=["kEpsilon", "kOmegaSST", "SpalartAllmaras"],
                        help="Turbulence model")
    parser.add_argument("--terrain", default=None, help="Path to terrain GeoTIFF (optional)")
    parser.add_argument("--nb-proc", type=int, default=2, help="Number of parallel processes")
    parser.add_argument("--output", default="cases/osm_neighborhood", help="Output directory")
    parser.add_argument("--no-run", action="store_true", help="Only generate, don't run")
    parser.add_argument("--sigfpe", action="store_true", help="Enable FOAM_SIGFPE")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    urban = download_osm_buildings(
        place=args.place,
        tags={"building": True},
        distance=args.distance,
    )

    urban = simplify_buildings(urban)

    if urban.building_count() == 0:
        print("No buildings found. Try a larger --distance or a different place.")
        sys.exit(1)

    terrain = load_terrain(args.terrain)

    case_dir = generate_case(
        urban=urban,
        direction_deg=args.direction,
        speed=args.speed,
        z0=args.z0,
        z_ref=args.z_ref,
        intensity=args.intensity,
        turbulence_model=args.turbulence_model,
        output_dir=output_dir,
        terrain=terrain,
        nb_proc=args.nb_proc,
        no_run=args.no_run,
        sigfpe=args.sigfpe,
    )

    urban.to_geojson(output_dir / "urban_model.geojson")
    print(f"\nUrbanModel saved to: {output_dir / 'urban_model.geojson'}")
    print(f"Run: checkMesh -case {case_dir}")


if __name__ == "__main__":
    main()
