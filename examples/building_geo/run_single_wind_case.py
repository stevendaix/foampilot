#!/usr/bin/env python3
"""
Generate and run a single wind direction CFD case.

Usage:
    PYTHONPATH=../../src python3 run_single_wind_case.py \
        --direction 270 \
        --speed 10.0 \
        --z0 0.3 \
        --nb-proc 4
"""

import argparse
import json
import sys
import time
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
from shapely.geometry import Polygon


def create_default_urban(n_buildings: int = 3, seed: int = 42) -> UrbanModel:
    """Create a default urban model with a few buildings or a larger grid."""
    import random

    random.seed(seed)
    urban = UrbanModel()

    if n_buildings <= 3:
        urban.add_building(Building(
            id="B1",
            footprint=Polygon([(0, 0), (40, 0), (40, 20), (0, 20)]),
            ground_z=0.0,
            roof_z=15.0,
        ))
        urban.add_building(Building(
            id="B2",
            footprint=Polygon([(50, 10), (80, 10), (80, 30), (50, 30)]),
            ground_z=0.0,
            roof_z=22.0,
        ))
        if n_buildings >= 3:
            urban.add_building(Building(
                id="B3",
                footprint=Polygon([(20, 40), (60, 40), (60, 70), (20, 70)]),
                ground_z=0.0,
                roof_z=18.0,
            ))
        return urban

    grid_size = int(n_buildings ** 0.5)
    spacing = 30.0
    building_id = 0

    for ix in range(grid_size):
        for iy in range(grid_size):
            if building_id >= n_buildings:
                break
            x = ix * spacing + random.uniform(-2, 2)
            y = iy * spacing + random.uniform(-2, 2)
            w = random.uniform(10, 20)
            d = random.uniform(10, 20)
            h = random.uniform(10, 30)
            footprint = Polygon([
                (x - w/2, y - d/2),
                (x + w/2, y - d/2),
                (x + w/2, y + d/2),
                (x - w/2, y + d/2),
            ])
            urban.add_building(Building(
                id=f"B{building_id+1:04d}",
                footprint=footprint,
                ground_z=0.0,
                roof_z=h,
            ))
            building_id += 1

    return urban


def generate_case(urban: UrbanModel, direction_deg: float, speed: float,
                  output_dir: Path, z0: float = 0.3, z_ref: float = 10.0,
                  intensity: float = 0.1, turbulence_model: str = "kEpsilon"):
    """Generate a single CFD case for a given wind direction and speed."""
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
        ground_size=3.0,
        algorithm_3d=4,
    ))
    builder.export_openfoam()

    xmin, ymin, zmin, xmax, ymax, zmax = geometry.domain_box
    Dx = xmax - xmin
    Dy = ymax - ymin
    Dz = zmax - zmin
    building_heights = [b.height for b in urban.buildings()]

    # --- Patch types: INLET/OUTLET → patch, walls → wall ---
    boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
    content = boundary_file.read_text()
    import re
    wall_patches = ["ground", "side_left", "side_right",
                    "buildings"]
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

    solver.system.ensure_decomposeParDict(4)
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

    solver.boundary.set_raw_condition("inlet", "nut", {"type": "zeroGradient", "value": "uniform 1e-5"})

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
        "u_star": u_star,
        "nu": float(nu.get_in("m^2/s") if hasattr(nu, 'get_in') else nu),
        "turbulence_model": turbulence_model,
        "n_buildings": len(urban.buildings()),
        "building_heights": building_heights,
        "domain_dims": {"Dx": Dx, "Dy": Dy, "Dz": Dz, "xmin": xmin, "ymin": ymin, "zmin": zmin},
    }
    with open(case_dir / "case_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Case generated: {case_dir}")
    return case_dir


def run_case(case_dir: Path, nb_proc: int = 2, check_only: bool = False, sigfpe: bool = False):
    """Run a single CFD case."""
    print(f"\n{'=' * 60}")
    print(f"Running case: {case_dir.name}")
    print(f"{'=' * 60}")

    log_file = case_dir / "log.incompressibleFluid"

    if log_file.exists() and not check_only:
        print("  Log exists — skipping.")
        return True

    if check_only and not log_file.exists():
        print("  No log file — case not yet run.")
        return False

    if not check_only:
        from foampilot.solver import Solver

        metadata = {}
        meta_path = case_dir / "case_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        solver = Solver(case_dir)
        solver.compressible = False
        solver.with_gravity = False
        solver.turbulence_model = metadata.get("turbulence_model", "kEpsilon")
        solver.transient = False

        if sigfpe:
            os.environ["FOAM_SIGFPE"] = "1"

        solver.run_simulation(nb_proc=nb_proc, log_filename="log.incompressibleFluid")

    if log_file.exists():
        print(f"  Log file: {log_file}")
        return True
    else:
        print(f"  Log file not found: {log_file}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate and run a single wind case")
    parser.add_argument("--direction", type=float, required=True, help="Wind direction in degrees")
    parser.add_argument("--speed", type=float, default=10.0, help="Wind speed at reference height (m/s)")
    parser.add_argument("--z0", type=float, default=0.3, help="Surface roughness length (m)")
    parser.add_argument("--z-ref", type=float, default=10.0, help="Reference height (m)")
    parser.add_argument("--intensity", type=float, default=0.1, help="Turbulence intensity")
    parser.add_argument("--turbulence-model", default="kEpsilon",
                        choices=["kEpsilon", "kOmegaSST", "SpalartAllmaras"])
    parser.add_argument("--nb-proc", type=int, default=2, help="Number of parallel processes")
    parser.add_argument("--n-buildings", type=int, default=3, help="Number of buildings in the synthetic neighborhood")
    parser.add_argument("--output", default="cases", help="Output directory for cases")
    parser.add_argument("--no-run", action="store_true", help="Only generate, don't run")
    parser.add_argument("--sigfpe", action="store_true", help="Enable FOAM_SIGFPE")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    urban = create_default_urban(n_buildings=args.n_buildings)

    case_dir = generate_case(
        urban=urban,
        direction_deg=args.direction,
        speed=args.speed,
        output_dir=output_dir,
        z0=args.z0,
        z_ref=args.z_ref,
        intensity=args.intensity,
        turbulence_model=args.turbulence_model,
    )

    if not args.no_run:
        run_case(case_dir, nb_proc=args.nb_proc, sigfpe=args.sigfpe)
    else:
        print(f"\nCase generated but not run: {case_dir}")
        print(f"To run later: PYTHONPATH=../../src python3 run_all_cases.py --cases-dir {output_dir}")


if __name__ == "__main__":
    main()
