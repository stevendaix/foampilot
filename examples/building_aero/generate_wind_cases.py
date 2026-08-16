#!/usr/bin/env python3
"""
Generate CFD cases for wind rose analysis.

For each wind direction sector in the wind rose (derived from EPW data),
this script:
  1. Creates a Gmsh geometry: axis-aligned fluid domain + rotated buildings.
  2. Meshes with Gmsh and exports to OpenFOAM polyMesh (direct export).
  3. Writes boundary conditions with a logarithmic wind profile at the inlet
     (codedFixedValue for U, k, epsilon).
  4. Configures the solver (simpleFoam, k-omega SST).

The wind rose is read from a JSON file produced by WeatherFileEPW.export_wind_frequencies().

Usage:
    PYTHONPATH=src python3 generate_wind_cases.py \\
        --epw path/to/weather.epw \\
        --directions 0,45,90,135,180,225,270,315  \\
        --z0 0.3 --z-ref 10.0 --intensity 0.1
"""

import argparse
import json
import math
import sys
import numpy as np
import gmsh
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))

from foampilot import Meshing, FluidMechanics, ValueWithUnit
from foampilot.solver import Solver
from foampilot.utilities.epw_weather_reader import WeatherFileEPW
from wind_profile import (
    log_wind_profile,
    friction_velocity,
    turbulence_quantities,
    rotation_angle_for_wind_direction,
    KAPPA,
    Z_REF,
)

DEFAULT_BUILDINGS_CONFIG = {
    "quartier": {
        "lot_width": 150.0,
        "lot_length": 300.0,
        "street_width": 20.0,
        "min_h": 15.0,
        "max_h": 40.0,
        "building_depth": 12.0,
        "gap": 5.0,
        "n_buildings_side": 5,
    },
    "domaine_fluide": {
        "upstream_H": 8.0,
        "downstream_H": 15.0,
        "lateral_H": 4.0,
        "top_H": 2.5,
    },
    "maillage": {
        "lc_min": 5.0,
        "lc_max": 15.0,
    },
    "fluide": {
        "nom": "Air",
        "temperature": 293.15,
        "pression": 101325,
    },
    "seed": 42,
}


def load_buildings_config() -> dict:
    """Load building configuration, using defaults if not found."""
    cfg_path = Path.cwd() / "buildings_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return DEFAULT_BUILDINGS_CONFIG


def create_wind_rose(epw_path: Path, direction_bin: float = 22.5,
                     speed_bins=None) -> dict:
    """Read EPW and compute wind frequency table.

    Returns dict: {direction_deg: [{"speed": float, "frequency": float}]}
    """
    if speed_bins is None:
        speed_bins = [0, 2, 4, 6, 8, 10, 15, 25]

    epw = WeatherFileEPW()
    epw.read(str(epw_path))
    freq = epw.compute_wind_frequencies(
        direction_bin=direction_bin, speed_bins=speed_bins
    )
    return {float(k): v for k, v in freq.items()}


def get_dominant_directions(wind_rose: dict, min_frequency: float = 0.02) -> list:
    """Extract dominant wind directions (sectors with frequency > min_frequency).

    Returns list of (direction_deg, representative_speed) tuples.
    """
    results = []
    for direction, speed_bins in wind_rose.items():
        total_freq = sum(sb["frequency"] for sb in speed_bins)
        if total_freq < min_frequency:
            continue
        weighted_speed = sum(
            sb["speed"] * sb["frequency"] for sb in speed_bins
        ) / total_freq if total_freq > 0 else 0
        results.append((direction, weighted_speed, total_freq))
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def setup_building_geometry(config: dict, rotation_angle: float):
    """Create a residential neighborhood geometry in the current Gmsh model.

    Buildings are generated inside a dedicated square zone with:
      - randomized sizes, heights and positions,
      - individual rotations,
      - a few open gaps/courtyards for realism.

    Returns (building_tags, building_heights).
    """
    q = config["quartier"]
    import random
    random.seed(config["seed"])

    rng = q.get("randomize", {})
    width_range = rng.get("width_range", [0.7, 1.4])
    depth_range = rng.get("depth_range", [0.7, 1.3])
    height_range = rng.get("height_range", [0.8, 1.2])
    position_jitter = rng.get("position_jitter", 0.25)
    rotation_jitter = rng.get("rotation_jitter", 20.0)

    building_tags = []
    building_heights = []

    # Dedicated square neighborhood zone
    zone_size = 220.0
    zone_center_x = 0.0
    zone_center_y = 0.0
    min_building_gap = 6.0

    n_buildings = q.get("n_buildings", 35)

    for _ in range(n_buildings):
        placed = False
        attempts = 0
        max_attempts = 100
        
        while not placed and attempts < max_attempts:
            w = random.uniform(12.0, 28.0) * random.uniform(*width_range)
            d = random.uniform(12.0, 28.0) * random.uniform(*depth_range)
            h = random.uniform(q["min_h"] * height_range[0], q["max_h"] * height_range[1])

            x = random.uniform(zone_center_x - zone_size / 2 + w / 2,
                               zone_center_x + zone_size / 2 - w / 2)
            y = random.uniform(zone_center_y - zone_size / 2 + d / 2,
                               zone_center_y + zone_size / 2 - d / 2)

            new_bbox = [
                x - w / 2 - min_building_gap,
                y - d / 2 - min_building_gap,
                0,
                x + w / 2 + min_building_gap,
                y + d / 2 + min_building_gap,
                h
            ]
            
            overlaps = False
            for existing_tag in building_tags:
                existing_bbox = gmsh.model.occ.getBoundingBox(3, existing_tag)
                if not (new_bbox[3] < existing_bbox[0] or 
                        new_bbox[0] > existing_bbox[3] or 
                        new_bbox[4] < existing_bbox[1] or 
                        new_bbox[1] > existing_bbox[4]):
                    overlaps = True
                    break
            
            if not overlaps:
                tag = gmsh.model.occ.addBox(x - w / 2, y - d / 2, 0, w, d, h)
                building_tags.append(tag)
                building_heights.append(h)
                placed = True
            
            attempts += 1
        
        if not placed:
            print(f"  WARNING: Could not place building after {max_attempts} attempts")
            continue

        rot = random.uniform(-rotation_jitter, rotation_jitter)
        if abs(rot) > 0.1:
            gmsh.model.occ.rotate([(3, tag)], x, y, 0, 0, 0, 1, math.radians(rot))
            gmsh.model.occ.synchronize()
            
            # Check overlap after rotation
            new_bbox = gmsh.model.occ.getBoundingBox(3, tag)
            for existing_tag in building_tags[:-1]:  # exclude the one we just added
                existing_bbox = gmsh.model.occ.getBoundingBox(3, existing_tag)
                if not (new_bbox[3] < existing_bbox[0] or 
                        new_bbox[0] > existing_bbox[3] or 
                        new_bbox[4] < existing_bbox[1] or 
                        new_bbox[1] > existing_bbox[4]):
                    # Overlap detected after rotation - remove and retry
                    gmsh.model.occ.remove([(3, tag)])
                    building_tags.pop()
                    building_heights.pop()
                    placed = False
                    break

    gmsh.model.occ.synchronize()

    if rotation_angle != 0:
        angle_rad = math.radians(rotation_angle)
        building_volumes = [(3, t) for t in building_tags]
        gmsh.model.occ.rotate(building_volumes, 0, 0, 0, 0, 0, 1, angle_rad)
        gmsh.model.occ.synchronize()

    return building_tags, building_heights



def create_fluid_domain(config: dict, building_tags: list = None) -> tuple:
    """Create the axis-aligned fluid domain box.

    Margins:
      - upstream/downstream/top: multiples of max building height H
      - lateral: multiples of building width D (lot_length / n_buildings_side)

    Returns (fluid_tag, (Dx, Dy, Dz, xmin, ymin, zmin)).
    """
    q = config["quartier"]
    d = config["domaine_fluide"]
    H = q["max_h"]

    # Approximate building width D from lot layout
    space = q["lot_length"] - 20
    bw = space / q["n_buildings_side"] - q["gap"]
    total_width = q["n_buildings_side"] * bw + (q["n_buildings_side"] - 1) * q["gap"]
    D = total_width

    # Compute margins
    upstream = d["upstream_H"] * H
    downstream = d["downstream_H"] * H
    lateral = d["lateral_D"] * D
    top_margin = d["top_H"] * H

    if building_tags:
        bboxes = [gmsh.model.occ.getBoundingBox(3, tag) for tag in building_tags]
        building_xmin = min(bb[0] for bb in bboxes)
        building_xmax = max(bb[3] for bb in bboxes)
        building_ymin = min(bb[1] for bb in bboxes)
        building_ymax = max(bb[4] for bb in bboxes)
        building_zmax = max(bb[5] for bb in bboxes)

        cx = (building_xmin + building_xmax) / 2
        cy = (building_ymin + building_ymax) / 2

        # Lateral margins measured from the outermost building faces
        xmin = cx - upstream
        xmax = building_xmax + downstream
        ymin = cy - lateral
        ymax = cy + lateral
        zmin = 0
        zmax = building_zmax + top_margin

        Dx = xmax - xmin
        Dy = ymax - ymin
        Dz = zmax - zmin

        fluid_tag = gmsh.model.occ.addBox(xmin, ymin, zmin, Dx, Dy, Dz)
    else:
        Dx = q["lot_length"] * 2
        Dy = q["lot_width"] * 2
        Dz = H * 3
        xmin, ymin, zmin = -Dx/2, -Dy/2, 0
        fluid_tag = gmsh.model.occ.addBox(xmin, ymin, zmin, Dx, Dy, Dz)

    gmsh.model.occ.synchronize()
    return fluid_tag, (Dx, Dy, Dz, xmin, ymin, zmin)

def generate_case(direction_deg: float, speed: float, frequency: float,
                  config: dict, output_dir: Path, z0: float,
                  z_ref: float, intensity: float,
                  fluid_props: dict, turbulence_model: str,
                  building_tags: list = None, building_bboxes: list = None):
    """Generate a single CFD case for a given wind direction and speed.

    If building_tags and building_bboxes are provided, reuse them instead
    of regenerating buildings (ensures identical buildings across directions).

    Returns the case directory path.
    """
    case_name = f"wind_{int(direction_deg)}deg"
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    nu = fluid_props["kinematic_viscosity"]
    print(f"\n{'=' * 60}")
    print(f"Generating case: {case_name}")
    print(f"  Wind direction: {direction_deg}°")
    print(f"  Speed (10 m): {speed:.2f} m/s")
    print(f"  Frequency: {frequency:.4f}")
    print(f"{'=' * 60}")

    # --- Gmsh geometry (Meshing creates the Gmsh model) ---
    mesh = Meshing(case_dir, mesher="gmsh")
    # Suppress verbose output for cleaner logs
    mesh.mesher.verbose = False

    rotation_angle = rotation_angle_for_wind_direction(direction_deg)

    # Reuse pre-generated buildings if provided
    if building_tags is None:
        building_tags, building_heights = setup_building_geometry(config, rotation_angle)
    else:
        # Reuse existing buildings: recreate them in the current Gmsh model
        import gmsh as _gmsh
        building_heights = []
        new_tags = []
        for i, (tag, bbox) in enumerate(zip(building_tags, building_bboxes)):
            xmin, ymin, zmin_b, xmax, ymax, zmax = bbox
            w = xmax - xmin
            d = ymax - ymin
            h = zmax - zmin_b
            new_tag = _gmsh.model.occ.addBox(xmin, ymin, zmin_b, w, d, h)
            new_tags.append(new_tag)
            building_heights.append(h)
        building_tags = new_tags
        _gmsh.model.occ.synchronize()
        if rotation_angle != 0:
            angle_rad = math.radians(rotation_angle)
            building_volumes = [(3, t) for t in building_tags]
            _gmsh.model.occ.rotate(building_volumes, 0, 0, 0, 0, 0, 1, angle_rad)
            _gmsh.model.occ.synchronize()

    # Get fresh bboxes from the current Gmsh model
    building_bboxes = [
        gmsh.model.occ.getBoundingBox(3, tag) for tag in building_tags
    ]

    fluid_tag, (Dx, Dy, Dz, xmin, ymin, zmin) = create_fluid_domain(config, building_tags)

    # --- Save building bboxes BEFORE cut ---
    building_bboxes = [
        gmsh.model.occ.getBoundingBox(3, tag) for tag in building_tags
    ]

    # --- Cut du fluide par les bâtiments ---
    # On soustrait les bâtiments du fluide pour obtenir le domaine fluide troué
    fluid_cut, _ = gmsh.model.occ.cut(
        [(3, fluid_tag)],
        [(3, t) for t in building_tags],
    )
    gmsh.model.occ.synchronize()
    fluid_volume = fluid_cut[0][1] if fluid_cut else fluid_tag

    # Identifier les faces du fluide adjacentes aux bâtiments
    # Ces faces sont dans la bbox des bâtiments et deviendront le patch buildings
    building_faces = []
    all_faces = gmsh.model.getEntities(dim=2)
    for dim, face in all_faces:
        try:
            com = gmsh.model.occ.getCenterOfMass(2, face)
        except Exception:
            continue
        for bbox_b in building_bboxes:
            if (bbox_b[0] - 0.1 <= com[0] <= bbox_b[3] + 0.1 and
                bbox_b[1] - 0.1 <= com[1] <= bbox_b[4] + 0.1 and
                bbox_b[2] - 0.1 <= com[2] <= bbox_b[5] + 0.1):
                building_faces.append(face)
                break

    if building_faces:
        mesh.mesher.assign_physical_groups(
            patch_map={"buildings": building_faces}
        )

    # Apply surface refinement to building faces if configured
    surface_refinement = config.get("maillage", {}).get("surface_refinement")
    if surface_refinement:
        lc_surface = config["maillage"]["lc_min"] * surface_refinement
        if building_faces:
            print(f"  Applying surface refinement to {len(building_faces)} building faces (lc={lc_surface:.2f})")
            gmsh.model.mesh.setSize([(2, tag) for tag in building_faces], lc_surface)

    # Bbox of the fluid volume (domain is axis-aligned, so no rotation)
    bbox = gmsh.model.occ.getBoundingBox(3, fluid_volume)
    xmax, ymax, zmax = bbox[3], bbox[4], bbox[5]

    # --- Patch assignment ---
    mesh.mesher.assign_boundary_patches(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        zmin=zmin, zmax=zmax
    )

    # --- Mesh generation ---
    m = config["maillage"]

    mesh.mesher.mesh_volume(
        lc_min=m["lc_min"],
        lc_max=m["lc_max"],
        optimize=True,
        algorithm_3d=m.get("algorithm_3d", 4),
    )

    # Export direct to OpenFOAM polyMesh
    mesh.mesher.export_to_openfoam_direct()
    mesh.mesher.finalize()

    # --- Patch types: INLET/OUTLET → patch, walls → wall ---
    boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
    content = boundary_file.read_text()
    import re
    wall_patches = ["GROUND", "SIDE_NORTH", "SIDE_SOUTH",
                    "buildings", "UNASSIGNED"]
    for patch_name in wall_patches:
        pattern = rf'({patch_name}\s*\{{\s*type\s+)patch(;)'
        content = re.sub(pattern, r'\1wall\2', content)

    # TOP is symmetry, not wall
    content = re.sub(
        r'(TOP\s*\{\s*type\s+)patch(;)',
        r'\1symmetry\2',
        content
    )
    boundary_file.write_text(content)

    # Ensure TOP boundary conditions are slip / zero-gradient where appropriate
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
            r'("TOP"\s*\{\s*type\s+)\w+(;\s*[^}]*\})',
            r'\1' + slip_fields[field_name] + r'\2',
            field_content
        )
        field_file.write_text(field_content)

    # --- Solver setup ---
    solver = Solver(case_dir)
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = turbulence_model
    solver.transient = False

    solver.constant.transportProperties.nu = nu

    # simpleFoam configuration
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.deltaT = 1.0

    # Let the run continue until convergence, not a fixed iteration count
    solver.system.controlDict.endTime = 2000
    # Write every time step but keep only the last converged result
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

    solver.system.ensure_decomposeParDict(4)
    solver.system.write()
    solver.constant.write()

    # --- Boundary conditions ---
    solver.boundary.initialize_boundary()

    # Compute ABL profile parameters
    u_star = friction_velocity(speed, z0, z_ref)

    # Inlet: codedFixedValue for U with logarithmic wind profile
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

    solver.boundary.set_raw_condition("INLET", "U", {
        "type": "codedFixedValue",
        "name": "inletVelocityProfile",
        "value": "uniform (0 0 0)",
        "code": "#{\n" + u_code + "\n#};",
    })

    # k at inlet (codedFixedValue)
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

    solver.boundary.set_raw_condition("INLET", "k", {
        "type": "codedFixedValue",
        "name": "inletTkeProfile",
        "value": f"uniform {1.5 * (intensity * speed) ** 2}",
        "code": "#{\n" + k_code + "\n#};",
    })

    # epsilon at inlet (codedFixedValue) — k-epsilon models only
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
        solver.boundary.set_raw_condition("INLET", "epsilon", {
            "type": "codedFixedValue",
            "name": "inletEpsilonProfile",
            "value": "uniform 0.1",
            "code": "#{\n" + eps_code + "\n#};",
        })

    # Outlet: pressure outlet (uniform 0)
    solver.boundary.apply_condition_with_wildcard(
        pattern="OUTLET",
        condition_type="pressureOutlet"
    )
    
    # Inlet pressure: zeroGradient (required for simpleFoam)
    solver.boundary.set_raw_condition("INLET", "p", {"type": "zeroGradient"})
    
    # Inlet omega (codedFixedValue with log profile for k-omega SST only)
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
        solver.boundary.set_raw_condition("INLET", "omega", {
            "type": "codedFixedValue",
            "name": "inletOmegaProfile",
            "value": "uniform 0",
            "code": "#{\n" + omega_code + "\n#};",
        })
    
    # Inlet nut: zeroGradient
    solver.boundary.set_raw_condition("INLET", "nut", {"type": "zeroGradient"})

    # TOP: symmetry boundary (free-slip top of domain)
    solver.boundary.set_condition("TOP", "symmetry")

    # Side boundaries: noFrictionWall (open lateral boundaries)
    solver.boundary.set_condition("SIDE_NORTH", "noFrictionWall")
    solver.boundary.set_condition("SIDE_SOUTH", "noFrictionWall")

    # Walls: noSlip with wall functions (already set by initialize_boundary)
    # nut, k, epsilon on walls are handled by wall functions

    solver.boundary.write_boundary_conditions(
        internal_field_overrides={"U": f"uniform ({speed} 0 0)"}
    )

    # Save case metadata
    metadata = {
        "direction_deg": direction_deg,
        "speed_10m": speed,
        "frequency": frequency,
        "z0": z0,
        "z_ref": z_ref,
        "intensity": intensity,
        "rotation_angle": rotation_angle,
        "u_star": u_star,
        "nu": float(nu.get_in("m^2/s") if hasattr(nu, 'get_in') else nu),
        "turbulence_model": turbulence_model,
        "n_buildings": len(building_tags),
        "building_heights": building_heights,
        "domain_dims": {"Dx": Dx, "Dy": Dy, "Dz": Dz, "xmin": xmin, "ymin": ymin, "zmin": zmin},
    }
    with open(case_dir / "case_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Case generated: {case_dir}")
    return case_dir


def main():
    parser = argparse.ArgumentParser(description="Generate CFD cases for wind rose analysis")
    parser.add_argument("--epw", required=True, help="Path to EPW weather file")
    parser.add_argument("--directions", default="all",
                        help="Comma-separated wind directions, or 'all' for dominant sectors")
    parser.add_argument("--z0", type=float, default=0.3, help="Surface roughness length (m)")
    parser.add_argument("--z-ref", type=float, default=10.0, help="Reference height (m)")
    parser.add_argument("--intensity", type=float, default=0.1, help="Turbulence intensity")
    parser.add_argument("--min-freq", type=float, default=0.02, help="Min frequency threshold")
    parser.add_argument("--direction-bin", type=float, default=22.5, help="Wind direction bin width (deg)")
    parser.add_argument("--output", default="cases", help="Output directory for cases")
    parser.add_argument("--turbulence-model", default="kOmegaSST",
                        choices=["kEpsilon", "kOmegaSST", "SpalartAllmaras"],
                        help="Turbulence model")
    args = parser.parse_args()

    epw_path = Path(args.epw)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Fluid properties ---
    available_fluids = FluidMechanics.get_available_fluids()
    fluid_mech = FluidMechanics(
        available_fluids["Air"],
        temperature=ValueWithUnit(293.15, "K"),
        pressure=ValueWithUnit(101325, "Pa"),
    )
    fluid_props = fluid_mech.get_fluid_properties()
    print(f"Air properties: nu = {fluid_props['kinematic_viscosity']}")

    # --- Wind rose ---
    config = load_buildings_config()
    wind_rose = create_wind_rose(epw_path, direction_bin=args.direction_bin)

    # Save wind rose JSON for post-processing
    wind_rose_clean = {str(k): v for k, v in wind_rose.items()}
    with open(output_dir.parent / "wind_rose.json", "w") as f:
        json.dump(wind_rose_clean, f, indent=2)
    with open(output_dir.parent / "buildings_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wind rose saved to {output_dir.parent / 'wind_rose.json'}")

    # --- Select directions ---
    if args.directions == "all":
        dominant = get_dominant_directions(wind_rose, min_frequency=args.min_freq)
        directions = [(d, s, f) for d, s, f in dominant]
        print(f"Dominant wind directions (freq > {args.min_freq}):")
        for d, s, f in directions:
            print(f"  {d:>5.1f}°  speed={s:.1f} m/s  freq={f:.4f}")
    else:
        selected = [float(d) for d in args.directions.split(",")]
        directions = []
        for d in selected:
            if d in wind_rose:
                bins = wind_rose[d]
                total_freq = sum(sb["frequency"] for sb in bins)
                weighted_speed = sum(sb["speed"] * sb["frequency"] for sb in bins) / total_freq
                directions.append((d, weighted_speed, total_freq))
            else:
                directions.append((d, 5.0, 0.0625))

    # --- Generate buildings once, reuse for all directions ---
    print("\nGenerating shared building geometry...")
    import gmsh
    gmsh.initialize()
    gmsh.model.add("shared_buildings")
    building_tags, building_heights = setup_building_geometry(config, rotation_angle=0.0)
    building_bboxes = [
        gmsh.model.occ.getBoundingBox(3, tag) for tag in building_tags
    ]
    gmsh.finalize()

    print(f"  {len(building_tags)} buildings generated with seed={config['seed']}")
    print(f"  Building heights: {[f'{h:.1f}' for h in building_heights]}")

    print(f"\nGenerating {len(directions)} CFD cases...")
    for direction, speed, frequency in directions:
        generate_case(
            direction, speed, frequency, config, output_dir,
            args.z0, args.z_ref, args.intensity,
            fluid_props, args.turbulence_model,
            building_tags=building_tags,
            building_bboxes=building_bboxes,
        )

    print(f"\n{'=' * 60}")
    print(f"All {len(directions)} cases generated in: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()