#!/usr/bin/env python3
"""
VoxCity -> OpenFOAM full pipeline example with dedicated post-processing.

This script demonstrates the complete chain using real VoxCity data:
    1. Load VoxCity HDF5 (pre-downloaded, no EE cost)
    2. Build Gmsh geometry from vector data (fragment + MeshAdapt)
    3. Export directly to OpenFOAM polyMesh
    4. Set up solver with log-wind profile BCs
    5. Run simulation
    6. Post-process with VoxCity-aware analysis

Usage:
    PYTHONPATH=../../foampilot/src:../voxcity_export_work/src:. python3 run_full_voxcity_pipeline.py \
        --hdf5 output/voxcity.h5 \
        --output neighborhood_case \
        --skip-run

    # Full run
    PYTHONPATH=../../foampilot/src:../voxcity_export_work/src:. python3 run_full_voxcity_pipeline.py \
        --hdf5 output/voxcity.h5 \
        --output neighborhood_case
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "foampilot" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "voxcity_export_work" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foampilot import FluidMechanics, ValueWithUnit
from foampilot.urban import Building, UrbanModel
from foampilot.urban.model.terrain import CFDTerrain
from foampilot.solver import Solver
from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing
from vector_builder import VectorGmshBuilder
from wind_profile import KAPPA, Z_REF

RHO_AIR = 1.225
NU_AIR = 1.5e-5


def load_voxcity_hdf5_urban(hdf5_path: str) -> tuple[UrbanModel, CFDTerrain]:
    """Load UrbanModel and CFDTerrain from a VoxCity HDF5 file.

    Uses the same loading logic as generate.py --voxcity-h5.
    """
    import shapely.ops
    from foampilot.urban.readers.voxcity_reader import VoxCityReader

    h5_path = Path(hdf5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"VoxCity HDF5 not found: {hdf5_path}")

    print(f"  Loading VoxCity from HDF5: {hdf5_path}")

    try:
        from voxcity.io import load_voxcity
        voxcity = load_voxcity(hdf5_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load VoxCity HDF5: {e}") from e

    urban = UrbanModel()
    terrain = CFDTerrain.flat(z=0.0)
    count = 0

    gdf = getattr(voxcity, "extras", {}).get("building_gdf")
    if gdf is None and hasattr(voxcity, "building_gdf"):
        gdf = voxcity.building_gdf
    if gdf is not None and len(gdf) > 0:
        try:
            from voxcity.geoprocessor.overlap import process_building_footprints_by_overlap
            gdf = process_building_footprints_by_overlap(gdf, overlap_threshold=0.5)
            print(f"  VoxCity overlap processing: merged buildings with >50% overlap")
        except Exception as e:
            print(f"  WARNING: VoxCity overlap processing failed ({e}), using raw GDF")

        try:
            import pyproj
            project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True).transform
            gdf_proj = gdf.copy()
            gdf_proj.geometry = gdf_proj.geometry.apply(
                lambda geom: shapely.ops.transform(project, geom) if geom is not None else None
            )
        except Exception:
            gdf_proj = gdf

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

            projected_footprints = None
            try:
                projected_row = gdf_proj.loc[idx]
                projected_geom = projected_row.geometry
                if projected_geom is not None and not projected_geom.is_empty:
                    if projected_geom.geom_type == "Polygon":
                        projected_footprints = [projected_geom]
                    elif projected_geom.geom_type == "MultiPolygon":
                        projected_footprints = list(projected_geom.geoms)
            except Exception:
                projected_footprints = None

            for footprint_idx, footprint in enumerate(footprints):
                area_m2 = 0.0
                if projected_footprints is not None and footprint_idx < len(projected_footprints):
                    area_m2 = projected_footprints[footprint_idx].area
                else:
                    try:
                        area_m2 = float(gdf_proj.loc[idx].geometry.area)
                    except Exception:
                        area_m2 = 0.0

                if area_m2 < 1.0:
                    continue
                height = float(getattr(row, "height", 9.0) or 9.0)
                use_footprint = footprint if projected_footprints is None else projected_footprints[footprint_idx]
                urban.add_building(Building(
                    id=f"vox_{idx}_{footprint_idx}" if len(footprints) > 1 else f"vox_{idx}",
                    footprint=use_footprint,
                    ground_z=0.0,
                    roof_z=height,
                    source="voxcity",
                    confidence=0.7,
                ))
                count += 1

    print(f"  Loaded {count} buildings from HDF5")
    return urban, terrain


def load_voxcity_metadata(hdf5_path: str) -> dict:
    """Extract VoxCity metadata for post-processing."""
    meta = {"hdf5_path": str(hdf5_path)}
    try:
        with h5py.File(hdf5_path, "r") as f:
            if "rectangle_vertices" in f:
                meta["aoi"] = f["rectangle_vertices"][:].tolist()
            vox = f.get("voxcity", {})
            if "building_height" in vox:
                meta["grid_shape"] = list(vox["building_height"].shape)
            if "dem" in vox:
                meta["dem_shape"] = list(vox["dem"].shape)
            extras = vox.get("extras_gdf", {})
            if "columns" in extras:
                try:
                    meta["gdf_columns"] = [c.decode() if isinstance(c, bytes) else c for c in extras["columns"][:]]
                except Exception:
                    pass
    except Exception as e:
        print(f"  WARNING: Could not extract metadata: {e}")
    return meta


def setup_openfoam_case(case_dir: Path, config: dict):
    """Set up OpenFOAM case files with log-wind profile BCs."""
    fluid_props = FluidMechanics.get_available_fluids()
    fluid_mech = FluidMechanics(
        fluid_props["Air"],
        temperature=ValueWithUnit(293.15, "K"),
        pressure=ValueWithUnit(101325, "Pa"),
    )
    props = fluid_mech.get_fluid_properties()
    nu = props["kinematic_viscosity"]

    solver_cfg = config["solver"]
    solver = Solver(case_dir)
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = solver_cfg["turbulence_model"]
    solver.transient = False

    solver.constant.transportProperties.nu = nu

    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = float(solver_cfg["end_time"])
    solver.system.controlDict.deltaT = float(solver_cfg["delta_t"])
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = int(solver_cfg["write_interval"])
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

    boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
    content = boundary_file.read_text()
    wall_patches = ["buildings", "ground", "side_left", "side_right"]
    for patch_name in wall_patches:
        pattern = rf'({patch_name}\s*\{{\s*type\s+)patch(;)'
        content = __import__("re").sub(pattern, r"\1wall\2", content)
    content = __import__("re").sub(r'(top\s*\{\s*type\s+)patch(;)', r"\1symmetry\2", content)
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
        field_content = __import__("re").sub(
            r'("top"\s*\{\s*type\s+)\w+(;\s*[^}]*\})',
            r"\1" + slip_fields[field_name] + r"\2",
            field_content,
        )
        field_file.write_text(field_content)

    solver.boundary.initialize_boundary()

    z0 = float(solver_cfg["z0"])
    z_ref = float(solver_cfg["z_ref"])
    speed = float(solver_cfg["u_ref"])
    intensity = float(solver_cfg["turbulence_intensity"])
    turbulence_model = solver_cfg["turbulence_model"]
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

    turbulence_model = solver_cfg["turbulence_model"]

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

    return solver, speed


def run_postprocessing(case_dir: Path, output_dir: Path, speed: float, hdf5_path: str = None, domain_bounds: tuple = None):
    """Run VoxCity-aware post-processing on the completed case."""
    from voxcity_dedicated_postprocess import run_voxcity_postprocessing
    run_voxcity_postprocessing(case_dir, output_dir, hdf5_path=hdf5_path, speed=speed,
                                domain_bounds=domain_bounds)


def main():
    parser = argparse.ArgumentParser(description="VoxCity -> OpenFOAM full pipeline with VoxCity post-processing")
    parser.add_argument("--hdf5", required=True, help="Path to VoxCity HDF5 file")
    parser.add_argument("--output-dir", default="neighborhood_case", help="Output case directory")
    parser.add_argument("--mesh-size", type=float, default=6.0, help="Gmsh mesh size (m)")
    parser.add_argument("--margin", type=float, default=None, help="Domain margin around buildings (m). None = automatic (4H/7.5H/2D/1.25H)")
    parser.add_argument("--skip-run", action="store_true", help="Skip simulation, only mesh + case setup")
    parser.add_argument("--post-only", action="store_true", help="Only run post-processing")
    parser.add_argument("--fill-gaps", action="store_true", help="Fill small gaps between buildings")
    parser.add_argument("--mesh-constraint", default="none", choices=["none", "proximity"],
                        help="Mesh sizing constraint")
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    case_dir = Path(args.output_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    if args.post_only:
        output_dir = case_dir / "post"
        run_postprocessing(case_dir, output_dir, hdf5_path=args.hdf5, speed=config["solver"]["u_ref"])
        return

    print("=" * 60)
    print("VoxCity -> OpenFOAM full pipeline")
    print("=" * 60)

    # Step 1: Load VoxCity data
    print("\n[1/7] Loading VoxCity data from HDF5...")
    urban, terrain = load_voxcity_hdf5_urban(args.hdf5)
    print(f"  Buildings: {urban.building_count()}")

    # Step 2: Build Gmsh mesh
    print(f"\n[2/7] Building Gmsh mesh (size={args.mesh_size} m)...")
    builder = VectorGmshBuilder(
        urban, terrain,
        mesh_size=args.mesh_size,
        mesh_constraint=args.mesh_constraint,
        fill_gaps=args.fill_gaps,
    )
    builder.build(margin=args.margin, bottom_offset=config["domain"]["bottom_offset"])
    builder.assign_patches()
    builder.build_mesh(mesh_size=args.mesh_size)

    domain_bounds = None
    try:
        import gmsh
        xmin, ymin, zmin, xmax, ymax, zmax = builder.urban.bbox()
        margin = args.margin
        bottom_offset = config["domain"]["bottom_offset"]
        if margin is None:
            heights = [b.roof_z - b.ground_z for b in builder.urban.buildings()]
            Hmax = max(heights) if heights else 10.0
            D = xmax - xmin
            margin_x_upstream = 8.0 * Hmax
            margin_x_downstream = 15.0 * Hmax
            margin_y = 4.0 * max(D, 1.0)
            margin_z = 2.5 * Hmax
            domain_bounds = (
                xmin - margin_x_upstream,
                ymin - margin_y,
                zmin - bottom_offset,
                xmax + margin_x_downstream,
                ymax + margin_y,
                zmax + margin_z,
            )
        else:
            domain_bounds = (
                xmin - margin,
                ymin - margin,
                zmin - bottom_offset,
                xmax + margin,
                ymax + margin,
                zmax + margin,
            )
    except Exception:
        domain_bounds = None

    # Step 3: Export to OpenFOAM
    print(f"\n[3/7] Exporting to OpenFOAM polyMesh...")
    builder.export_openfoam(case_dir)
    builder.finalize()

    # Step 3b: Mesh quality check with foampilot quality gates
    print(f"\n[3b/7] Checking mesh quality...")
    try:
        from foampilot.mesh.quality.openfoam_quality import OpenFOAMQualityAnalyzer, console_report
        import subprocess

        checkmesh_log = case_dir / "log.checkMesh"
        if not checkmesh_log.exists():
            print("  Running checkMesh...")
            proc = subprocess.run(
                ["checkMesh", "-allGeometry", "-allTopology", "-case", str(case_dir.resolve())],
                capture_output=True, text=True, timeout=120, cwd=str(case_dir),
            )
            checkmesh_log.write_text(proc.stdout + proc.stderr, encoding="utf-8", errors="ignore")

        analyzer = OpenFOAMQualityAnalyzer(case_dir)
        quality_report = analyzer.analyze()
        print(console_report(quality_report, case_name=str(case_dir)))
        gate_status = quality_report.get("gate", {}).get("status", "UNKNOWN")
        if gate_status == "BAD":
            print(f"  ERROR: Mesh quality gate is BAD — aborting before solver run.")
            sys.exit(1)
        elif gate_status == "WARNING":
            print(f"  WARNING: Mesh quality gate is WARNING — continuing but results may be degraded.")
        else:
            print(f"  Mesh quality gate: {gate_status} — proceeding to solver.")
    except Exception as e:
        print(f"  WARNING: Mesh quality check failed ({e}) — proceeding without quality gate.")

    # Step 4: Set up OpenFOAM case
    print(f"\n[4/7] Setting up OpenFOAM case...")
    solver, speed = setup_openfoam_case(case_dir, config)

    # Step 5: Run simulation
    if not args.skip_run:
        print(f"\n[5/7] Running simulation...")
        solver.run_simulation(nb_proc=1)
    else:
        print(f"\n[5/7] Skipping simulation (--skip-run)")

    # Step 6: Post-processing
    print(f"\n[6/7] Post-processing...")
    output_dir = case_dir / "post"
    run_postprocessing(case_dir, output_dir, hdf5_path=args.hdf5, speed=speed,
                       domain_bounds=domain_bounds)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"Case directory: {case_dir}")
    print(f"Buildings: {urban.building_count()}")
    print(f"Post-processing: {output_dir}")
    print(f"\nCommands:")
    print(f"  checkMesh -case {case_dir}")
    print(f"  Rerun post-processing: python3 {Path(__file__).name} --post-only --output-dir {case_dir}")


if __name__ == "__main__":
    main()
