#!/usr/bin/env python3
"""
Neighborhood CFD Demo — VoxCity → Gmsh → OpenFOAM → run → post-process.

Complete pipeline for a realistic urban neighborhood:
  1. Download VoxCity data (buildings + DEM) for a real AOI in Paris.
  2. Build a single-fluid Gmsh mesh with boundary patches.
  3. Export directly to OpenFOAM polyMesh.
  4. Set up solver & BCs (log wind profile at inlet, k-epsilon).
  5. Run the simulation with foampilot.
  6. Post-process: slices, Cp on buildings, mesh quality, statistics.

Usage:
    PYTHONPATH=../../../src python3 generate.py
    PYTHONPATH=../../../src python3 generate.py --skip-run
    PYTHONPATH=../../../src python3 generate.py --post-only
    PYTHONPATH=../../../src python3 generate.py --use-cache
"""

import argparse
import json
import sys
from pathlib import Path

import gmsh

import numpy as np
import shapely.ops

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "voxcity_export_work" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foampilot import FluidMechanics, ValueWithUnit
from foampilot.urban import Building, UrbanModel
from foampilot.urban.model.terrain import CFDTerrain
from foampilot.solver import Solver
from foampilot.urban.readers.voxcity_reader import VoxCityReader
from foampilot.postprocess.openfoam_pyvista import FoamPostProcessing
from vector_builder_build123 import VectorGmshBuilder
from wind_profile import KAPPA, Z_REF

RHO_AIR = 1.225


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def build_voxcity_urban(config: dict, use_cache: bool = False, voxcity_h5: str | None = None):
    """Load real VoxCity data for the AOI defined in config."""
    from shapely.validation import make_valid
    from voxcity.generator import get_voxcity
    from voxcity.io import load_voxcity
    import ee

    try:
        ee.Initialize(project="openfoam-project")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project="openfoam-project")

    aoi = config["aoi"]["rectangle_vertices"]
    vox_cfg = config["voxcity"]

    if voxcity_h5:
        h5_path = Path(voxcity_h5)
        if not h5_path.exists():
            raise FileNotFoundError(f"VoxCity HDF5 not found: {h5_path}")
        print(f"  Loading VoxCity from HDF5: {h5_path}")
        voxcity = load_voxcity(h5_path)
        urban = UrbanModel()
        terrain = CFDTerrain.flat(z=0.0)
        count = 0

        gdf = getattr(voxcity, "extras", {}).get("building_gdf")
        if gdf is None and hasattr(voxcity, "building_gdf"):
            gdf = voxcity.building_gdf
        if gdf is not None and len(gdf) > 0:
            try:
                import pyproj
                project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True).transform
                gdf_proj = gdf.copy()
                gdf_proj.geometry = gdf_proj.geometry.apply(lambda geom: shapely.ops.transform(project, geom) if geom is not None else None)
            except Exception:
                gdf_proj = gdf

            def clean_footprint(geom, min_area_m2=1.0, simplify_tol=0.5, rounding_precision=1):
                if geom is None or geom.is_empty:
                    return None
                try:
                    from shapely.validation import make_valid
                    import shapely
                    geom = make_valid(geom)
                    geom = geom.buffer(0.0)
                    if geom.is_empty:
                        return None
                    geom = shapely.wkt.loads(shapely.wkt.dumps(geom, rounding_precision=rounding_precision))
                    geom = make_valid(geom)
                    geom = geom.buffer(0.0)
                    if geom.is_empty:
                        return None
                    if simplify_tol > 0.0:
                        geom = geom.simplify(simplify_tol, preserve_topology=True)
                        geom = make_valid(geom)
                        geom = geom.buffer(0.0)
                    if geom.is_empty:
                        return None
                    if geom.area < min_area_m2:
                        return None
                    if geom.geom_type == "MultiPolygon":
                        polygons = [p for p in geom.geoms if not p.is_empty and p.area >= min_area_m2]
                        if not polygons:
                            return None
                        from shapely.ops import unary_union
                        geom = unary_union(polygons)
                        geom = make_valid(geom)
                        geom = geom.buffer(0.0)
                    if geom.is_empty or geom.area < min_area_m2:
                        return None
                    return geom
                except Exception:
                    return None

            def merge_nearby_buildings(polys_heights, distance=1.0, height_tol=1.0):
                """Merge nearby footprints while preserving a representative height.

                Buildings are grouped by similar height, then footprints whose buffered
                envelopes touch are fused. The returned height is the area-weighted
                average of the source buildings in each resulting component.
                """
                if not polys_heights:
                    return []
                from shapely.validation import make_valid

                height_groups = []
                for fp, height in polys_heights:
                    for group in height_groups:
                        if abs(height - group[0][1]) <= height_tol:
                            group.append((fp, height))
                            break
                    else:
                        height_groups.append([(fp, height)])

                merged_buildings = []
                for group in height_groups:
                    expanded = [fp.buffer(distance / 2.0) for fp, _ in group]
                    components = shapely.ops.unary_union(expanded)
                    components = make_valid(components).buffer(0.0)
                    if components.is_empty:
                        continue
                    if components.geom_type == "Polygon":
                        components = [components]
                    else:
                        components = list(components.geoms)

                    for component in components:
                        source = [(fp, h) for fp, h in group if fp.intersects(component)]
                        if not source:
                            continue
                        merged = component.buffer(-distance / 2.0)
                        merged = make_valid(merged).buffer(0.0)
                        if merged.is_empty:
                            continue
                        parts = [merged] if merged.geom_type == "Polygon" else list(merged.geoms)
                        total_area = sum(fp.area for fp, _ in source) or 1.0
                        representative_height = sum(fp.area * h for fp, h in source) / total_area
                        for part in parts:
                            if not part.is_empty and part.area >= 2.0:
                                merged_buildings.append((part, representative_height))
                return merged_buildings

            cleaned_footprints = []
            cleaned_heights = []
            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                footprints = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
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
                    if projected_footprints is not None and footprint_idx < len(projected_footprints):
                        use_footprint = projected_footprints[footprint_idx]
                    else:
                        use_footprint = footprint
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
                    cleaned = clean_footprint(use_footprint, min_area_m2=1.0, simplify_tol=0.5, rounding_precision=1)
                    if cleaned is None:
                        continue
                    cleaned_footprints.append(cleaned)
                    height = float(getattr(row, "height", 9.0) or 9.0)
                    cleaned_heights.append(height)

            polys_heights = list(zip(cleaned_footprints, cleaned_heights))
            merged_buildings = merge_nearby_buildings(polys_heights, distance=1.0)
            print(f"  Cleaned: {len(cleaned_footprints)} footprints -> Merged: {len(merged_buildings)} buildings")

            # Close small gaps between merged buildings: buffer(+gap) then buffer(-gap)
            # This fuses buildings separated by less than `gap` meters without changing outer shape.
            gap = 0.5
            merged_buildings = [(fp.buffer(gap / 2.0), height) for fp, height in merged_buildings]
            merged_buildings = [(make_valid(fp), height) for fp, height in merged_buildings]
            merged_buildings = [(fp.buffer(-gap / 2.0), height) for fp, height in merged_buildings]
            merged_buildings = [(make_valid(fp), height) for fp, height in merged_buildings]
            merged_buildings = [(fp.buffer(0.0), height) for fp, height in merged_buildings if not fp.is_empty]

            # Post-merge cleanup: remove tiny fragments created by boolean/buffer ops
            def post_merge_cleanup(buildings, min_area=2.0):
                cleaned = []
                for fp, height in buildings:
                    if fp is None or fp.is_empty:
                        continue
                    fp = make_valid(fp).buffer(0.0)
                    if fp.is_empty:
                        continue
                    parts = [fp] if fp.geom_type == "Polygon" else list(fp.geoms)
                    cleaned.extend((part, height) for part in parts if not part.is_empty and part.area >= min_area)
                return cleaned

            merged_buildings = post_merge_cleanup(merged_buildings, min_area=2.0)
            print(f"  After post-merge cleanup: {len(merged_buildings)} buildings")

            try:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f"Footprint Processing — {len(gdf)} raw → {len(cleaned_footprints)} cleaned → {len(merged_buildings)} merged", fontsize=14)
                ax1 = axes[0]
                ax1.set_title("Raw Footprints")
                for idx, row in gdf.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    if geom.geom_type == "Polygon":
                        polys = [geom]
                    elif geom.geom_type == "MultiPolygon":
                        polys = list(geom.geoms)
                    else:
                        continue
                    for poly in polys:
                        x, y = poly.exterior.xy
                        ax1.fill(x, y, alpha=0.5, edgecolor="black", linewidth=0.5)
                ax1.set_aspect("equal")
                ax1.set_xlabel("X (m)")
                ax1.set_ylabel("Y (m)")
                ax1.grid(True, alpha=0.3)
                ax2 = axes[1]
                ax2.set_title("Cleaned Footprints")
                for fp in cleaned_footprints:
                    x, y = fp.exterior.xy
                    ax2.fill(x, y, alpha=0.5, edgecolor="black", linewidth=0.5)
                ax2.set_aspect("equal")
                ax2.set_xlabel("X (m)")
                ax2.set_ylabel("Y (m)")
                ax2.grid(True, alpha=0.3)
                ax3 = axes[2]
                ax3.set_title("Merged Buildings")
                for fp, _ in merged_buildings:
                    x, y = fp.exterior.xy
                    ax3.fill(x, y, alpha=0.5, edgecolor="black", linewidth=0.5)
                ax3.set_aspect("equal")
                ax3.set_xlabel("X (m)")
                ax3.set_ylabel("Y (m)")
                ax3.grid(True, alpha=0.3)
                plt.tight_layout()
                out_path = Path(__file__).resolve().parent / "footprint_processing_steps.png"
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  Saved footprint processing image: {out_path}")
            except Exception as exc:
                print(f"  WARNING: Could not generate footprint image: {exc}")

            for i, (footprint, height) in enumerate(merged_buildings):
                urban.add_building(Building(
                    id=f"merged_{i}",
                    footprint=footprint,
                    ground_z=0.0,
                    roof_z=height,
                    source="voxcity_merged",
                    confidence=0.7,
                ))
        print(f"  Loaded {urban.building_count()} buildings from HDF5")
        return urban, terrain

    cache_path = Path(__file__).resolve().parent / "output" / "voxcity.h5"
    if use_cache and cache_path.exists():
        print(f"  Using cached VoxCity data: {cache_path}")
    elif use_cache:
        print(f"  No cache found at {cache_path}, will download")

    reader = VoxCityReader(
        meshsize=vox_cfg["meshsize"],
        building_source=vox_cfg.get("building_source"),
        dem_source=vox_cfg.get("dem_source"),
    )
    return reader.read(aoi)


def build_mesh(urban: UrbanModel, terrain: CFDTerrain, config: dict, mesh_constraint: str = "none", fill_gaps: bool = False):
    """Build Gmsh geometry and export to OpenFOAM."""
    mesh_cfg = config["mesh"]
    domain_cfg = config["domain"]

    builder = VectorGmshBuilder(urban, terrain, mesh_size=mesh_cfg["mesh_size"], mesh_constraint=mesh_constraint, fill_gaps=fill_gaps)
    builder.build(
        margin=mesh_cfg["margin"],
        bottom_offset=domain_cfg["bottom_offset"],
    )
    builder.assign_patches()
    builder.build_mesh(mesh_size=mesh_cfg["mesh_size"])
    return builder


def export_openfoam(builder, case_dir: Path):
    """Export mesh to OpenFOAM polyMesh."""
    builder.export_openfoam(case_dir)
    builder.finalize()


def setup_solver(case_dir: Path, config: dict):
    """Set up OpenFOAM case with wind-profile BCs."""
    solver_cfg = config["solver"]
    fluid_props = FluidMechanics.get_available_fluids()
    fluid_mech = FluidMechanics(
        fluid_props["Air"],
        temperature=ValueWithUnit(293.15, "K"),
        pressure=ValueWithUnit(101325, "Pa"),
    )
    props = fluid_mech.get_fluid_properties()
    nu = props["kinematic_viscosity"]

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

    solver.system.ensure_decomposeParDict(1)
    solver.system.write()
    solver.constant.write()

    # Fix boundary patch types
    import re
    boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
    content = boundary_file.read_text()
    wall_patches = ["ground", "side_left", "side_right", "buildings"]
    for patch_name in wall_patches:
        pattern = rf'({patch_name}\s*\{{\s*type\s+)patch(;)'
        content = re.sub(pattern, r'\1wall\2', content)
    content = re.sub(r'(top\s*\{\s*type\s+)patch(;)', r'\1symmetry\2', content)
    boundary_file.write_text(content)

    # Fix top BCs in field files
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

    # BCs
    solver.boundary.initialize_boundary()

    z0 = float(solver_cfg["z0"])
    z_ref = float(solver_cfg["z_ref"])
    speed = float(solver_cfg["u_ref"])
    intensity = float(solver_cfg["turbulence_intensity"])
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

    return solver, speed


def run_postprocessing(case_dir: Path, output_dir: Path, speed: float):
    """Run post-processing on the completed case."""
    from voxcity_postprocess import ensure_vtk, load_latest, generate_visualizations, export_statistics

    print("\n" + "=" * 60)
    print("Post-processing")
    print("=" * 60)

    foam_post = FoamPostProcessing(case_path=case_dir)
    ensure_vtk(foam_post)
    structure = load_latest(foam_post)

    p_ref = 0.5 * RHO_AIR * speed ** 2
    generate_visualizations(case_dir, structure, p_ref=p_ref,
                            pedestrian_height=1.75, output_dir=output_dir)
    export_statistics(case_dir, structure, output_dir, u_ref=speed)


def main():
    parser = argparse.ArgumentParser(description="Neighborhood CFD Demo — VoxCity → OpenFOAM")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--output-dir", default="neighborhood_case", help="Output case directory")
    parser.add_argument("--skip-run", action="store_true", help="Skip simulation")
    parser.add_argument("--post-only", action="store_true", help="Only run post-processing")
    parser.add_argument("--use-cache", action="store_true", help="Use cached VoxCity HDF5 if available")
    parser.add_argument("--voxcity-h5", default=None, help="Path to VoxCity HDF5 file to load directly (skips download)")
    parser.add_argument("--mesh-constraint", default="none", choices=["none", "proximity"], help="Mesh sizing constraint")
    parser.add_argument("--fill-gaps", action="store_true", help="Fill small gaps between nearby buildings")
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent / args.config
    config = load_config(config_path)

    case_dir = Path(args.output_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    if args.post_only:
        run_postprocessing(case_dir, case_dir / "post", speed=config["solver"]["u_ref"])
        return

    print("=" * 60)
    print("Neighborhood CFD Demo")
    print("=" * 60)

    # Step 1: Urban model
    print("\n[1/5] Loading urban model...")
    urban, terrain = build_voxcity_urban(config, use_cache=args.use_cache, voxcity_h5=args.voxcity_h5)
    print(f"  VoxCity: {urban.building_count()} buildings")

    # Step 2: Mesh
    print("\n[2/5] Building Gmsh mesh...")
    mesh_cfg = config["mesh"]
    builder = build_mesh(urban, terrain, config, mesh_constraint=args.mesh_constraint, fill_gaps=args.fill_gaps)

    # Step 3: Export
    print("\n[3/5] Exporting to OpenFOAM...")
    export_openfoam(builder, case_dir)

    # Step 4: Solver
    print("\n[4/5] Setting up solver and BCs...")
    solver, speed = setup_solver(case_dir, config)

    # Step 5: Run
    if not args.skip_run:
        print("\n[5/5] Running simulation...")
        solver.run_simulation(nb_proc=1)
    else:
        print("\n[5/5] Skipping simulation (--skip-run)")

    # Summary
    print("\n" + "=" * 60)
    print("Case generated successfully!")
    print("=" * 60)
    print(f"Case directory: {case_dir}")
    print(f"Buildings: {urban.building_count()}")
    print("\nTo run post-processing:")
    print(f"  PYTHONPATH=../../../src python3 postprocess.py --case {case_dir}")
    print("\nOr rerun simulation:")
    print(f"  PYTHONPATH=../../../src python3 generate.py --output-dir {case_dir}")

    # Auto-run post-processing if simulation completed
    if not args.skip_run:
        run_postprocessing(case_dir, case_dir / "post", speed=speed)


if __name__ == "__main__":
    main()
