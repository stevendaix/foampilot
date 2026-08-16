#!/usr/bin/env python3
"""
Systematic mesh experiment comparing different mesh generation strategies.

Runs a matrix of mesh configurations (M0-M4) and collects quality metrics
from both Gmsh and OpenFOAM, plus CFD convergence results.

Usage:
    PYTHONPATH=src python3 mesh_experiment.py \
        --base-case cases/wind_0deg \
        --output experiments/mesh_study
"""

import argparse
import json
import sys
import time
import shutil
import re
from pathlib import Path
from datetime import datetime

_EXAMPLES_DIR = Path(__file__).resolve().parents[4] / "examples" / "building_aero"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import gmsh
import numpy as np

from foampilot.mesh.quality.gmsh_quality import GmshQualityAnalyzer, QualityThresholds
from foampilot.mesh.quality.openfoam_quality import OpenFOAMQualityAnalyzer
from foampilot.mesh.adaptation.adaptive_mesher import AdaptiveMeshImprover
from foampilot.report import SimulationReport


MESH_CONFIGS = {
    "M0_baseline": {
        "description": "Current baseline: TetGen, lc_min=5, lc_max=15",
        "algorithm_3d": 4,
        "lc_min": 5.0,
        "lc_max": 15.0,
        "optimize": True,
        "optimize_methods": ["Netgen", "Relocate3D"],
    },
    "M1_optimized": {
        "description": "Baseline + Gmsh optimization",
        "algorithm_3d": 4,
        "lc_min": 5.0,
        "lc_max": 15.0,
        "optimize": True,
        "optimize_methods": ["Gmsh", "Netgen", "Relocate3D"],
    },
    "M2_surface_refinement": {
        "description": "Surface refinement on buildings (0.7x)",
        "algorithm_3d": 4,
        "lc_min": 5.0,
        "lc_max": 15.0,
        "optimize": True,
        "optimize_methods": ["Netgen", "Relocate3D"],
        "surface_refinement": 0.7,
    },
    "M3_surface_volume_refinement": {
        "description": "Surface + volume refinement",
        "algorithm_3d": 4,
        "lc_min": 5.0,
        "lc_max": 15.0,
        "optimize": True,
        "optimize_methods": ["Netgen", "Relocate3D"],
        "surface_refinement": 0.7,
        "volume_refinement": 0.8,
    },
    "M4_hxt": {
        "description": "HXT algorithm instead of TetGen",
        "algorithm_3d": 10,
        "lc_min": 5.0,
        "lc_max": 15.0,
        "optimize": True,
        "optimize_methods": ["Netgen", "Relocate3D"],
    },
}


def run_mesh_experiment(base_case: Path, output_dir: Path, max_cells: int = 1_500_000):
    """Run the complete mesh experiment matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / f"mesh_experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": timestamp,
        "base_case": str(base_case),
        "max_cells": max_cells,
        "configs": {},
    }

    for config_name, config in MESH_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"Testing: {config_name}")
        print(f"  {config['description']}")
        print(f"{'=' * 60}")

        case_dir = experiment_dir / config_name
        case_dir.mkdir(parents=True, exist_ok=True)

        result = run_single_mesh_test(base_case, case_dir, config_name, config, max_cells)
        results["configs"][config_name] = result

        cells_val = result.get('cells', 'N/A')
        cells_str = f"{cells_val:,}" if isinstance(cells_val, int) else str(cells_val)
        print(f"  Cells:      {cells_str}")
        print(f"  Gmsh minSICN: {result.get('gmsh_min_sicn', 'N/A')}")
        print(f"  maxNonOrtho: {result.get('max_non_ortho', 'N/A')}")
        print(f"  faces>70:   {result.get('faces_over_70', 'N/A')}")
        print(f"  Converged:  {result.get('converged', 'N/A')}")

    summary_path = experiment_dir / "mesh_experiment_report.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = experiment_dir / "mesh_experiment.csv"
    write_csv(results, csv_path)

    print(f"\n{'=' * 60}")
    print(f"Experiment complete. Results in: {experiment_dir}")
    print(f"  JSON: {summary_path}")
    print(f"  CSV:  {csv_path}")
    print(f"{'=' * 60}")

    return results


def run_single_mesh_test(base_case: Path, case_dir: Path, config_name: str,
                         config: dict, max_cells: int) -> dict:
    """Run a single mesh configuration test."""
    result = {
        "config_name": config_name,
        "description": config["description"],
        "timestamp": datetime.now().isoformat(),
    }

    try:
        shutil.copytree(base_case, case_dir, dirs_exist_ok=True)
        generate_mesh(case_dir, config, max_cells)

        gmsh_result = analyze_gmsh(case_dir)
        result.update(gmsh_result)

        feature_result = analyze_mesh_features(case_dir)
        result.update(feature_result)

        foam_result = analyze_openfoam(case_dir)
        result.update(foam_result)

        cfd_result = run_cfd(case_dir)
        result.update(cfd_result)

        result["status"] = "SUCCESS"

    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
        print(f"  ERROR: {e}")

    return result


def _write_boundary_from_physical_groups(case_dir: Path):
    """Write OpenFOAM boundary file from Gmsh 2-D physical groups."""
    boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
    content = boundary_file.read_text()

    wall_patches = ["GROUND", "SIDE_NORTH", "SIDE_SOUTH",
                    "buildings", "UNASSIGNED"]
    for patch_name in wall_patches:
        pattern = rf'({patch_name}\s*\{{\s*type\s+)patch(;)'
        content = re.sub(pattern, r'\1wall\2', content)

    content = re.sub(
        r'(TOP\s*\{\s*type\s+)patch(;)',
        r'\1symmetry\2',
        content
    )
    boundary_file.write_text(content)


def generate_mesh(case_dir: Path, config: dict, max_cells: int):
    """Generate mesh with given configuration."""
    from foampilot import Meshing

    gmsh.initialize()
    gmsh.model.add(f"mesh_{case_dir.name}")

    try:
        from generate_wind_cases import setup_building_geometry, create_fluid_domain
        import json
        config_path = Path("/home/steven/foampilot/examples/building_aero/buildings_config.json")
        if not config_path.exists():
            config_path = case_dir.parent / "buildings_config.json"
        if config_path.exists():
            with open(config_path) as f:
                buildings_config = json.load(f)
        else:
            buildings_config = {
                "quartier": {"max_h": 40.0, "lot_length": 300.0, "n_buildings_side": 5, "gap": 5.0},
                "seed": 42,
            }

        mesh = Meshing(case_dir, mesher="gmsh")
        mesh.mesher.verbose = False

        building_tags, _ = setup_building_geometry(buildings_config, rotation_angle=0.0)
        building_bboxes = [
            gmsh.model.occ.getBoundingBox(3, tag) for tag in building_tags
        ]
        fluid_tag, (Dx, Dy, Dz, xmin, ymin, zmin) = create_fluid_domain(buildings_config, building_tags)

        fluid_cut, _ = gmsh.model.occ.cut(
            [(3, fluid_tag)],
            [(3, t) for t in building_tags],
        )
        gmsh.model.occ.synchronize()
        fluid_volume = fluid_cut[0][1] if fluid_cut else fluid_tag

        bbox = gmsh.model.occ.getBoundingBox(3, fluid_volume)
        xmax, ymax, zmax = bbox[3], bbox[4], bbox[5]

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

        mesh.mesher.assign_boundary_patches(
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            zmin=zmin, zmax=zmax
        )

        m = {"lc_min": config["lc_min"], "lc_max": config["lc_max"]}

        if config.get("surface_refinement") or config.get("volume_refinement"):
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

            if config.get("surface_refinement"):
                lc_surface = config["lc_min"] * config["surface_refinement"]
                bbox = gmsh.model.occ.getBoundingBox(3, fluid_volume)
                xmin, ymin, zmin, xmax, ymax, zmax = bbox
                all_surfaces = gmsh.model.getEntities(dim=2)
                building_surfaces = []
                for surf in all_surfaces:
                    try:
                        sbbox = gmsh.model.occ.getBoundingBox(2, surf[1])
                        sxmin, symin, szmin, sxmax, symax, szmax = sbbox
                        cx = (sxmin + sxmax) / 2
                        cy = (symin + symax) / 2
                        cz = (szmin + szmax) / 2
                        if (abs(cz - zmin) > 1e-6 and abs(cz - zmax) > 1e-6 and
                            abs(cx - xmin) > 1e-6 and abs(cx - xmax) > 1e-6 and
                            abs(cy - ymin) > 1e-6 and abs(cy - ymax) > 1e-6):
                            building_surfaces.append(surf[1])
                    except Exception:
                        pass

                if building_surfaces:
                    print(f"  Applying surface refinement to {len(building_surfaces)} building surfaces (lc={lc_surface})")
                    field_dist = gmsh.model.mesh.field.add("Distance")
                    gmsh.model.mesh.field.setNumbers(field_dist, "SurfacesList", building_surfaces)
                    gmsh.model.mesh.field.setNumber(field_dist, "Sampling", 100)
                    field_thresh = gmsh.model.mesh.field.add("Threshold")
                    gmsh.model.mesh.field.setNumber(field_thresh, "InField", field_dist)
                    gmsh.model.mesh.field.setNumber(field_thresh, "DistMin", 0.0)
                    gmsh.model.mesh.field.setNumber(field_thresh, "DistMax", 20.0)
                    gmsh.model.mesh.field.setNumber(field_thresh, "LcMin", lc_surface)
                    gmsh.model.mesh.field.setNumber(field_thresh, "LcMax", config["lc_max"])
                    gmsh.model.mesh.field.setAsBackgroundMesh(field_thresh)

            if config.get("volume_refinement"):
                lc_volume = config["lc_min"] * config["volume_refinement"]
                volumes = [v[1] for v in gmsh.model.getEntities(dim=3)]
                gmsh.model.mesh.setSize([(3, t) for t in volumes], lc_volume)

        mesh.mesher.mesh_volume(
            lc_min=m["lc_min"],
            lc_max=m["lc_max"],
            optimize=True,
            algorithm_3d=4,
        )

        mesh.mesher.export_to_openfoam_direct()
        mesh.mesher.finalize()

        boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
        content = boundary_file.read_text()
        wall_patches = ["GROUND", "SIDE_NORTH", "SIDE_SOUTH",
                        "buildings", "UNASSIGNED"]
        for patch_name in wall_patches:
            pattern = rf'({patch_name}\s*\{{\s*type\s+)patch(;)'
            content = re.sub(pattern, r'\1wall\2', content)

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

        print(f"  Mesh generated and exported to: {case_dir}")

    finally:
        gmsh.finalize()


def analyze_gmsh(case_dir: Path) -> dict:
    """Analyze Gmsh mesh quality."""
    result = {}
    try:
        gmsh.initialize()
        gmsh.model.add(f"analyze_{case_dir.name}")
        msh_path = case_dir / "mesh.msh"
        if msh_path.exists():
            gmsh.open(str(msh_path))
        else:
            result["gmsh_skipped"] = True
            return result

        analyzer = GmshQualityAnalyzer(QualityThresholds(
            min_sicn=0.20, min_sj=0.30, min_sige=0.30,
            gamma_min=0.30, min_volume=1e-10, max_aspect_ratio=10.0
        ))
        report = analyzer.analyze()

        result["cells"] = report.cells
        result["nodes"] = report.nodes
        result["tetrahedra"] = report.tetrahedra
        result["surface_triangles"] = report.surface_triangles

        if report.volume_metrics:
            sicn_stats = report.volume_metrics.get("min_sicn", {})
            result["gmsh_min_sicn"] = sicn_stats.get("min", 0.0)
            result["gmsh_p01_sicn"] = sicn_stats.get("P01", 0.0)
            result["gmsh_p50_sicn"] = sicn_stats.get("P50", 0.0)
            result["gmsh_p99_sicn"] = sicn_stats.get("P99", 0.0)

            sj_stats = report.volume_metrics.get("min_sj", {})
            result["gmsh_min_sj"] = sj_stats.get("min", 0.0)

            vol_stats = report.volume_metrics.get("volume", {})
            result["gmsh_min_volume"] = vol_stats.get("min", 0.0)

        if report.surface_metrics:
            angle_stats = report.surface_metrics.get("min_angle", {})
            result["gmsh_min_angle"] = angle_stats.get("min", 0.0)
            result["gmsh_max_angle"] = report.surface_metrics.get("max_angle", {}).get("max", 0.0)

        result["gmsh_passed"] = report.gmsh_passed
        result["gmsh_bad_count"] = len(report.bad_elements)

        analyzer.export_quality_vtk(case_dir / "mesh_quality.vtu")
        if report.bad_elements:
            analyzer.export_bad_elements_vtk(case_dir / "bad_elements.vtu")

    except Exception as e:
        result["gmsh_error"] = str(e)
    finally:
        gmsh.finalize()

    return result


def analyze_mesh_features(case_dir: Path) -> dict:
    """Detect micro-features in the mesh (small elements, slivers, etc.)."""
    result = {}
    try:
        gmsh.initialize()
        gmsh.model.add(f"features_{case_dir.name}")
        msh_path = case_dir / "mesh.msh"
        if not msh_path.exists():
            result["features_skipped"] = True
            return result
        gmsh.open(str(msh_path))

        from foampilot.mesh.adaptation.mesh_feature_detector import MeshFeatureDetector, MeshFeatureThresholds
        detector = MeshFeatureDetector(MeshFeatureThresholds(
            min_triangle_area=0.001,
            min_tet_volume=1e-6,
            max_aspect_ratio=15.0,
        ))
        report = detector.analyze()

        result["small_triangles"] = report.small_triangles
        result["small_tets"] = report.small_tets
        result["high_aspect_ratio_tets"] = report.high_aspect_ratio_tets
        result["high_aspect_ratio_tris"] = report.high_aspect_ratio_tris
        result["sliver_tets"] = report.sliver_tets
        result["mesh_feature_passed"] = (
            report.small_triangles == 0 and
            report.small_tets == 0 and
            report.sliver_tets == 0
        )

        if report.problematic_elements:
            detector.export_problematic_surfaces(case_dir / "problematic_surfaces.vtu")
            detector.export_problematic_edges(case_dir / "problematic_edges.vtu")

    except Exception as e:
        result["features_error"] = str(e)
    finally:
        gmsh.finalize()

    return result


def analyze_openfoam(case_dir: Path) -> dict:
    """Run checkMesh and parse results."""
    result = {}
    try:
        analyzer = OpenFOAMQualityAnalyzer(case_dir)
        foam_report = analyzer.analyze()
        metrics = foam_report.get("metrics", {})

        result["max_non_ortho"] = metrics.get("max_non_orthogonality", 0.0)
        result["faces_over_65"] = metrics.get("faces_above_65", 0)
        result["faces_over_70"] = metrics.get("faces_above_70", 0)
        result["max_skewness"] = metrics.get("max_skewness", 0.0)
        result["min_volume"] = metrics.get("min_volume", 0.0)
        result["max_aspect_ratio"] = metrics.get("max_aspect_ratio", 0.0)
        result["mesh_score"] = foam_report.get("mesh_score", {}).get("total", 0.0)
        result["foam_status"] = foam_report.get("gate", {}).get("status", "UNKNOWN")

    except Exception as e:
        result["foam_error"] = str(e)

    return result


def run_cfd(case_dir: Path, nb_proc: int = 2, max_iterations: int = 500) -> dict:
    """Run simpleFoam and collect convergence metrics using SimulationReport."""
    result = {}
    try:
        import subprocess
        from foampilot.solver import Solver

        solver = Solver(case_dir)
        solver.system.controlDict.endTime = max_iterations
        solver.system.fvSolution.SIMPLE["nNonOrthogonalCorrectors"] = "5"
        solver.system.write()

        start = time.time()
        log_file = case_dir / "log.incompressibleFluid"

        proc = subprocess.run(
            ["mpirun", "-np", str(nb_proc), "simpleFoam", "-case", str(case_dir)],
            capture_output=True, text=True, timeout=600
        )
        result["cfd_return_code"] = proc.returncode
        result["cfd_time"] = time.time() - start

        if proc.returncode == 0 or (case_dir / str(max_iterations)).exists():
            result["converged"] = True
            report = SimulationReport(case_dir)
            report._parse_log()
            conv_report = report.get_convergence_report()
            result["final_residuals"] = report._extract_final_residuals()
            result["convergence_state"] = conv_report.get("state", "unknown")
            result["multi_criteria_converged"] = conv_report.get("multi_criteria_converged", False)
            result["cd"] = conv_report.get("cd", 0.0)
            result["cl"] = conv_report.get("cl", 0.0)
            result["mean_u"] = conv_report.get("mean_u", 0.0)
        else:
            result["converged"] = False
            result["cfd_error"] = proc.stderr[-500:] if proc.stderr else "Unknown"

    except subprocess.TimeoutExpired:
        result["converged"] = False
        result["cfd_error"] = "Timeout"
    except Exception as e:
        result["converged"] = False
        result["cfd_error"] = str(e)

    return result


def write_csv(results: dict, csv_path: Path):
    """Write experiment results as CSV."""
    import csv
    configs = results.get("configs", {})
    if not configs:
        return

    fieldnames = ["config_name", "description", "cells", "nodes", "gmsh_min_sicn",
                  "gmsh_p01_sicn", "gmsh_p50_sicn", "gmsh_p99_sicn", "gmsh_min_sj",
                  "max_non_ortho", "faces_over_65", "faces_over_70", "max_skewness",
                  "min_volume", "max_aspect_ratio", "mesh_score", "converged",
                  "cfd_time", "status"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for config_name, data in configs.items():
            writer.writerow(data)


def main():
    parser = argparse.ArgumentParser(description="Mesh experiment matrix")
    parser.add_argument("--base-case", type=Path, required=True,
                        help="Base case directory with geometry and config")
    parser.add_argument("--output", type=Path, default=Path("experiments"),
                        help="Output directory for experiments")
    parser.add_argument("--max-cells", type=int, default=1_500_000,
                        help="Maximum cell count budget")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to test (default: all)")
    args = parser.parse_args()

    if args.configs:
        configs = {k: v for k, v in MESH_CONFIGS.items() if k in args.configs}
    else:
        configs = MESH_CONFIGS

    run_mesh_experiment(args.base_case, args.output, args.max_cells)


if __name__ == "__main__":
    main()
