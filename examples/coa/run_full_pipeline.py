#!/usr/bin/env python3
"""
Full TBAD pipeline: NIfTI → STL → CAD → Mesh → OpenFOAM.
Generates a complete, runnable OpenFOAM case from medical images.
"""
import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import gmsh
import trimesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "patient_id": 58,
    "data_dir": "imageTBAD",
    "output_dir": "pipeline_output",
    "centerline_spacing_mm": 2.0,
    "mesh": {
        "lc_min": 0.5,
        "lc_max": 4.0,
        "boundary_layers": 3,
        "boundary_layer_factor": 0.5,
        "decimate": False,
        "target_faces": 50000,
    },
    "fluid": {
        "name": "Blood",
        "rho": 1060,
        "nu": 3.77e-6,
        "non_newtonian": False,
    },
    "rheology": {
        "model": "Newtonian",
        "carreau_yasuda": {
            "nu0": 13.96e-6,
            "nuInf": 3.77e-6,
            "lambda": 12.3,
            "a": 0.6,
            "n": 0.216,
        },
    },
    "solver": {
        "turbulence": "laminar",
        "transient": False,
        "endTime": 1,
    },
}


def load_config() -> dict:
    cfg_path = Path.cwd() / "tbad_pipeline_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()


# ============================================================================
# STEP 1: Extract STL from NIfTI
# ============================================================================

def step1_extract_stl(patient_id: int, data_dir: Path, output_dir: Path, config: dict = None) -> dict:
    """Extract TL, FL, wall STL from NIfTI images.

    Optionally decimates STL files with pyfqmr when config['mesh']['decimate']
    is True.
    """
    from data_preproc.extract_tbad_full import extract_lumens, extract_aorta_wall

    image_path = data_dir / f"{patient_id}_image.nii.gz"
    label_path = data_dir / f"{patient_id}_label.nii.gz"

    if not image_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Patient {patient_id}: missing NIfTI files")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Step 1/4] Extracting STL for patient {patient_id}...")

    mc = (config or {}).get("mesh", {})
    target_tl = 30000 if mc.get("decimate") else 50000
    target_fl = 20000 if mc.get("decimate") else 30000

    lumens = extract_lumens(label_path, output_dir, target_tl=target_tl, target_fl=target_fl, verbose=False)
    if not lumens["success"]:
        raise RuntimeError("Lumen extraction failed")

    wall_path = output_dir / f"patient{patient_id}_wall.stl"
    wall_stats = extract_aorta_wall(image_path, label_path, wall_path)

    tl_path = output_dir / "tbad_TL_walls.stl"
    fl_path = output_dir / "tbad_FL_walls.stl"

    # Optional decimation with pyfqmr
    decimated = {}
    if mc.get("decimate", False):
        from mesh_utils import decimate_stl
        target = mc.get("target_faces", 50000)
        for name, path in [("TL", tl_path), ("FL", fl_path)]:
            dec_path = output_dir / f"tbad_{name}_walls_decimated.stl"
            info = decimate_stl(path, dec_path, target_faces=target)
            decimated[name] = info
            logger.info(f"  {name} decimation: {info['original_faces']:,} → {info.get('decimated_faces', info['original_faces']):,} faces")

    logger.info(f"  TL: {tl_path.name} ({lumens['tl_stats']['faces']:,} faces)")
    logger.info(f"  FL: {fl_path.name} ({lumens['fl_stats']['faces']:,} faces)")
    logger.info(f"  Wall: {wall_path.name} ({wall_stats['faces']:,} faces)")

    return {
        "patient_id": patient_id,
        "tl_stl": tl_path,
        "fl_stl": fl_path,
        "wall_stl": wall_path,
        "output_dir": output_dir,
        "decimated": decimated,
    }


# ============================================================================
# STEP 2: CAD Reconstruction
# ============================================================================

def step2_cad_reconstruction(stl_path: Path, case_dir: Path, config: dict) -> dict:
    """CAD reconstruction: centerlines → sections → B-splines → loft."""
    from cad_reconstruction.cad_reconstruction import CADReconstruction

    logger.info(f"[Step 2/4] CAD reconstruction from {stl_path.name}...")

    recon = CADReconstruction(case_dir=case_dir, centerline_spacing_mm=config["centerline_spacing_mm"])
    result = recon.run(stl_path)

    logger.info(f"  Centerlines: {result['centerline_points']} points")
    logger.info(f"  Sections: {result['sections']}")
    logger.info(f"  Loft: {result['loft_result'].get('loft', 'N/A')}")

    result["centerline_file"] = str(case_dir / "centerline.npy")

    return result


# ============================================================================
# STEP 3: Volume Meshing via snappyHexMesh (purpose-built for STL surfaces)
# ============================================================================

def step3_mesh_gmsh(case_dir: Path, stl_path: Path, config: dict, centerline: np.ndarray = None) -> dict:
    """Generate volume mesh with boundary layers using snappyHexMesh.

    Medical STL surfaces have poor triangle quality that breaks Gmsh's OCC geometry
    creation. snappyHexMesh is purpose-built for these STL files — it meshes the
    interior of closed triangulated surfaces without requiring CAD geometry.
    """
    logger.info("[Step 3/4] Volume meshing with snappyHexMesh...")

    import shutil
    from mesh_utils import run_checkmesh
    from foampilot import Meshing, ValueWithUnit

    mc = config["mesh"]
    max_faces = mc.get("target_faces", 5000)

    try:
        MM_TO_M = 0.001

        raw_mesh = trimesh.load(str(stl_path), process=True)
        if not isinstance(raw_mesh, trimesh.Trimesh):
            raw_mesh = raw_mesh.dump(concatenate=True)
        raw_mesh = raw_mesh.process(True)
        raw_mesh.merge_vertices()
        raw_mesh.fill_holes()
        raw_mesh.fix_normals()

        # Ensure normals face outward
        centroid = raw_mesh.centroid
        face_centroids = raw_mesh.triangles_center
        vec_to_face = face_centroids - centroid
        normal_dot = np.sum(vec_to_face * raw_mesh.face_normals, axis=1)
        if np.mean(normal_dot) < 0:
            logger.info("  Inverting inward-facing normals")
            raw_mesh.invert()

        if not raw_mesh.is_watertight:
            logger.warning("  STL mesh is not watertight — attempting repair")
            raw_mesh.fill_holes()
            raw_mesh.fix_normals()

        raw_mesh.vertices *= MM_TO_M
        logger.info(f"  STL: {len(raw_mesh.faces):,} faces, watertight={raw_mesh.is_watertight}")

        tri_surf = case_dir / "constant" / "triSurface"
        tri_surf.mkdir(parents=True, exist_ok=True)
        cleaned_stl = tri_surf / stl_path.name
        raw_mesh.export(str(cleaned_stl))

        min_edge = float(raw_mesh.edges_unique_length.min()) if len(raw_mesh.edges_unique_length) > 0 else 0.0
        logger.info(f"  STL quality: {len(raw_mesh.faces):,} faces, min_edge={min_edge:.4f} m, watertight={raw_mesh.is_watertight}")

        if centerline is not None and len(centerline) > 0:
            cl = np.asarray(centerline)
            center = cl[len(cl) // 2]
            if np.max(np.abs(center)) > 100:
                center = center * MM_TO_M
            logger.info(f"  Using centerline midpoint as locationInMesh: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
        else:
            center = raw_mesh.centroid
        location_in_mesh = [float(x) for x in center]

        bbox = raw_mesh.bounds
        vessel_diameter = float(np.max(bbox[1] - bbox[0]))

        lc_min = mc.get("lc_min", 0.0005)
        lc_max = mc.get("lc_max", 0.004)

        # --- Set up minimal OpenFOAM case files for snappyHexMesh ---
        from foampilot import Solver
        solver = Solver(case_dir)
        solver.compressible = False
        solver.with_gravity = False
        solver.turbulence_model = mc.get("turbulence_model", "laminar")
        solver.constant.transportProperties.nu = ValueWithUnit(3.77e-6, "m^2/s")
        solver.system.controlDict.application = "foamRun"
        solver.system.controlDict.startTime = 0
        solver.system.controlDict.endTime = 1
        solver.system.controlDict.deltaT = 1
        solver.system.controlDict.writeInterval = 1
        solver.system.controlDict.adjustTimeStep = False
        solver.system.write()  # noqa: E731
        solver.constant.write()

        # --- Configure snappyHexMesh ---
        mesher = Meshing(case_dir, mesher="snappy")
        snappy = mesher.mesher
        snappy.stl_file = stl_path.name
        snappy.locationInMesh = location_in_mesh
        snappy.geometry = {
            "wall_aorta": {
                "type": "triSurfaceMesh",
                "file": stl_path.name,
                "name": "wall_aorta",
            }
        }

        snappy.castellatedMeshControls["maxLocalCells"] = 1000000
        snappy.castellatedMeshControls["maxGlobalCells"] = 20000000

        snappy.castellatedMeshControls["refinementSurfaces"] = {
            "wall_aorta": {"level": (0, 0)}
        }
        snappy.castellatedMeshControls["locationInMesh"] = location_in_mesh

        snappy.snapControls["tolerance"] = 4.0

        n_layers = mc.get("boundary_layers", 0)
        snappy.addLayers = n_layers > 0
        if snappy.addLayers:
            snappy.add_layer("wall_aorta", n_layers)
            snappy.addLayersControls["finalLayerThickness"] = mc.get("boundary_layer_factor", 0.5)
            snappy.addLayersControls["expansionRatio"] = 1.2

        mesher.write()
        snappy.write_block_mesh_dict(padding=0.5, base_cell_size=lc_max)
        logger.info("  Running snappyHexMesh (blockMesh + surfaceFeatureExtract + snappyHexMesh)...")
        snappy.run()
        logger.info("  snappyHexMesh completed")

        # Run checkMesh validation
        checkmesh_result = run_checkmesh(case_dir)
        if checkmesh_result:
            if checkmesh_result.get("passed"):
                logger.info("  checkMesh: PASSED")
            else:
                logger.warning(f"  checkMesh: FAILED - {checkmesh_result.get('errors', [])}")

        n_cells = checkmesh_result.get("metrics", {}).get("n_cells", 0) if checkmesh_result else 0

        return {
            "nodes": n_cells,
            "elements": n_cells,
            "mesh_file": str(case_dir / "constant" / "polyMesh"),
            "mesh_dir": str(case_dir / "constant" / "polyMesh"),
            "checkmesh": checkmesh_result if checkmesh_result else {},
        }
    except Exception as exc:
        logger.error(f"  snappyHexMesh failed: {exc}")
        raise
    finally:
        try:
            import gmsh
            gmsh.finalize()
        except Exception:
            pass


# ============================================================================
# STEP 4: OpenFOAM Case Setup
# ============================================================================

def step4_openfoam_case(case_dir: Path, config: dict, mesh_file: Path = None, centerline: np.ndarray = None):
    """Create complete OpenFOAM case with boundary conditions.

    Supports both Newtonian and non-Newtonian (Carreau-Yasuda) blood models.
    """
    logger.info("[Step 4/4] Building OpenFOAM case...")

    from foampilot import ValueWithUnit, Solver
    from foampilot.constant.transportPropertiesFile import NonNewtonianModels

    fluid = config["fluid"]
    solver_cfg = config["solver"]
    rheology = config.get("rheology", {})

    # Verify mesh exists
    if mesh_file is None or not mesh_file.exists():
        candidates = [
            case_dir / "mesh.msh",
            case_dir / "constant" / "polyMesh",
            Path(config["output_dir"]) / f"patient{config['patient_id']}" / "mesh" / "mesh.msh",
        ]
        for candidate in candidates:
            if candidate.exists():
                mesh_file = candidate
                logger.info(f"  Using existing mesh: {mesh_file}")
                break
        else:
            raise FileNotFoundError(f"No mesh found for OpenFOAM case. Expected: {mesh_file}")

    logger.info(f"  Mesh file: {mesh_file}")
    logger.info(f"  Mesh exists: {mesh_file.exists()}")
    if mesh_file.is_dir():
        n_cells = 0
        boundary = mesh_file / "boundary"
        if boundary.exists():
            content = boundary.read_text()
            m = re.search(r'nCells\s+(\d+)', content)
            if m:
                n_cells = int(m.group(1))
        logger.info(f"  Mesh cells: {n_cells}")
    else:
        logger.info(f"  Mesh size: {mesh_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Setup solver
    rho = ValueWithUnit(fluid["rho"], "kg/m^3")
    nu = ValueWithUnit(fluid["nu"], "m^2/s")

    solver = Solver(case_dir)
    solver.compressible = False
    solver.with_gravity = False
    solver.transient = solver_cfg["transient"]
    solver.turbulence_model = solver_cfg["turbulence"]

    # Configure transport properties
    use_non_newtonian = fluid.get("non_newtonian", False) or rheology.get("model", "Newtonian") != "Newtonian"

    if use_non_newtonian:
        model = rheology.get("model", "CarreauYasuda")
        logger.info(f"  Fluid model: {model} (non-Newtonian blood)")

        if model == "CarreauYasuda":
            cy = rheology.get("carreau_yasuda", {})
            solver.constant.transportProperties.set_non_newtonian(
                model=NonNewtonianModels.CARREAU_YASUDA,
                rho=fluid["rho"],
                nu0=cy.get("nu0", 13.96e-6),
                nuInf=cy.get("nuInf", 3.77e-6),
                **{"lambda": cy.get("lambda", 12.3)},
                n=cy.get("n", 0.216),
                a=cy.get("a", 0.6),
            )
        else:
            solver.constant.transportProperties.set_non_newtonian(
                model=model,
                rho=fluid["rho"],
                **rheology.get("coeffs", {}),
            )
    else:
        logger.info("  Fluid model: Newtonian (blood)")
        solver.constant.transportProperties.nu = nu
        solver.constant.transportProperties.rho = rho

    solver.system.controlDict.application = "foamRun"
    solver.system.controlDict.startTime = 0
    solver.system.controlDict.endTime = solver_cfg["endTime"]
    solver.system.controlDict.deltaT = 1
    solver.system.controlDict.writeInterval = solver_cfg["endTime"]

    solver.system.fvSolution.solvers["p"]["solver"] = "smoothSolver"
    solver.system.fvSolution.solvers["p"]["smoother"] = "GAMG"
    solver.system.fvSolution.solvers["U"]["solver"] = "smoothSolver"

    solver.system.write()
    solver.constant.write()

    # Verify case structure
    logger.info("  Verifying OpenFOAM case structure...")
    required_dirs = ["0", "constant", "system"]
    for d in required_dirs:
        dpath = case_dir / d
        if not dpath.exists():
            dpath.mkdir(parents=True, exist_ok=True)
            logger.info(f"    Created: {dpath}")
        else:
            logger.info(f"    Exists: {dpath}")

    # Boundary conditions
    solver.boundary.initialize_boundary()
    solver.boundary.fields["U"]["default"] = {"type": "noSlip"}
    solver.boundary.fields["p"]["default"] = {"type": "zeroGradient"}
    solver.boundary.write_boundary_conditions()

    logger.info(f"  OpenFOAM case ready: {case_dir}")
    logger.info(f"  To run: foamRun -case {case_dir}")
    return case_dir


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(patient_id: int, config: dict, skip_steps: list = None):
    """Run complete pipeline for one patient."""
    skip_steps = skip_steps or []

    data_dir = Path(config["data_dir"])
    output_base = Path(config["output_dir"])
    output_base.mkdir(parents=True, exist_ok=True)

    patient_dir = output_base / f"patient{patient_id}"
    cad_dir = patient_dir / "cad"
    mesh_dir = patient_dir / "mesh"
    of_dir = patient_dir / "openfoam"

    results = {"patient_id": patient_id}
    mesh_file = None
    centerline = None

    # Step 1
    if 1 not in skip_steps:
        stl_info = step1_extract_stl(patient_id, data_dir, patient_dir, config)
        tl_stl = stl_info["tl_stl"]
        wall_stl = stl_info["wall_stl"]
        results["stl"] = {k: str(v) for k, v in stl_info.items() if isinstance(v, Path)}
    else:
        tl_stl = patient_dir / "tbad_TL_walls.stl"
        wall_stl = patient_dir / f"patient{patient_id}_wall.stl"
        if not tl_stl.exists():
            raise FileNotFoundError(f"Step 1 required: {tl_stl}")

    # Step 2
    if 2 not in skip_steps:
        cad_result = step2_cad_reconstruction(tl_stl, cad_dir, config)
        results["cad"] = cad_result
        # Load centerline for adaptive mesh sizing in step 3
        cl_file = Path(cad_result.get("centerline_file", ""))
        if cl_file.exists():
            centerline = np.load(cl_file)
    else:
        cad_dir = patient_dir / "cad"
        cl_file = cad_dir / "centerline.npy"
        if cl_file.exists():
            centerline = np.load(cl_file)

    # Step 3
    if 3 not in skip_steps:
        mesh_result = step3_mesh_gmsh(mesh_dir, tl_stl, config, centerline=centerline)
        mesh_file = Path(mesh_result["mesh_file"])
        results["mesh"] = mesh_result
    else:
        mesh_dir = patient_dir / "mesh"
        mesh_file = mesh_dir / "mesh.msh"
    
    # Step 4
    if 4 not in skip_steps:
        of_result = step4_openfoam_case(of_dir, config, mesh_file=mesh_file, centerline=centerline)
        results["openfoam"] = str(of_dir)
    else:
        of_dir = patient_dir / "openfoam"
    
    # Save summary
    summary_path = patient_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Pipeline complete for patient {patient_id}")
    logger.info(f"  STL:      {patient_dir}")
    logger.info(f"  CAD:      {cad_dir}")
    logger.info(f"  Mesh:     {mesh_dir}")
    logger.info(f"  OpenFOAM: {of_dir}")
    logger.info(f"  Summary:  {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Full TBAD → OpenFOAM pipeline")
    parser.add_argument("--patient", type=int, default=DEFAULT_CONFIG["patient_id"],
                        help="Patient ID")
    parser.add_argument("--data-dir", default=DEFAULT_CONFIG["data_dir"],
                        help="NIfTI directory")
    parser.add_argument("--output", default=DEFAULT_CONFIG["output_dir"],
                        help="Output base directory")
    parser.add_argument("--skip", nargs="*", type=int, default=[],
                        help="Skip steps (1-4)")
    parser.add_argument("--lc-min", type=float, default=DEFAULT_CONFIG["mesh"]["lc_min"],
                        help="Min mesh size")
    parser.add_argument("--lc-max", type=float, default=DEFAULT_CONFIG["mesh"]["lc_max"],
                        help="Max mesh size")
    parser.add_argument("--layers", type=int, default=DEFAULT_CONFIG["mesh"]["boundary_layers"],
                        help="Boundary layers")
    parser.add_argument("--decimate", action="store_true",
                        help="Decimate STL files with pyfqmr (target_faces)")
    parser.add_argument("--target-faces", type=int, default=DEFAULT_CONFIG["mesh"]["target_faces"],
                        help="Target faces after STL decimation")
    parser.add_argument("--non-newtonian", action="store_true",
                        help="Use Carreau-Yasuda non-Newtonian blood model")
    parser.add_argument("--mesh-only", action="store_true",
                        help="Only run mesh generation (skip OpenFOAM case)")
    parser.add_argument("--of-only", action="store_true",
                        help="Only run OpenFOAM case setup (skip mesh generation)")
    args = parser.parse_args()
    
    config = load_config()
    config["data_dir"] = args.data_dir
    config["output_dir"] = args.output
    config["mesh"]["lc_min"] = args.lc_min
    config["mesh"]["lc_max"] = args.lc_max
    config["mesh"]["boundary_layers"] = args.layers
    config["mesh"]["decimate"] = args.decimate
    config["mesh"]["target_faces"] = args.target_faces

    if args.non_newtonian:
        config["fluid"]["non_newtonian"] = True
        config["rheology"]["model"] = "CarreauYasuda"
    
    skip_steps = list(args.skip)
    if args.mesh_only:
        skip_steps = [4]
    elif args.of_only:
        skip_steps = [1, 2, 3]
    
    start = time.time()
    try:
        result = run_pipeline(args.patient, config, skip_steps=skip_steps)
        elapsed = time.time() - start
        logger.info(f"Done in {elapsed:.1f}s")
        if "openfoam" in result:
            print(f"\nFinal case: {result['openfoam']}")
        elif "mesh" in result:
            print(f"\nMesh file: {result['mesh']['mesh_file']}")
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
