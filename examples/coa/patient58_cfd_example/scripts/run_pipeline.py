#!/usr/bin/env python3
"""
Complete CFD pipeline for patient 58 using foampilot modules.

Handles: STL processing → snappyHexMesh → patch splitting → BC setup → CFD run → post-processing.

Uses foampilot.mesh.snappymesh, foampilot.Meshing, foampilot.Solver, and foampilot.boundary.

Prerequisites:
    - OpenFOAM 13 environment sourced
    - PYTHONPATH includes foampilot/src

Usage:
    source /opt/openfoam13/etc/bashrc
    PYTHONPATH=/home/steven/foampilot/foampilot/src python3 scripts/run_pipeline.py --all
    PYTHONPATH=/home/steven/foampilot/foampilot/src python3 scripts/run_pipeline.py --mesh-only
    PYTHONPATH=/home/steven/foampilot/foampilot/src python3 scripts/run_pipeline.py --run-sim
    PYTHONPATH=/home/steven/foampilot/foampilot/src python3 scripts/run_pipeline.py --post-process
"""
import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CASE_DIR = Path(__file__).resolve().parent.parent
STL_FILE = CASE_DIR / "constant" / "triSurface" / "tbad_TL_walls.stl"
CENTERLINE_FILE = CASE_DIR / "centerline.npy"  # if available from step 2


def run_cmd(cmd, cwd=None, timeout=120):
    """Run a command and return (stdout+stderr, returncode)."""
    logger.info(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else str(CASE_DIR),
                       capture_output=True, text=True, timeout=timeout)
    output = r.stdout + r.stderr
    if r.returncode != 0:
        logger.error(f"Command failed (rc={r.returncode})")
        for line in output.split('\n'):
            if 'Error' in line or 'error' in line or 'FATAL' in line:
                logger.error(f"  {line}")
    return output, r.returncode


def step0_extract_centerline():
    """Extract centerline from original STL using CenterlineExtractor."""
    logger.info("=== Step 0: Centerline Extraction ===")
    
    if CENTERLINE_FILE.exists():
        logger.info(f"  Centerline already exists at {CENTERLINE_FILE}, skipping extraction")
        return True
    
    sys.path.insert(0, str(CASE_DIR.parent))
    from cad_reconstruction.centerline_extractor import CenterlineExtractor
    
    stl_path = CASE_DIR.parent.parent / "data_preproc" / "tbad_stl_output" / "tbad_TL_walls.stl"
    if not stl_path.exists():
        raise FileNotFoundError(f"Original STL not found: {stl_path}")
    
    extractor = CenterlineExtractor(resampling_step_mm=1.0)
    cl_mm = extractor.extract(stl_path)
    cl_meters = cl_mm * 0.001
    np.save(str(CENTERLINE_FILE), cl_meters)
    logger.info(f"  Saved centerline ({len(cl_meters)} points) to {CENTERLINE_FILE}")
    return True


def step1_prepare_stl():
    """Prepare STL: scale from mm to meters. No normal processing needed — snappyHexMesh handles it."""
    logger.info("=== Step 1: STL Preparation ===")
    
    if not STL_FILE.exists():
        # Check if original STL exists in pipeline_output
        src = CASE_DIR.parent.parent / "pipeline_output" / "patient58" / "tbad_TL_walls.stl"
        if src.exists():
            shutil = __import__('shutil')
            shutil.copy(str(src), str(STL_FILE))
            logger.info(f"  Copied ST from {src}")
        else:
            raise FileNotFoundError(f"STL not found: {STL_FILE}")
    
    # Load and scale to meters
    m = trimesh.load(str(STL_FILE), process=True)
    if not isinstance(m, trimesh.Trimesh):
        m = m.dump(concatenate=True)
    
    # Scale (already scaled in this case, but verify)
    if m.vertices.max() > 10:
        m.vertices *= 0.001
        m.export(str(STL_FILE))
        logger.info(f"  Scaled STL to meters: {len(m.faces)} faces, bounds {m.bounds}")
    else:
        logger.info(f"  STL already in meters: {len(m.faces)} faces")
    
    # Find centerline point for locationInMesh
    if CENTERLINE_FILE.exists():
        cl = np.load(str(CENTERLINE_FILE))
    else:
        cl = np.array([
            [0.282, 0.3175, 0.045],
            [0.2696, 0.2813, 0.0473],
            [0.257, 0.245, 0.0496],
        ])
    
    loc = cl[len(cl) // 4]
    if not m.contains([loc])[0]:
        logger.warning(f"  trimesh.contains() reports 25% centerline point as outside (unreliable for medical STLs)")
        inside_pts = [pt for pt in cl if m.contains([pt])[0]]
        if inside_pts:
            loc = inside_pts[0]
            logger.info(f"  Using verified inside centerline point: ({loc[0]:.4f}, {loc[1]:.4f}, {loc[2]:.4f})")
        else:
            logger.warning("  No centerline points verified inside by trimesh.contains(); using 50% point")
            loc = cl[len(cl) // 2]
    
    logger.info(f"  locationInMesh = ({loc[0]:.4f}, {loc[1]:.4f}, {loc[2]:.4f})")
    
    dict_path = CASE_DIR / "system" / "snappyHexMeshDict"
    if dict_path.exists():
        text = dict_path.read_text()
        new_line = f"    locationInMesh ({loc[0]:.6f} {loc[1]:.6f} {loc[2]:.6f});"
        text = re.sub(r'^\s*locationInMesh\s*\([^)]*\);', new_line, text, flags=re.MULTILINE)
        dict_path.write_text(text)
        logger.info(f"  Updated snappyHexMeshDict locationInMesh")
    
    return list(loc)


def step2_run_snappy(location_in_mesh):
    """Run snappyHexMesh to generate volume mesh."""
    logger.info("=== Step 2: snappyHexMesh ===")
    
    # Clean previous mesh
    import shutil
    pm = CASE_DIR / "constant" / "polyMesh"
    if pm.exists():
        shutil.rmtree(pm)
    
    # blockMesh
    out, rc = run_cmd(["blockMesh"], timeout=30)
    if rc != 0:
        return False
    
    # snappyHexMesh
    out, rc = run_cmd(["snappyHexMesh", "-overwrite", "-case", "."], timeout=120)
    logger.info(f"  snappyHexMesh: {'SUCCESS' if rc == 0 else 'FAILED'}")
    
    # checkMesh
    out, rc = run_cmd(["checkMesh"], timeout=30)
    if "Mesh OK" in out:
        logger.info("  checkMesh: PASSED")
    else:
        logger.warning("  checkMesh: issues detected")
    
    # Extract cell count
    m = re.search(r'cells:\s+(\d+)', out)
    if m:
        n_cells = int(m.group(1))
        logger.info(f"  Cells: {n_cells}")
    
    return rc == 0


def compute_face_normal(face_vertex_indices, points):
    pts = points[face_vertex_indices]
    if len(pts) < 3:
        return np.array([0.0, 0.0, 0.0])
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0])
    return normal / norm


def compute_face_area(face_vertex_indices, points):
    pts = points[face_vertex_indices]
    if len(pts) < 3:
        return 0.0
    return 0.5 * np.abs(np.sum(np.cross(pts[:-1], pts[1:])))


def build_boundary_face_data(faces, points, patches):
    centers = []
    normals = []
    areas = []
    face_indices = []

    for name, info in patches.items():
        if isinstance(info, dict):
            sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        else:
            sf, nf = info[0], info[1]
        for fi in range(sf, sf + nf):
            face = faces[fi]
            pts = points[face]
            center = pts.mean(axis=0)
            normal = compute_face_normal(face, points)
            area = compute_face_area(face, points)
            centers.append(center)
            normals.append(normal)
            areas.append(area)
            face_indices.append(fi)

    return {
        "centers": np.array(centers),
        "normals": np.array(normals),
        "areas": np.array(areas),
        "face_indices": np.array(face_indices),
    }


def build_face_adjacency(face_indices, faces, points, centers):
    pt_to_faces = {}
    for idx, fi in enumerate(face_indices):
        for pt in faces[fi]:
            pt_to_faces.setdefault(pt, []).append(idx)

    adjacency = {i: set() for i in range(len(face_indices))}
    for idx, fi in enumerate(face_indices):
        face_pts = set(faces[fi])
        for pt in face_pts:
            for nb_idx in pt_to_faces.get(pt, []):
                if nb_idx != idx and np.linalg.norm(centers[idx] - centers[nb_idx]) < 0.05:
                    adjacency[idx].add(nb_idx)
    return {k: list(v) for k, v in adjacency.items()}


def region_growing_cap(seed_indices, centers, normals, adjacency, max_angle_deg=30.0, max_dist=0.08):
    max_angle = np.radians(max_angle_deg)
    region = set(seed_indices)
    queue = list(seed_indices)
    seed_normal = normals[list(seed_indices)].mean(axis=0)
    seed_normal = seed_normal / (np.linalg.norm(seed_normal) + 1e-9)

    while queue:
        cur = queue.pop(0)
        for nb in adjacency.get(cur, []):
            if nb in region:
                continue
            if np.linalg.norm(centers[cur] - centers[nb]) > max_dist:
                continue
            angle = np.arccos(np.clip(np.abs(np.dot(normals[cur], seed_normal)), -1, 1))
            if angle < max_angle:
                region.add(nb)
                queue.append(nb)
    return region


def p04_detect_caps(reader):
    mesh = reader.mesh
    points = reader._points
    faces = reader._faces
    patches = reader.boundary_patches

    centroids = mesh.cell_centers().points
    pca = PCA(n_components=3)
    pca.fit(centroids)
    axis = pca.components_[0]
    if axis[2] > 0:
        axis = -axis
    axis = axis / np.linalg.norm(axis)

    centers = []
    normals = []
    face_indices = []

    for name, info in patches.items():
        sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        for fi in range(sf, sf + nf):
            face = faces[fi]
            pts = points[face]
            center = pts.mean(axis=0)
            normal = compute_face_normal(face, points)
            centers.append(center)
            normals.append(normal)
            face_indices.append(fi)

    centers = np.array(centers)
    normals = np.array(normals)
    face_indices = np.array(face_indices)

    adjacency = build_face_adjacency(face_indices, faces, points, centers)

    proj = centers @ axis
    sorted_idx = np.argsort(proj)
    min_seeds = sorted_idx[:5].tolist()
    max_seeds = sorted_idx[-5:].tolist()

    cap1_region = region_growing_cap(min_seeds, centers, normals, adjacency, max_angle_deg=30.0, max_dist=0.08)
    cap2_region = region_growing_cap(max_seeds, centers, normals, adjacency, max_angle_deg=30.0, max_dist=0.08)

    cap1_faces = set(face_indices[list(cap1_region)].tolist()) if len(cap1_region) >= 3 else set()
    cap2_faces = set(face_indices[list(cap2_region)].tolist()) if len(cap2_region) >= 3 else set()

    s1 = np.dot(centers[list(cap1_region)].mean(axis=0) - centers.mean(axis=0), axis) if cap1_region else 0
    s2 = np.dot(centers[list(cap2_region)].mean(axis=0) - centers.mean(axis=0), axis) if cap2_region else 0

    if s1 < s2:
        inlet_faces, outlet_faces = cap1_faces, cap2_faces
    else:
        inlet_faces, outlet_faces = cap2_faces, cap1_faces

    return {
        "inlet_faces": inlet_faces,
        "outlet_faces": outlet_faces,
        "axis": axis,
        "centers": centers,
        "face_indices": face_indices,
    }


def step3_split_patches():
    """Split boundary into INLET, OUTLET, WALL using P04 region growing."""
    logger.info("=== Step 3: Patch splitting (P04 region growing) ===")

    from foampilot.postprocess import OpenFOAMDirectReader

    reader = OpenFOAMDirectReader(case_path=CASE_DIR)
    mesh = reader.mesh
    points = reader._points
    faces = reader._faces
    owner = reader._owner

    patches = reader.boundary_patches
    logger.info(f"  Current patches: {list(patches.keys())}")

    p04 = p04_detect_caps(reader)
    inlet_faces = p04["inlet_faces"]
    outlet_faces = p04["outlet_faces"]

    all_boundary_faces = set()
    for name, info in patches.items():
        sf, nf = info.get("startFace", 0), info.get("nFaces", 0)
        for fi in range(sf, sf + nf):
            all_boundary_faces.add(fi)

    wall_faces = all_boundary_faces - inlet_faces - outlet_faces

    inlet_ids = sorted(inlet_faces)
    outlet_ids = sorted(outlet_faces)
    wall_ids = sorted(wall_faces)

    logger.info(f"  INLET: {len(inlet_ids)} faces")
    logger.info(f"  OUTLET: {len(outlet_ids)} faces")
    logger.info(f"  WALL: {len(wall_ids)} faces")

    wa_start = min(sf for sf, nf in ((info.get("startFace", 0), info.get("nFaces", 0)) for info in patches.values()))
    perm = list(range(wa_start)) + inlet_ids + outlet_ids + wall_ids
    new_faces = [faces[p] for p in perm]
    new_owner = [owner[p] for p in perm]

    n_inlet = len(inlet_ids)
    n_outlet = len(outlet_ids)
    n_wall = len(wall_ids)

    pm = CASE_DIR / "constant" / "polyMesh"

    with open(pm / "faces", 'w') as f:
        f.write('FoamFile\n{\n    format      ascii;\n    class       faceList;\n    location    "constant/polyMesh";\n    object      faces;\n}\n\n')
        f.write(f'{len(new_faces)}\n\n(\n')
        for face in new_faces:
            f.write(f'    {len(face)} ({" ".join(str(v) for v in face)})\n')
        f.write(')\n')

    with open(pm / "owner", 'w') as f:
        f.write('FoamFile\n{\n    format      ascii;\n    class       labelList;\n    location    "constant/polyMesh";\n    object      owner;\n}\n\n')
        f.write(f'{len(new_owner)}\n\n(\n')
        for val in new_owner:
            f.write(f'    {val}\n')
        f.write(')\n')

    with open(pm / "boundary", 'w') as f:
        f.write('FoamFile\n{\n    format      ascii;\n    class       polyBoundaryMesh;\n    location    "constant/polyMesh";\n    object      boundary;\n}\n\n')
        f.write('3\n(\n')
        f.write(f'    INLET\n    {{\n        type            patch;\n        nFaces          {n_inlet};\n        startFace       {wa_start};\n    }}\n')
        f.write(f'    OUTLET\n    {{\n        type            patch;\n        nFaces          {n_outlet};\n        startFace       {wa_start + n_inlet};\n    }}\n')
        f.write(f'    WALL\n    {{\n        type            wall;\n        nFaces          {n_wall};\n        startFace       {wa_start + n_inlet + n_outlet};\n    }}\n')
        f.write(')\n')

    logger.info("  Boundary file written with P04 INLET, OUTLET, WALL patches")

    out, rc = run_cmd(["checkMesh"], timeout=30)
    if "Mesh OK" in out:
        logger.info("  checkMesh: PASSED")
    return True


def step4_setup_cfd():
    """Set up OpenFOAM case with boundary conditions using foampilot.Solver."""
    logger.info("=== Step 4: OpenFOAM case setup ===")
    
    from foampilot import Solver, ValueWithUnit
    
    solver = Solver(CASE_DIR)
    solver.compressible = False
    solver.with_gravity = False
    solver.transient = False
    solver.turbulence_model = "laminar"
    
    # Transport properties (Newtonian blood model)
    solver.constant.transportProperties.nu = ValueWithUnit(3.77e-6, "m^2/s")
    
    # ControlDict
    solver.system.controlDict.application = "foamRun"
    solver.system.controlDict.startTime = 0
    solver.system.controlDict.endTime = 100
    solver.system.controlDict.deltaT = 1
    solver.system.controlDict.writeInterval = 50
    
    # fvScheme
    solver.system.fvSchemes.divSchemes.update({
        "div(phi,U)": "Gauss linearUpwind grad(U)",
        "div(phi,nuEff)": "Gauss linear",
    })
    
    # fvSolution (OF13 requires PIMPLE + explicit Final solvers)
    solver.system.fvSolution.solvers = {
        "p": {"solver": "GAMG", "tolerance": 1e-6, "relTol": 0.1},
        "pFinal": {"solver": "GAMG", "tolerance": 1e-6, "relTol": 0},
        "U": {"solver": "smoothSolver", "smoother": "GaussSeidel", "tolerance": 1e-6, "relTol": 0.1},
        "UFinal": {"solver": "smoothSolver", "smoother": "GaussSeidel", "tolerance": 1e-6, "relTol": 0},
    }
    solver.system.fvSolution.PIMPLE = {
        "momentumPredictor": True,
        "nOuterCorrectors": 1,
        "nCorrectors": 0,
        "nNonOrthogonalCorrectors": 0,
    }
    solver.system.fvSolution.relaxationFactors = {
        "p": 0.3,
        "U": 0.7,
    }
    
    solver.system.write()
    solver.constant.write()
    
    # Set up boundary conditions
    solver.boundary.initialize_boundary()
    solver.boundary.fields["U"]["default"] = {"type": "noSlip"}
    solver.boundary.fields["p"]["default"] = {"type": "zeroGradient"}
    
    # INLET: velocity inlet (0.4 m/s typical aortic flow)
    solver.boundary.fields["U"]["INLET"] = {
        "type": "fixedValue",
        "value": "uniform (0.4 0 0)",
    }
    solver.boundary.fields["p"]["INLET"] = {"type": "zeroGradient"}
    
    # OUTLET: pressure outlet
    solver.boundary.fields["U"]["OUTLET"] = {"type": "zeroGradient"}
    solver.boundary.fields["p"]["OUTLET"] = {
        "type": "fixedValue",
        "value": "uniform 0",
    }
    
    # WALL: no-slip
    solver.boundary.fields["U"]["WALL"] = {"type": "noSlip"}
    solver.boundary.fields["p"]["WALL"] = {"type": "zeroGradient"}
    
    solver.boundary.write_boundary_conditions()
    
    logger.info("  Boundary conditions: INLET (fixedValue U), OUTLET (fixedValue p), WALL (noSlip)")
    logger.info("  Case ready for simulation")
    return True


def step5_run_simulation(max_iterations=100):
    """Run the CFD simulation."""
    logger.info("=== Step 5: CFD Simulation ===")
    
    # Clean previous results
    import shutil
    for t in ["1", "10", "50", "100"]:
        td = CASE_DIR / t
        if td.exists():
            shutil.rmtree(td)
    
    log_file = CASE_DIR / "log.simpleFoam"
    proc = subprocess.Popen(
        ["simpleFoam", "-case", "."],
        cwd=str(CASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    with open(log_file, 'w') as f:
        for line in proc.stdout:
            f.write(line)
            if "Solving for U" in line or "Time =" in line:
                logger.info(f"  {line.strip()}")
    
    proc.wait()
    logger.info(f"  simpleFoam exit code: {proc.returncode}")
    
    # Check convergence
    residuals = []
    for line in log_file.read_text().split('\n'):
        m = re.search(r'Initial residual = ([\d.e+-]+)', line)
        if m:
            residuals.append(float(m.group(1)))
    
    if residuals:
        logger.info(f"  Final Ux residual: {residuals[-3]:.2e}")
        logger.info(f"  Convergence: {'YES' if residuals[-1] < 1e-5 else 'NO'}")
    
    return proc.returncode == 0


def step6_post_process():
    """Post-process CFD results using foampilot."""
    logger.info("=== Step 6: Post-processing ===")
    
    from foampilot import Meshing
    
    # Run foamLog to parse residuals
    run_cmd(["foamLog", "log.simpleFoam"], timeout=15)
    logger.info("  Parsed log.simpleFoam")
    
    # Generate VTK for visualization
    run_cmd(["foamToVTK", "-time", "50"], timeout=30)
    logger.info("  Exported VTK at time 50")
    
    # Run function objects for mass flow
    run_cmd(["foamRun", "-solver", "incompressibleFluid", "-postProcess",
             "-func", "faceFieldValue", "-time", "50"], timeout=30)
    logger.info("  Post-processed flow rates")
    
    # Calculate forces on WALL
    run_cmd(["foamRun", "-solver", "incompressibleFluid", "-postProcess",
             "-func", "forces", "-time", "50"], timeout=30)
    logger.info("  Calculated wall forces")
    
    logger.info("  Post-processing complete. Results in postProcessing/")


def main():
    parser = argparse.ArgumentParser(description="Patient 58 CFD pipeline with foampilot")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    parser.add_argument("--mesh-only", action="store_true", help="Only generate mesh")
    parser.add_argument("--run-sim", action="store_true", help="Only run CFD simulation")
    parser.add_argument("--post-process", action="store_true", help="Only post-process")
    args = parser.parse_args()
    
    if args.all or args.mesh_only:
        step0_extract_centerline()
        loc = step1_prepare_stl()
        step2_run_snappy(loc)
        step3_split_patches()
        step4_setup_cfd()
        
        if not args.mesh_only:
            step5_run_simulation()
            step6_post_process()
    
    if args.run_sim and not args.all:
        step5_run_simulation()
    
    if args.post_process:
        step6_post_process()
    
    logger.info("=== Pipeline complete ===")
    logger.info(f"Case directory: {CASE_DIR}")
    logger.info("To visualize: paraview -case .")
    return 0


if __name__ == "__main__":
    sys.exit(main())
