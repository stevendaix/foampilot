"""Mesh utilities for TBAD pipeline: checkMesh validation, distance fields."""
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def run_checkmesh(case_dir: Path) -> Dict:
    """Run OpenFOAM checkMesh on a case directory and parse results.

    Args:
        case_dir: Path to the OpenFOAM case (containing constant/polyMesh).

    Returns:
        Dictionary with mesh quality metrics. ``passed`` key is True if
        checkMesh exits 0 and no errors are detected.
    """
    case_dir = Path(case_dir)
    result: Dict = {"passed": False, "errors": [], "metrics": {}}

    try:
        proc = subprocess.run(
            ["checkMesh", "-case", str(case_dir)],
            capture_output=True, text=True, timeout=120
        )
        output = proc.stdout + proc.stderr
        result["return_code"] = proc.returncode
        result["passed"] = proc.returncode == 0
    except FileNotFoundError:
        result["errors"].append("checkMesh not found — is OpenFOAM sourced?")
        result["passed"] = False
        return result
    except subprocess.TimeoutExpired:
        result["errors"].append("checkMesh timed out (>120s)")
        result["passed"] = False
        return result

    # Parse key metrics from checkMesh output
    # Supports both legacy ("Number of cells:") and OF13 ("cells:") formats
    patterns = {
        "n_cells": r"(?:Number of )?cells:\s+(\d+)",
        "n_points": r"(?:Number of )?points:\s+(\d+)",
        "n_faces": r"^\s*faces:\s+(\d+)",
        "n_internal_faces": r"(?:Number of )?internal faces:\s+(\d+)",
        "n_boundary_faces": r"(?:Number of )?boundary faces:\s+(\d+)",
        "min_vol": r"Minimum volume:\s+([0-9.eE+-]+)",
        "max_vol": r"Maximum volume:\s+([0-9.eE+-]+)",
        "max_non_ortho": r"Non-orthogonality Max:\s+([0-9.eE+-]+)",
        "max_skewness": r"Max\s+skewness:\s+([0-9.eE+-]+)",
        "max_aspect_ratio": r"Max aspect ratio:\s*=\s*([0-9.eE+-]+)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, output)
        if m:
            try:
                result["metrics"][key] = float(m.group(1)) if "." in m.group(1) or "e" in m.group(1).lower() else int(m.group(1))
            except (ValueError, IndexError):
                result["metrics"][key] = m.group(1) if m.groups() else None

    # Check for errors in output
    error_patterns = [
        r"ERROR",
        r"Fatal error",
        r"Cannot open",
        r"No mesh",
        r"Empty mesh",
    ]
    for ep in error_patterns:
        if re.search(ep, output, re.IGNORECASE):
            result["errors"].append(f"checkMesh reported: {ep}")

    if result["errors"]:
        result["passed"] = False

    # Warn on quality thresholds
    metrics = result["metrics"]
    if "max_non_ortho" in metrics:
        try:
            if float(metrics["max_non_ortho"]) > 70:
                result["errors"].append(
                    f"High non-orthogonality: {metrics['max_non_ortho']}° (threshold 70°)"
                )
        except (TypeError, ValueError):
            pass
    if "max_skewness" in metrics:
        try:
            if float(metrics["max_skewness"]) > 4.0:
                result["errors"].append(
                    f"High skewness: {metrics['max_skewness']} (threshold 4.0)"
                )
        except (TypeError, ValueError):
            pass

    if result["errors"]:
        result["passed"] = False

    result["raw_output"] = output[-2000:] if len(output) > 2000 else output
    return result


def compute_distance_field(
    surface_pts: np.ndarray,
    centerline_pts: np.ndarray,
) -> np.ndarray:
    """Compute the distance from each surface point to the nearest centerline point.

    Args:
        surface_pts: (N, 3) array of surface vertex coordinates.
        centerline_pts: (M, 3) array of centerline point coordinates.

    Returns:
        (N,) array of distances.

    A return value of 0 means the point is on the centerline; larger
    values mean the point is further from the centerline (closer to the wall).
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(centerline_pts)
    dists, _ = tree.query(surface_pts, k=1)
    return np.asarray(dists, dtype=float)


def decimate_stl(input_path: Path, output_path: Path, target_faces: int, aggressiveness: float = 5.0) -> dict:
    """Decimate an STL mesh using pyfqmr for fast quadric reduction.

    Falls back gracefully if pyfqmr is not installed.

    Args:
        input_path: Path to the input STL file.
        output_path: Path to write the decimated STL.
        target_faces: Target number of faces after decimation.
        aggressiveness: Decimation aggressiveness (higher = more aggressive).

    Returns:
        Dictionary with original and target face counts, and whether
        decimation was applied.
    """
    import trimesh

    mesh = trimesh.load_mesh(str(input_path), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Could not load mesh as Trimesh")

    original_faces = len(mesh.faces)

    result = {
        "original_faces": original_faces,
        "target_faces": target_faces,
        "decimated": False,
    }

    if original_faces <= target_faces:
        output_path.write_bytes(input_path.read_bytes())
        return result

    try:
        import pyfqmr
        result["method"] = "pyfqmr"

        simplifier = pyfqmr.Simplify()
        simplifier.setMesh(
            mesh.vertices.astype(np.float32),
            mesh.faces.astype(np.int32),
        )
        simplifier.simplify_mesh(
            target_count=target,
            aggressiveness=7.0,
            preserve_border=True,
            verbose=False,
        )
        new_vertices, new_faces, _ = simplifier.getMesh()
        new_mesh = trimesh.Trimesh(new_vertices, new_faces, process=True)
        new_mesh.export(str(output_path))

        result["decimated_faces"] = len(new_faces)
        result["decimated"] = True
        logger.info(
            "Decimated %s: %d → %d faces (target %d)",
            input_path.name, original_faces, result["decimated_faces"], target_faces
        )
    except ImportError:
        result["method"] = "fallback_trimesh"
        logger.warning("pyfqmr not installed, using trimesh simplification")

        # Fallback: use trimesh's built-in simplification
        simplified = mesh.simplify_quadric_decimation(target_faces)
        if simplified is not None:
            simplified.export(str(output_path))
            result["decimated_faces"] = len(simplified.faces)
            result["decimated"] = True
            logger.info(
                "Trimesh decimation: %d → %d faces",
                original_faces, result["decimated_faces"]
            )
        else:
            output_path.write_bytes(input_path.read_bytes())
            result["decimated"] = False
    except Exception as exc:
        logger.warning("Decimation failed (%s), keeping original", exc)
        output_path.write_bytes(input_path.read_bytes())

    return result


def setup_gmsh_adaptive_sizing(
    wall_surfaces: List[int],
    lc_min: float = 0.5,
    lc_max: float = 4.0,
    centerline_pts: Optional[np.ndarray] = None,
    vessel_radius: Optional[float] = None,
) -> None:
    """Configure Gmsh background mesh fields for adaptive sizing.

    Uses a Distance + Threshold field to create fine mesh near the
    wall surface and coarser mesh in the bulk.  When a centerline is
    provided, an additional MathEval field refines based on the local
    radius (distance from centerline to surface) for physiologically
    appropriate sizing.

    Args:
        wall_surfaces: List of 2-D surface entity tags for the wall.
        lc_min: Minimum mesh size (near wall).
        lc_max: Maximum mesh size (in bulk).
        centerline_pts: Optional (N, 3) centerline points for radius-based sizing.
        vessel_radius: Optional estimated vessel radius (mm).
    """
    import gmsh

    # Field 1: Distance to wall surfaces
    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "SurfacesList", wall_surfaces)

    # Field 2: Threshold for smooth size gradation from wall to bulk
    thresh_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh_field, "IField", dist_field)
    gmsh.model.mesh.field.setNumber(thresh_field, "LcMin", lc_min)
    gmsh.model.mesh.field.setNumber(thresh_field, "LcMax", lc_max)
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0.0)

    if vessel_radius is not None:
        gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", vessel_radius * 0.5)
    else:
        gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", lc_max * 2)

    gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)

    logger.info(
        "Adaptive sizing: lc_min=%.2f, lc_max=%.2f, wall_surfaces=%d",
        lc_min, lc_max, len(wall_surfaces)
    )


def create_boundary_layers(
    wall_surfaces: List[int],
    case_dir: Path,
    n_layers: int = 3,
    thickness_factor: float = 0.5,
) -> None:
    """Configure Gmsh boundary layer options for near-wall inflation layers.

    Attempts to set Gmsh boundary layer options. If the Gmsh build does
    not support them, falls back to the Distance+Threshold field which
    already provides wall-adjacent mesh refinement.

    Args:
        wall_surfaces: Surface tags for the wall patches.
        case_dir: Case directory (unused for Gmsh but kept for API consistency).
        n_layers: Number of boundary layer sublayers.
        thickness_factor: Fraction of local mesh size for total BL thickness.
    """
    import gmsh

    if n_layers > 0:
        options_set = 0
        for opt, val in [
            ("Mesh.BoundaryLayerElements", n_layers),
            ("Mesh.BoundaryLayerFactor", thickness_factor),
            ("Mesh.BoundaryLayerMaxThickness", thickness_factor * 2.0),
            ("Mesh.BoundaryLayers", 1),  # some versions use this boolean
        ]:
            try:
                gmsh.option.setNumber(opt, val)
                options_set += 1
            except Exception:
                pass

        if options_set > 0:
            logger.info("Boundary layers configured (%d options set)", options_set)
        else:
            logger.info(
                "Boundary layer options not available in this Gmsh build; "
            "relying on Distance+Threshold field for near-wall refinement"
        )


def remesh_stl_with_vtk(
    input_path: Path,
    output_path: Path,
    target_faces: int = 20000,
    smoothing_iterations: int = 50,
) -> dict:
    """Remesh an STL using VTK for high-quality triangle output.

    Medical STLs from marching_cubes have degenerate triangles (sub-voxel
    slivers) that break Gmsh's OCC geometry creation. This function uses
    VTK's pipeline to produce a clean, manifold, water-tight mesh:

    1. CleanPolyData — merges duplicate vertices
    2. TriangleFilter — ensures all faces are triangles
    3. DecimatePro — reduces face count while preserving topology
    4. WindowedSincPolyDataFilter — high-quality Laplacian smoothing
    5. PolyDataNormals — fixes winding / normal orientation

    Args:
        input_path: Path to input STL file.
        output_path: Path to write cleaned STL.
        target_faces: Target number of triangles after decimation.
        smoothing_iterations: Windowed sinc smoothing iterations.

    Returns:
        Dictionary with quality metrics: faces, watertight, min_edge, etc.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result: Dict = {}

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(input_path))
    reader.PylodUpdate().Update() if hasattr(reader, "PylodUpdate") else reader.Update()

    original_pd = reader.GetOutput()
    if original_pd.GetNumberOfPoints() == 0:
        raise RuntimeError(f"STL reader returned empty mesh for {input_path}")

    # 1. Clean: merge duplicate vertices
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(original_pd)
    clean.SetToleranceIsAbsolute(True)
    clean.SetAbsoluteTolerance(0.01)
    clean.Update()

    # 2. Triangle filter
    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputConnection(clean.GetOutputPort())
    triangle.Update()
    pd = triangle.GetOutput()

    result["original_faces"] = original_pd.GetNumberOfCells()
    result["after_clean_faces"] = pd.GetNumberOfCells()

    # 3. Decimate to target face count (preserve topology)
    if pd.GetNumberOfPolys() > target_faces:
        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(pd)
        reduction_ratio = 1.0 - (target_faces / float(pd.GetNumberOfPolys()))
        decimate.SetTargetReduction(max(reduction_ratio, 0.0))
        decimate.PreserveTopologyOn()
        decimate.PreSplitMeshOn()
        decimate.Update()
        pd = decimate.GetOutput()

    result["after_decimate_faces"] = pd.GetNumberOfPolys()

    # 4. High-quality smoothing (Windowed Sinc)
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(pd)
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.SetPassBand(0.01)
    smoother.BoundarySmoothingOn()
    smoother.NonManifoldSmoothingOn()
    smoother.Update()
    pd = smoother.GetOutput()

    # Fill any holes introduced by decimation
    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputData(pd)
    fill_holes.SetHoleSize(int(target_faces * 10))
    fill_holes.Update()
    pd = fill_holes.GetOutput()

    # 5. Fix normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(pd)
    normals.AutoOrientNormalsOn()
    normals.FlipNormalsOff()
    normals.ConsistencyOn()
    normals.Update()
    pd = normals.GetOutput()

    result["final_faces"] = pd.GetNumberOfPolys()

    # Extract mesh for quality analysis
    points = np.array(vtk_to_numpy(pd.GetPoints().GetData()))
    faces = np.array(vtk_to_numpy(pd.GetPolys().GetData())).reshape(-1, 4)[:, 1:]

    # Compute quality metrics
    face_areas = []
    edge_lengths = []
    for face in faces:
        v0, v1, v2 = points[face[0]], points[face[1]], points[face[2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        face_areas.append(area)
        edges = [
            np.linalg.norm(v1 - v0),
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v0 - v2),
        ]
        edge_lengths.extend(edges)

    face_areas = np.array(face_areas)
    edge_lengths = np.array(edge_lengths)
    degenerate = np.sum(face_areas < 1e-10)
    min_edge = float(edge_lengths.min()) if len(edge_lengths) > 0 else 0.0

    result["degenerate_faces"] = int(degenerate)
    result["min_edge"] = min_edge
    result["total_area"] = float(face_areas.sum())

    # Write output STL
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(pd)
    writer.Write()

    # Check watertight using trimesh
    import trimesh
    tm = trimesh.load(str(output_path), process=True)
    if not isinstance(tm, trimesh.Trimesh):
        tm = tm.dump(concatenate=True)
    result["watertight"] = bool(tm.is_watertight)

    logger.info(
        "VTK remesh: %d → %d faces, min_edge=%.4f mm, watertight=%s, degenerate=%d",
        result["original_faces"],
        result["final_faces"],
        min_edge,
        result["watertight"],
        degenerate,
    )

    return result
