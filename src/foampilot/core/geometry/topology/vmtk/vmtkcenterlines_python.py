import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
import vtk
from scipy.spatial import cKDTree

from .pypes import vmtkBaseScript

logger = logging.getLogger(__name__)

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from .vmtksurfacepreprocess_local import preprocess_surface, SurfaceModel, read_polydata
from .vmtksurfacecapper_local import vmtkSurfaceCapper
from .vmtkdelaunay_local import build_delaunay, DelaunayVolume
from .vmtkinternaltetrahedra_local import classify_internal_tetrahedra, InternalTetraMesh
from .vmtkvoronoi_local import build_voronoi_from_tetrahedra, filter_voronoi_by_clearance, extract_seed_component, simplify_voronoi, VoronoiGraph
from .vmtkfastmarching_local import vmtkFastMarchingLocal, find_voronoi_seeds, Pole, Centerline, VoronoiGraph as FastMarchingVoronoiGraph
from .vmtkcenterlinegeometry_local import compute_centerline_geometry, Centerline as GeometryCenterline
from .vmtkcenterlineresampling_local import resample_centerline
from .vmtkcenterlinesections_local import vmtkCenterlineSectionsLocal
from .vmtkcenterlinesnetwork_local import build_centerline_network, CenterlineNetwork


@dataclass
class PhaseResult:
    phase: str
    elapsed: float
    warnings: List[str] = field(default_factory=list)
    data: Any = None


@dataclass
class PipelineReport:
    backend: str
    acceleration: str
    numba_available: bool
    phase_timings: Dict[str, float] = field(default_factory=dict)
    max_memory_mb: float = 0.0
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    status: str = "PASS"


def _write_polydata(pd: vtk.vtkPolyData, path: Path):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(pd)
    writer.Write()


def _write_unstructured(ug: vtk.vtkUnstructuredGrid, path: Path):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(ug)
    writer.Write()


def _centerline_to_vtp(centerline) -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    for p in centerline.points:
        pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    lines = vtk.vtkCellArray()
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(len(centerline.points))
    for i in range(len(centerline.points)):
        line.GetPointIds().SetId(i, i)
    lines.InsertNextCell(line)

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(lines)

    for arr_name, arr in [
        ("MaximumInscribedSphereRadius", centerline.radii),
        ("Abscissas", centerline.abscissas),
        ("Curvature", centerline.curvature),
        ("Torsion", centerline.torsion),
        ("Tortuosity", np.full(len(centerline.points), centerline.tortuosity, dtype=float)),
        ("FrenetTangent", centerline.frenet_tangents),
        ("ParallelTransportNormals", centerline.parallel_transport_normals),
        ("ParallelTransportBinormals", centerline.parallel_transport_binormals),
    ]:
        vtk_arr = vtk.vtkDoubleArray()
        vtk_arr.SetName(arr_name)
        if arr.ndim == 1:
            vtk_arr.SetNumberOfComponents(1)
            for v in arr:
                vtk_arr.InsertNextTuple1(float(v))
        else:
            vtk_arr.SetNumberOfComponents(3)
            for v in arr:
                vtk_arr.InsertNextTuple3(float(v[0]), float(v[1]), float(v[2]))
        pd.GetPointData().AddArray(vtk_arr)

    return pd


def _voronoi_to_vtp(voronoi: VoronoiGraph) -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    for p in voronoi.points:
        pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))

    radius_arr = vtk.vtkDoubleArray()
    radius_arr.SetName("MaximumInscribedSphereRadius")
    for r in voronoi.radii:
        radius_arr.InsertNextTuple1(float(r))

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.GetPointData().AddArray(radius_arr)

    lines = vtk.vtkCellArray()
    for e in voronoi.edges:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, int(e[0]))
        line.GetPointIds().SetId(1, int(e[1]))
        lines.InsertNextCell(line)
    pd.SetLines(lines)
    return pd


def _build_synthetic_centerline_mesh(length: float = 10.0, radius: float = 1.0, n_points: int = 100) -> trimesh.Trimesh:
    t = np.linspace(0, length, n_points)
    pts = np.column_stack([t, np.zeros(n_points), np.zeros(n_points)])
    rad = np.full(n_points, radius, dtype=float)
    return _centerline_to_mesh(pts, rad)


def _centerline_to_mesh(pts: np.ndarray, rad: np.ndarray, n_sides: int = 32) -> trimesh.Trimesh:
    n = len(pts)
    vertices = []
    faces = []
    for i in range(n):
        if i < n - 1:
            direction = pts[i + 1] - pts[i]
        else:
            direction = pts[i] - pts[i - 1]
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-12:
            direction = direction / direction_norm
        else:
            direction = np.array([0.0, 0.0, 1.0])
        if abs(direction[0]) < 0.9:
            up = np.array([1.0, 0.0, 0.0])
        else:
            up = np.array([0.0, 1.0, 0.0])
        n_dir = np.cross(direction, up)
        n_dir /= np.linalg.norm(n_dir) + 1e-12
        b_dir = np.cross(direction, n_dir)
        b_dir /= np.linalg.norm(b_dir) + 1e-12
        r = rad[i]
        for j in range(n_sides):
            angle = 2 * math.pi * j / n_sides
            v = pts[i] + r * (math.cos(angle) * n_dir + math.sin(angle) * b_dir)
            vertices.append(v)
    for i in range(n - 1):
        for j in range(n_sides):
            a = i * n_sides + j
            b = i * n_sides + (j + 1) % n_sides
            c = (i + 1) * n_sides + j
            d = (i + 1) * n_sides + (j + 1) % n_sides
            faces.append([a, c, b])
            faces.append([b, c, d])
    return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)


def _vtk_to_centerline(pd: vtk.vtkPolyData) -> Optional[GeometryCenterline]:
    n = pd.GetNumberOfPoints()
    if n < 2:
        return None
    pts = np.array([pd.GetPoint(i) for i in range(n)], dtype=float)
    rad_arr = pd.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if rad_arr is not None:
        rad = np.array([rad_arr.GetTuple1(i) for i in range(n)], dtype=float)
    else:
        rad = np.full(n, 1e-3, dtype=float)
    return compute_centerline_geometry(pts, rad)


def run_pipeline(
    input_path: Path,
    backend: str = "python_eikonal",
    acceleration: str = "auto",
    resampling_step: float = 1.0,
    output_path: Optional[Path] = None,
    voronoi_output: Optional[Path] = None,
    delaunay_output: Optional[Path] = None,
    diagnostics_output: Optional[Path] = None,
) -> Tuple[Optional[Centerline], PipelineReport]:
    t0 = time.perf_counter()
    report = PipelineReport(
        backend=backend,
        acceleration=acceleration,
        numba_available=bool(NUMBA_AVAILABLE),
    )

    use_numba = False
    if acceleration == "numba" and NUMBA_AVAILABLE:
        use_numba = True
    elif acceleration == "auto" and NUMBA_AVAILABLE:
        use_numba = True

    phase_timings: Dict[str, float] = {}

    logger.info("Phase A: Preprocess surface")
    t_start = time.perf_counter()
    surface = read_polydata(input_path)
    model = preprocess_surface(surface)
    phase_timings["preprocess"] = time.perf_counter() - t_start
    report.phase_timings = phase_timings

    logger.info("Phase B: Cap surface")
    t_start = time.perf_counter()
    capper = vmtkSurfaceCapper()
    capper.Surface = model.polydata
    capper.Execute()
    capped = capper.Output
    caps = capper.Caps
    loops = capper.BoundaryLoops
    phase_timings["capping"] = time.perf_counter() - t_start
    report.phase_timings = phase_timings
    if not loops:
        report.warnings.append("No boundary loops detected")

    logger.info("Phase C: Build Delaunay volume")
    t_start = time.perf_counter()
    delaunay = build_delaunay(capped)
    phase_timings["delaunay"] = time.perf_counter() - t_start
    report.phase_timings = phase_timings

    logger.info("Phase C/D: Classify internal tetrahedra")
    t_start = time.perf_counter()
    seed_cell_id = None
    if caps and loops:
        cap_center = loops[0].barycenter
        min_dist = float("inf")
        for cell_id in range(delaunay.mesh.GetNumberOfCells()):
            cell = delaunay.mesh.GetCell(cell_id)
            pts = np.array([delaunay.mesh.GetPoint(cell.GetPointId(i)) for i in range(4)], dtype=float)
            centroid = pts.mean(axis=0)
            d = np.linalg.norm(centroid - cap_center)
            if d < min_dist:
                min_dist = d
                seed_cell_id = cell_id
    internal = classify_internal_tetrahedra(delaunay, capped, seed_cell_id=seed_cell_id, validate_level2=True, subresolution_factor=1.0)
    phase_timings["internal_tets"] = time.perf_counter() - t_start
    report.phase_timings = phase_timings
    report.quality_metrics["n_internal_tets"] = internal.n_internal
    report.quality_metrics["n_seed_component"] = len(internal.seed_component)

    logger.info("Phase C/E: Build Voronoi dual")
    t_start = time.perf_counter()
    voronoi = build_voronoi_from_tetrahedra(internal.tetrahedra, acceleration=acceleration, internal_only=True)
    voronoi = filter_voronoi_by_clearance(voronoi, capped, radius_floor=1e-12)
    phase_timings["voronoi"] = time.perf_counter() - t_start
    report.phase_timings = phase_timings
    report.quality_metrics["voronoi_n_points"] = voronoi.n_points
    report.quality_metrics["voronoi_n_edges"] = voronoi.n_edges

    if delaunay_output and delaunay.mesh is not None:
        _write_unstructured(delaunay.mesh, delaunay_output)
    if voronoi_output:
        _write_polydata(_voronoi_to_vtp(voronoi), voronoi_output)

    logger.info("Phase D: Select poles and seeds")
    t_start = time.perf_counter()
    source_ids = []
    target_ids = []
    seed_positions = None
    seed_voronoi_ids = None
    if voronoi.n_points > 0 and loops and caps:
        cap_centers = np.array([loop.barycenter for loop in loops])
        cap_normals_list = [loop.pca_normal for loop in loops]
        normal_arr = delaunay.mesh.GetPointData().GetNormals()
        if normal_arr is not None:
            seed_voronoi_ids, seed_positions = find_voronoi_seeds(
                delaunay.mesh, cap_centers, cap_normals_list, internal.tetrahedra
            )
        all_cap_ids = list(range(len(caps)))
        source_ids = all_cap_ids
        target_ids = all_cap_ids
    elif voronoi.n_points >= 2:
        source_ids = [0]
        target_ids = [1]
    phase_timings["poles"] = time.perf_counter() - t_start
    report.phase_timings = phase_timings

    logger.info("Phase E: Compute path (backend=%s, numba=%s)", backend, use_numba)
    t_start = time.perf_counter()

    cap_centers = np.array([loop.barycenter for loop in loops]) if loops else np.zeros((0, 3))
    cap_normals = np.array([loop.pca_normal for loop in loops]) if loops else np.zeros((0, 3))

    internal_tet_centroids = np.array([t.centroid for t in internal.tetrahedra if t.is_internal]) if internal.tetrahedra else np.zeros((0, 3))
    if len(internal_tet_centroids) > 0:
        origin = internal_tet_centroids.min(axis=0) - 1.0
        max_extent = internal_tet_centroids.max(axis=0) + 1.0
        spacing = np.maximum((max_extent - origin) / 32.0, 0.1)
        shape = np.ceil((max_extent - origin) / spacing).astype(int)
        shape = np.maximum(shape, 1)
        mask = np.zeros(shape, dtype=bool)
        voxel_coords = ((internal_tet_centroids - origin) / spacing).astype(int)
        for vc in voxel_coords:
            idx = tuple(np.clip(vc, 0, shape - 1))
            mask[idx] = True
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=2)
    else:
        mask = np.ones((1, 1, 1), dtype=bool)
        origin = np.zeros(3)
        spacing = np.ones(3)

    fm = vmtkFastMarchingLocal()
    fm.VoronoiDiagram = FastMarchingVoronoiGraph(
        points=voronoi.points,
        radii=voronoi.radii,
        edges=voronoi.edges,
        polys=getattr(voronoi, 'polys', []),
        polys_edges=getattr(voronoi, 'polys_edges', []),
    )
    fm.CapCenters = cap_centers
    fm.CapNormals = cap_normals
    fm.SeedPositions = np.array(seed_positions, dtype=float) if seed_positions is not None else None
    fm.SeedVoronoiIds = seed_voronoi_ids if seed_voronoi_ids is not None else None
    fm.Backend = backend
    fm.RadiusFloor = 1e-12
    fm.EikonalRelaxationIters = 500
    fm.InternalVolumeMask = mask
    fm.VolumeOrigin = origin
    fm.VoxelSpacing = tuple(spacing.tolist())

    if not source_ids and not target_ids and loops:
        source_ids = list(range(len(caps)))
        target_ids = list(range(len(caps)))
    elif not source_ids and not target_ids and voronoi.n_points >= 2:
        source_ids = [0]
        target_ids = [1]

    fm.SourceIds = source_ids
    fm.TargetIds = target_ids
    fm.Execute()

    phase_timings["fast_marching"] = time.perf_counter() - t_start
    report.phase_timings = phase_timings

    centerlines = fm.Centerlines if fm.Centerlines else []
    if not centerlines:
        report.warnings.append("No valid centerline path found")
        report.status = "WARNING"
        empty = GeometryCenterline(
            points=np.array([]).reshape(0, 3),
            radii=np.array([], dtype=float),
            abscissas=np.array([], dtype=float),
            tangents=np.array([]).reshape(0, 3),
            curvature=np.array([], dtype=float),
            torsion=np.array([], dtype=float),
            tortuosity=0.0,
            frenet_tangents=np.array([]).reshape(0, 3),
            parallel_transport_normals=np.array([]).reshape(0, 3),
            parallel_transport_binormals=np.array([]).reshape(0, 3),
        )
        return empty, report

    median_voronoi_radius = float(np.median(voronoi.radii)) if len(voronoi.radii) > 0 else 1.0
    def _centerline_score(c):
        if len(c.radii) == 0:
            return float('inf')
        mean_r = float(np.nanmean(c.radii))
        radius_score = abs(mean_r - median_voronoi_radius)
        length = float(c.abscissas[-1]) if len(c.abscissas) > 0 else 0.0
        length_score = abs(length - 2.0 * median_voronoi_radius * len(voronoi.points) ** 0.5)
        return radius_score + 0.01 * length_score

    cl = min(centerlines, key=_centerline_score)

    logger.info("Phase F: Append cap endpoints")
    t_start = time.perf_counter()
    if len(cl.points) >= 2 and len(caps) >= 2 and len(loops) >= 2:
        src_loop = loops[0]
        tgt_loop = loops[-1]
        new_pts = np.vstack([src_loop.barycenter.reshape(1, 3), cl.points, tgt_loop.barycenter.reshape(1, 3)])
        new_rads = np.concatenate([[cl.radii[0]], cl.radii, [cl.radii[-1]]])
        seg_lengths = np.linalg.norm(np.diff(new_pts, axis=0), axis=1)
        new_abscissas = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        new_tangents = np.vstack([cl.tangents[0].reshape(1, 3), cl.tangents, cl.tangents[-1].reshape(1, 3)])
        cl = GeometryCenterline(
            points=new_pts,
            radii=new_rads,
            abscissas=new_abscissas,
            tangents=new_tangents,
            curvature=cl.curvature,
            torsion=cl.torsion,
            tortuosity=cl.tortuosity,
            frenet_tangents=cl.frenet_tangent,
            parallel_transport_normals=cl.parallel_transport_normals,
            parallel_transport_binormals=cl.parallel_transport_binormals,
        )
    phase_timings["endpoints"] = time.perf_counter() - t_start
    report.phase_timings = phase_timings

    logger.info("Phase G: Resample centerline")
    t_start = time.perf_counter()
    cl_resampled = resample_centerline(
        cl.points, cl.radii, cl.abscissas, cl.tangents, cl.curvature, cl.torsion,
        step_length=resampling_step,
    )
    phase_timings["resampling"] = time.perf_counter() - t_start
    report.phase_timings = phase_timings

    logger.info("Phase G: Compute sections")
    t_start = time.perf_counter()
    sectioner = vmtkCenterlineSectionsLocal()
    sectioner.Surface = capped
    sectioner.Centerline = cl_resampled.points
    sectioner.NumberOfSections = 100
    sectioner.ResamplingNumberOfPoints = 64
    sectioner.UseLocalSearch = True
    sectioner.LocalSearchRadius = 10.0
    sectioner.MinArea = 1e-10
    sectioner.MinScore = 200.0
    sectioner.Execute()
    sections = sectioner.CenterlineSections if sectioner.CenterlineSections else []
    phase_timings["sections"] = time.perf_counter() - t_start
    report.phase_timings = phase_timings
    report.quality_metrics["n_sections"] = len(sections)

    logger.info("Phase H: Build network")
    t_start = time.perf_counter()
    network = build_centerline_network(
        [cl_resampled.points],
        [cl_resampled.radii],
        [cl_resampled.abscissas],
        [cl_resampled.tangents],
        [cl_resampled.curvature],
        [cl_resampled.torsion],
        surface=capped,
    )
    phase_timings["network"] = time.perf_counter() - t_start
    report.phase_timings = phase_timings
    report.quality_metrics["network_n_points"] = len(network.points) if network else 0
    report.quality_metrics["network_n_edges"] = len(network.edges) if network else 0

    elapsed_total = time.perf_counter() - t0
    report.max_memory_mb = 0.0
    report.quality_metrics["total_elapsed_s"] = elapsed_total

    if output_path:
        _write_polydata(_centerline_to_vtp(cl_resampled), output_path)
        logger.info("Wrote centerline to %s", output_path)

    if diagnostics_output:
        diag = {
            "backend": backend,
            "acceleration": acceleration,
            "numba_available": bool(NUMBA_AVAILABLE),
            "phase_timings": {k: round(v, 4) for k, v in report.phase_timings.items()},
            "quality_metrics": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in report.quality_metrics.items()},
            "warnings": report.warnings,
            "status": report.status,
        }
        with open(diagnostics_output, "w") as f:
            json.dump(diag, f, indent=2)
        logger.info("Wrote diagnostics to %s", diagnostics_output)

    return cl_resampled, report


class vmtkCenterlinesPython(vmtkBaseScript):
    def __init__(self):
        super().__init__()
        self.InputFileName: str = ""
        self.Backend: str = "python_eikonal"
        self.Acceleration: str = "auto"
        self.ResamplingStepLength: float = 1.0
        self.OutputFileName: str = ""
        self.VoronoiOutputFileName: str = ""
        self.DelaunayOutputFileName: str = ""
        self.DiagnosticsOutputFileName: str = ""
        self.Centerlines: Optional[vtk.vtkPolyData] = None
        self.Report: Optional[Dict[str, Any]] = None

    def Execute(self) -> None:
        input_path = Path(self.InputFileName)
        output_path = Path(self.OutputFileName) if self.OutputFileName else None
        voronoi_output = Path(self.VoronoiOutputFileName) if self.VoronoiOutputFileName else None
        delaunay_output = Path(self.DelaunayOutputFileName) if self.DelaunayOutputFileName else None
        diagnostics_output = Path(self.DiagnosticsOutputFileName) if self.DiagnosticsOutputFileName else None

        cl, report = run_pipeline(
            input_path=input_path,
            backend=self.Backend,
            acceleration=self.Acceleration,
            resampling_step=self.ResamplingStepLength,
            output_path=output_path,
            voronoi_output=voronoi_output,
            delaunay_output=delaunay_output,
            diagnostics_output=diagnostics_output,
        )

        if cl is not None and len(cl.points) > 0:
            self.Centerlines = _centerline_to_vtp(cl)
        else:
            self.Centerlines = vtk.vtkPolyData()
        self.Report = {
            "backend": report.backend,
            "acceleration": report.acceleration,
            "numba_available": report.numba_available,
            "phase_timings": report.phase_timings,
            "max_memory_mb": report.max_memory_mb,
            "quality_metrics": report.quality_metrics,
            "warnings": report.warnings,
            "status": report.status,
        }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="VMTK-like centerline reconstruction (Python)")
    parser.add_argument("--input", required=True, type=Path, help="Input surface STL/VTP")
    parser.add_argument("--backend", default="python_eikonal", choices=["python_eikonal", "dijkstra"])
    parser.add_argument("--acceleration", default="auto", choices=["numpy", "numba", "auto"])
    parser.add_argument("--resampling-step", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--voronoi-output", type=Path, default=None)
    parser.add_argument("--delaunay-output", type=Path, default=None)
    parser.add_argument("--diagnostics-output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    _, report = run_pipeline(
        input_path=args.input,
        backend=args.backend,
        acceleration=args.acceleration,
        resampling_step=args.resampling_step,
        output_path=args.output,
        voronoi_output=args.voronoi_output,
        delaunay_output=args.delaunay_output,
        diagnostics_output=args.diagnostics_output,
    )

    print(json.dumps({
        "backend": report.backend,
        "acceleration": report.acceleration,
        "numba_available": report.numba_available,
        "phase_timings": {k: round(v, 4) for k, v in report.phase_timings.items()},
        "max_memory_mb": round(report.max_memory_mb, 2),
        "quality_metrics": report.quality_metrics,
        "warnings": report.warnings,
        "status": report.status,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
