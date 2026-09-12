import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import trimesh
import vtk

from .pypes import vmtkBaseScript
from ..section_extractor import Section as BaseSection
from ..section_extractor import (
    _build_local_frame,
    _compute_polygon_area,
    _resample_contour,
    _select_best_polyline,
)

logger = logging.getLogger(__name__)


def _trimesh_to_vtk_polydata(mesh: trimesh.Trimesh) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    for p in mesh.vertices:
        points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    polys = vtk.vtkCellArray()
    for face in mesh.faces:
        polys.InsertNextCell(3)
        polys.InsertCellPoint(int(face[0]))
        polys.InsertCellPoint(int(face[1]))
        polys.InsertCellPoint(int(face[2]))
    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetPolys(polys)
    return pd


def _vtk_polydata_to_trimesh(pd: vtk.vtkPolyData) -> trimesh.Trimesh:
    pts = []
    for i in range(pd.GetNumberOfPoints()):
        p = pd.GetPoint(i)
        pts.append([p[0], p[1], p[2]])
    faces = []
    polys = pd.GetPolys()
    polys.InitTraversal()
    pt_ids = vtk.vtkIdList()
    while polys.GetNextCell(pt_ids):
        if pt_ids.GetNumberOfIds() >= 3:
            faces.append([pt_ids.GetId(0), pt_ids.GetId(1), pt_ids.GetId(2)])
    return trimesh.Trimesh(np.array(pts, dtype=float), np.array(faces, dtype=int), process=False)


def _point_in_polygon_2d(points_2d: np.ndarray, query: np.ndarray) -> bool:
    x, y = points_2d[:, 0], points_2d[:, 1]
    qx, qy = float(query[0]), float(query[1])
    n = len(x)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(x[i]), float(y[i])
        xj, yj = float(x[j]), float(y[j])
        if ((yi > qy) != (yj > qy)) and (qx < (xj - xi) * (qy - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


@dataclass
class LocalSection:
    center: np.ndarray
    direction: np.ndarray
    tangent: np.ndarray
    points: np.ndarray
    radius: float
    area: float
    perimeter: float
    phase_locked_points: np.ndarray
    base_section: Optional[BaseSection] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)
        self.direction = np.asarray(self.direction, dtype=float)
        self.tangent = np.asarray(self.tangent, dtype=float)
        self.points = np.asarray(self.points, dtype=float)
        self.phase_locked_points = np.asarray(self.phase_locked_points, dtype=float)
        d_norm = np.linalg.norm(self.direction)
        if d_norm > 1e-12:
            self.direction = self.direction / d_norm
        t_norm = np.linalg.norm(self.tangent)
        if t_norm > 1e-12:
            self.tangent = self.tangent / t_norm


def _parallel_transport_frame(tangents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(tangents)
    normals = np.zeros((n, 3))
    binormals = np.zeros((n, 3))

    t0 = tangents[0]
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(t0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    n0 = np.cross(t0, arbitrary)
    n0 /= np.linalg.norm(n0) + 1e-12
    b0 = np.cross(t0, n0)

    normals[0] = n0
    binormals[0] = b0

    for i in range(1, n):
        prev_t = tangents[i - 1]
        curr_t = tangents[i]

        axis = np.cross(prev_t, curr_t)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            normals[i] = normals[i - 1]
            binormals[i] = binormals[i - 1]
            continue

        k = axis / axis_norm
        cos_angle = float(np.clip(np.dot(prev_t, curr_t), -1.0, 1.0))
        sin_angle = np.sqrt(max(1.0 - cos_angle * cos_angle, 0.0))
        one_minus_cos = 1.0 - cos_angle

        n_prev = normals[i - 1]
        n_curr = n_prev * cos_angle + np.cross(k, n_prev) * sin_angle + k * np.dot(k, n_prev) * one_minus_cos
        normals[i] = n_curr / (np.linalg.norm(n_curr) + 1e-12)
        binormals[i] = np.cross(curr_t, normals[i])
        binormals[i] /= np.linalg.norm(binormals[i]) + 1e-12

    return normals, binormals


def _extract_loops_from_cutter(output: vtk.vtkPolyData) -> List[np.ndarray]:
    loops: List[np.ndarray] = []
    lines = output.GetLines()
    if lines is None or lines.GetNumberOfCells() == 0:
        return loops

    lines.InitTraversal()
    pt_ids = vtk.vtkIdList()
    while lines.GetNextCell(pt_ids):
        if pt_ids.GetNumberOfIds() < 3:
            continue
        loop = []
        for i in range(pt_ids.GetNumberOfIds()):
            loop.append(output.GetPoint(pt_ids.GetId(i)))
        loops.append(np.asarray(loop, dtype=float))

    return loops


def _build_local_polydata(surface: vtk.vtkPolyData, cell_ids: vtk.vtkIdList) -> vtk.vtkPolyData:
    point_set = set()
    for i in range(cell_ids.GetNumberOfIds()):
        cell = surface.GetCell(cell_ids.GetId(i))
        ids = cell.GetPointIds()
        for j in range(ids.GetNumberOfIds()):
            point_set.add(ids.GetId(j))

    old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(point_set))}
    new_pts = vtk.vtkPoints()
    for old_id in sorted(point_set):
        p = surface.GetPoint(old_id)
        new_pts.InsertNextPoint(p[0], p[1], p[2])

    new_polys = vtk.vtkCellArray()
    for i in range(cell_ids.GetNumberOfIds()):
        cell = surface.GetCell(cell_ids.GetId(i))
        ids = cell.GetPointIds()
        n_pts = ids.GetNumberOfIds()
        new_polys.InsertNextCell(n_pts)
        for j in range(n_pts):
            new_polys.InsertCellPoint(old_to_new[ids.GetId(j)])

    local = vtk.vtkPolyData()
    local.SetPoints(new_pts)
    local.SetPolys(new_polys)
    return local


def _score_contour(
    loop: np.ndarray,
    center: np.ndarray,
    plane_normal: np.ndarray,
    prev_loop: Optional[np.ndarray],
) -> float:
    score = 0.0
    is_closed = len(loop) > 2 and np.allclose(loop[0], loop[-1], atol=1e-8)

    if is_closed:
        score += 1000.0
    else:
        score -= 500.0

    centroid = loop.mean(axis=0)
    dist = np.linalg.norm(centroid - center)
    score -= dist * 10.0

    if is_closed:
        u, v_dir = _build_local_frame(plane_normal)
        pts_2d = np.column_stack(((loop - center) @ u, (loop - center) @ v_dir))
        if _point_in_polygon_2d(pts_2d[:-1], np.array([0.0, 0.0])):
            score += 500.0

        area = abs(_compute_polygon_area(pts_2d))
        score += min(area, 1000.0)

    if prev_loop is not None and len(prev_loop) > 0:
        prev_centroid = prev_loop.mean(axis=0)
        continuity_dist = np.linalg.norm(centroid - prev_centroid)
        score -= continuity_dist * 5.0

    return score


def _lock_phase(
    current: np.ndarray,
    previous: Optional[np.ndarray],
    plane_normal: np.ndarray,
) -> np.ndarray:
    if previous is None or len(current) < 3:
        return current

    prev_first = previous[0]

    if not np.allclose(current[0], current[-1], atol=1e-8):
        current = np.vstack([current, current[0:1]])

    u, v_dir = _build_local_frame(plane_normal)
    curr_2d = np.column_stack(((current - current.mean(axis=0)) @ u, (current - current.mean(axis=0)) @ v_dir))
    prev_2d = np.column_stack(((previous - previous.mean(axis=0)) @ u, (previous - previous.mean(axis=0)) @ v_dir))

    curr_area = _compute_polygon_area(curr_2d)
    prev_area = _compute_polygon_area(prev_2d)

    if (curr_area > 0) != (prev_area > 0):
        current = current[::-1]

    dists = np.linalg.norm(current[:-1] - prev_first, axis=1)
    best_idx = int(np.argmin(dists))

    return np.roll(current[:-1], -best_idx, axis=0)


def _cut_surface(
    surface: vtk.vtkPolyData,
    center: np.ndarray,
    plane_normal: np.ndarray,
    use_local: bool,
    local_radius: float,
) -> List[np.ndarray]:
    plane = vtk.vtkPlane()
    plane.SetOrigin(float(center[0]), float(center[1]), float(center[2]))
    plane.SetNormal(float(plane_normal[0]), float(plane_normal[1]), float(plane_normal[2]))

    target_surface = surface

    if use_local and local_radius > 0:
        try:
            locator = vtk.vtkCellLocator()
            locator.SetDataSet(surface)
            locator.BuildLocator()

            cell_ids = vtk.vtkIdList()
            locator.FindCellsWithinBounds(center, local_radius, cell_ids)

            if cell_ids.GetNumberOfIds() > 0 and cell_ids.GetNumberOfIds() < surface.GetNumberOfCells():
                target_surface = _build_local_polydata(surface, cell_ids)
        except Exception:
            pass

    cutter = vtk.vtkCutter()
    cutter.SetInputData(target_surface)
    cutter.SetCutFunction(plane)
    cutter.Update()

    return _extract_loops_from_cutter(cutter.GetOutput())


def _compute_section_properties(points: np.ndarray, center: np.ndarray, plane_normal: np.ndarray) -> Tuple[float, float, float]:
    if len(points) < 3:
        return 0.0, 0.0, 0.0

    u, v_dir = _build_local_frame(plane_normal)
    pts_2d = np.column_stack(((points - center) @ u, (points - center) @ v_dir))
    area = float(abs(_compute_polygon_area(pts_2d)))

    closed = np.vstack([points, points[0]])
    perimeter = float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))
    radius = float(np.sqrt(area / np.pi)) if area > 0.0 else 0.0

    return radius, area, perimeter


class vmtkCenterlineSectionsLocal(vmtkBaseScript):
    def __init__(self):
        super().__init__()
        self.Surface: Optional[vtk.vtkPolyData] = None
        self.Centerlines: Optional[vtk.vtkPolyData] = None
        self.CenterlineSections: Optional[List[LocalSection]] = None
        self.NumberOfSections: int = 100
        self.ResamplingNumberOfPoints: int = 64
        self.LocalSearchRadius: float = 10.0
        self.UseLocalSearch: bool = True
        self.MinArea: float = 1e-10
        self.MinScore: float = 200.0
        self.Centerline: Optional[np.ndarray] = None

    def _extract_centerline_points(self) -> np.ndarray:
        if self.Centerline is not None:
            return np.asarray(self.Centerline, dtype=float)

        if self.Centerlines is None:
            raise ValueError("No centerline provided")

        return np.array([self.Centerlines.GetPoint(i) for i in range(self.Centerlines.GetNumberOfPoints())], dtype=float)

    def _sample_stations(self, centerline: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        diffs = np.diff(centerline, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_length = cumulative[-1]

        if total_length < 1e-12:
            raise ValueError("Degenerate centerline")

        n = self.NumberOfSections
        target_d = np.linspace(0, total_length, n, endpoint=False)

        sample_points = []
        for d in target_d:
            idx = int(np.searchsorted(cumulative, d))
            idx = min(idx, len(centerline) - 1)
            if idx == 0:
                sample_points.append(centerline[0])
            else:
                seg_start = cumulative[idx - 1]
                seg_end = cumulative[idx]
                t = (d - seg_start) / (seg_end - seg_start + 1e-12)
                pt = centerline[idx - 1] + t * diffs[idx - 1]
                sample_points.append(pt)

        sample_points = np.array(sample_points, dtype=float)

        raw_tangents = np.zeros_like(sample_points)
        raw_tangents[0] = sample_points[1] - sample_points[0]
        raw_tangents[-1] = sample_points[-1] - sample_points[-2]
        for i in range(1, len(sample_points) - 1):
            raw_tangents[i] = sample_points[i + 1] - sample_points[i - 1]

        norms = np.linalg.norm(raw_tangents, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        tangents = raw_tangents / norms

        smoothed = np.zeros_like(tangents)
        for i in range(len(tangents)):
            start = max(0, i - 1)
            end = min(len(tangents), i + 2)
            smoothed[i] = tangents[start:end].mean(axis=0)
        norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        tangents = smoothed / norms

        return sample_points, tangents

    def Execute(self) -> None:
        if self.Surface is None:
            self.PrintError("Error: No input surface.")
            return

        centerline = self._extract_centerline_points()
        if len(centerline) < 2:
            self.PrintError("Error: Centerline has too few points.")
            return

        centers, tangents = self._sample_stations(centerline)
        normals, binormals = _parallel_transport_frame(tangents)

        sections: List[LocalSection] = []
        prev_loop: Optional[np.ndarray] = None

        for i in range(len(centers)):
            center = centers[i]
            direction = tangents[i]

            loops = _cut_surface(
                self.Surface, center, direction,
                use_local=self.UseLocalSearch,
                local_radius=self.LocalSearchRadius,
            )

            if not loops:
                logger.debug("No loops at station %d", i)
                continue

            scored = []
            for loop in loops:
                score = _score_contour(loop, center, direction, prev_loop)
                scored.append((score, loop))
            scored.sort(key=lambda x: x[0], reverse=True)

            best_score, best_loop = scored[0]

            if best_score < self.MinScore and i > 0:
                mid_center = 0.5 * (centers[i - 1] + center)
                mid_loops = _cut_surface(self.Surface, mid_center, direction, False, 0.0)
                if mid_loops:
                    mid_scored = [
                        (_score_contour(lp, mid_center, direction, prev_loop), lp)
                        for lp in mid_loops
                    ]
                    mid_scored.sort(key=lambda x: x[0], reverse=True)
                    if mid_scored[0][0] > best_score:
                        best_score, best_loop = mid_scored[0]

            if best_score < self.MinScore:
                logger.debug("Rejected ambiguous cut at station %d", i)
                continue

            if len(best_loop) < 3:
                continue

            diffs_seg = np.diff(best_loop, axis=0)
            keep = np.concatenate([[True], np.linalg.norm(diffs_seg, axis=1) > 1e-12])
            best_loop = best_loop[keep]
            if len(best_loop) < 3:
                continue

            if not np.allclose(best_loop[0], best_loop[-1], atol=1e-8):
                best_loop = np.vstack([best_loop, best_loop[0:1]])

            resampled = _resample_contour(best_loop, self.ResamplingNumberOfPoints)
            phase_locked = _lock_phase(resampled, prev_loop, direction)

            radius, area, perimeter = _compute_section_properties(phase_locked, center, direction)

            if area < self.MinArea:
                logger.debug("Section area below threshold at station %d", i)
                continue

            base = BaseSection(center=center, direction=direction, points=phase_locked)
            sec = LocalSection(
                center=center.copy(),
                direction=direction.copy(),
                tangent=binormals[i].copy(),
                points=resampled.copy(),
                radius=radius,
                area=area,
                perimeter=perimeter,
                phase_locked_points=phase_locked.copy(),
                base_section=base,
                metadata={
                    "station_index": i,
                    "resampling_points": self.ResamplingNumberOfPoints,
                    "source_loop_size": len(best_loop),
                    "cut_score": float(best_score),
                },
            )
            sections.append(sec)
            prev_loop = phase_locked

        self.CenterlineSections = sections
        self.PrintLog(f"Computed {len(sections)} local centerline sections")
