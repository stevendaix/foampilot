import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import vtk

from .pypes import vmtkBaseScript

logger = logging.getLogger(__name__)


def read_polydata(path) -> vtk.vtkPolyData:
    path = str(path)
    if path.lower().endswith(".stl"):
        reader = vtk.vtkSTLReader()
    elif path.lower().endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    elif path.lower().endswith(".ply"):
        reader = vtk.vtkPLYReader()
    else:
        raise ValueError(f"Surface inconnue: {path}")
    reader.SetFileName(path)
    reader.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(reader.GetOutput())
    return result


@dataclass
class BoundaryLoop:
    boundary_id: int
    ordered_point_ids: np.ndarray
    barycenter: np.ndarray
    pca_normal: np.ndarray
    perimeter: float
    projected_area: float
    planarity: float


@dataclass
class Cap:
    boundary_id: int
    cap_center_id: int
    polydata: vtk.vtkPolyData
    area: float
    is_valid: bool
    validation_errors: List[str] = field(default_factory=list)


def _pca_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    u, v, normal = vh[0], vh[1], vh[2]
    if np.dot(np.cross(u, v), normal) < 0.0:
        v = -v
    return center, u, v, normal


def _compute_boundary_barycenter(loop_pts: np.ndarray) -> np.ndarray:
    n = len(loop_pts)
    if n < 2:
        return loop_pts.mean(axis=0)
    barycenter = np.zeros(3, dtype=float)
    weight_sum = 0.0
    for i in range(n):
        p0 = loop_pts[i]
        p1 = loop_pts[(i + 1) % n]
        edge_vec = p1 - p0
        weight = float(np.linalg.norm(edge_vec))
        if weight < 1e-12:
            continue
        midpoint = (p0 + p1) / 2.0
        barycenter += midpoint * weight
        weight_sum += weight
    if weight_sum < 1e-12:
        return loop_pts.mean(axis=0)
    return barycenter / weight_sum


def _planar_polygon_area(points_2d: np.ndarray) -> float:
    if len(points_2d) < 3:
        return 0.0
    area = 0.0
    n = len(points_2d)
    for i in range(n):
        x1, y1 = points_2d[i]
        x2, y2 = points_2d[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * abs(area)


def _planarity_score(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    centered = points - points.mean(axis=0)
    _, s, _ = np.linalg.svd(centered)
    if s.size < 3:
        return 0.0
    return float(s[2] / (s[0] + s[1] + s[2] + 1e-12))


def _is_convex(points_2d: np.ndarray) -> bool:
    n = len(points_2d)
    if n < 3:
        return True
    signed_area = 0.0
    for i in range(n):
        x1, y1 = points_2d[i]
        x2, y2 = points_2d[(i + 1) % n]
        signed_area += x1 * y2 - x2 * y1
    reference_sign = np.sign(signed_area)
    if reference_sign == 0:
        return True
    for i in range(n):
        a = points_2d[i]
        b = points_2d[(i + 1) % n]
        c = points_2d[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-10 and np.sign(cross) != reference_sign:
            return False
    return True


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > point[1]) != (yj > point[1])) and (
            point[0] < (xj - xi) * (point[1] - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _chain_boundary_loops(boundary_edges: np.ndarray, points: Optional[np.ndarray] = None, merge_tol: float = 1e-6) -> List[np.ndarray]:
    if boundary_edges.size == 0:
        return []

    if points is not None:
        node_map = {}
        unique_nodes = []
        for i in range(len(points)):
            key = tuple(np.round(points[i] / merge_tol).astype(int))
            if key not in node_map:
                node_map[key] = len(unique_nodes)
                unique_nodes.append(i)
        edges = [(node_map[tuple(np.round(points[a] / merge_tol).astype(int))],
                  node_map[tuple(np.round(points[b] / merge_tol).astype(int))])
                 for a, b in boundary_edges]
    else:
        edges = [(int(a), int(b)) for a, b in boundary_edges]

    adjacency: Dict[int, List[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    for node_id, neighbors in adjacency.items():
        degree = len(neighbors)
        if degree != 2:
            continue

    visited_edges = set()
    loops = []
    for start_edge in edges:
        a, b = start_edge
        if (a, b) in visited_edges or (b, a) in visited_edges:
            continue
        loop = [a, b]
        visited_edges.add((a, b))
        visited_edges.add((b, a))
        current = b
        while True:
            neighbors = adjacency.get(current, [])
            next_node = None
            for nb in neighbors:
                if (current, nb) in visited_edges:
                    continue
                next_node = nb
                break
            if next_node is None:
                break
            loop.append(next_node)
            visited_edges.add((current, next_node))
            visited_edges.add((next_node, current))
            current = next_node
            if current == a:
                break
        if len(loop) >= 3 and loop[0] == loop[-1]:
            loops.append(np.array(loop[:-1], dtype=np.int64))
    return loops


def _create_fan_cap(points_3d: np.ndarray, normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    center, u, v, _ = _pca_frame(points_3d)
    uv = np.column_stack(((points_3d - center) @ u, (points_3d - center) @ v))
    cap_center_2d = uv.mean(axis=0)
    n = len(uv)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend([3, i, j, n])
    all_points_2d = np.vstack([uv, cap_center_2d.reshape(1, 2)])
    return np.array(faces, dtype=np.int64), all_points_2d, cap_center_2d


def _create_constrained_cap(points_3d: np.ndarray, normal: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        import mapbox_earcut as earcut
    except ImportError:
        logger.warning("mapbox_earcut not available, falling back to fan triangulation")
        return None

    center, u, v, _ = _pca_frame(points_3d)
    uv = np.column_stack(((points_3d - center) @ u, (points_3d - center) @ v))
    polygon = uv.flatten().tolist()
    try:
        triangles = earcut(polygon, holes=None, dim=2)
    except Exception as exc:
        logger.warning(f"mapbox_earcut triangulation failed: {exc}")
        return None
    if len(triangles) == 0:
        return None
    faces = triangles.reshape(-1, 3)
    faces = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces])
    return faces.reshape(-1), uv


def _validate_cap(
    cap_points_2d: np.ndarray,
    cap_faces: np.ndarray,
    boundary_points_2d: np.ndarray,
    cap_normal: np.ndarray,
    cap_center_3d: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    v: Optional[np.ndarray] = None,
    center_3d: Optional[np.ndarray] = None,
    other_centers: Optional[List[np.ndarray]] = None,
    min_separation: float = 0.0,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if len(cap_faces) == 0:
        errors.append("Empty cap")
        return False, errors

    face_array = cap_faces.reshape(-1, 4)[:, 1:]
    if len(face_array) == 0:
        errors.append("No triangles in cap")
        return False, errors

    if _planar_polygon_area(boundary_points_2d) <= 0.0:
        errors.append("Non-positive boundary area")
        return False, errors

    for idx, face in enumerate(face_array):
        if len(face) < 3:
            errors.append(f"Degenerate triangle {idx}")
            continue
        a_2d = cap_points_2d[face[0]]
        b_2d = cap_points_2d[face[1]]
        c_2d = cap_points_2d[face[2]]
        if center_3d is not None and u is not None and v is not None:
            a_3d = center_3d + a_2d[0] * u + a_2d[1] * v
            b_3d = center_3d + b_2d[0] * u + b_2d[1] * v
            c_3d = center_3d + c_2d[0] * u + c_2d[1] * v
            cross = np.cross(b_3d - a_3d, c_3d - a_3d)
        else:
            cross = np.cross(b_2d - a_2d, c_2d - a_2d)
        if np.linalg.norm(cross) < 1e-12:
            errors.append(f"Degenerate triangle {idx}")
        else:
            tri_normal = cross / np.linalg.norm(cross)
            if np.dot(tri_normal, cap_normal) < 0.0:
                errors.append(f"Normal inconsistency in triangle {idx}")

    cap_center = cap_points_2d.mean(axis=0)
    if not _point_in_polygon(cap_center, boundary_points_2d):
        errors.append("Cap center outside boundary polygon")

    if other_centers is not None and min_separation > 0.0:
        for oc in other_centers:
            if np.linalg.norm(cap_center - oc) < min_separation:
                errors.append("Cap too close to another surface")
                break

    return len(errors) == 0, errors


class vmtkSurfaceCapper(vmtkBaseScript):
    def __init__(self):
        super().__init__()
        self.Surface: Optional[vtk.vtkPolyData] = None
        self.InputFileName: str = ""
        self.BoundaryLoops: List[BoundaryLoop] = []
        self.Caps: List[Cap] = []
        self.CapCenterIds: Optional[vtk.vtkIdList] = None
        self.Output: Optional[vtk.vtkPolyData] = None
        self.CapDisplacement: float = 0.1
        self.InPlaneDisplacement: float = 0.1

    def Execute(self):
        if self.Surface is None and self.InputFileName:
            self.Surface = read_polydata(self.InputFileName)
        if self.Surface is None:
            self.PrintError("Error: No input surface.")
            return

        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(self.Surface)
        clean.Update()
        surface = clean.GetOutput()

        boundary_edges_filter = vtk.vtkFeatureEdges()
        boundary_edges_filter.SetInputData(surface)
        boundary_edges_filter.BoundaryEdgesOn()
        boundary_edges_filter.FeatureEdgesOff()
        boundary_edges_filter.ManifoldEdgesOff()
        boundary_edges_filter.NonManifoldEdgesOff()
        boundary_edges_filter.Update()
        b_output = boundary_edges_filter.GetOutput()
        b_lines = b_output.GetLines()
        edge_array = []
        if b_lines is not None:
            b_lines.InitTraversal()
            b_id_list = vtk.vtkIdList()
            while b_lines.GetNextCell(b_id_list):
                if b_id_list.GetNumberOfIds() == 2:
                    edge_array.append([b_id_list.GetId(0), b_id_list.GetId(1)])
        boundary_edges_arr = np.array(edge_array, dtype=np.int64)

        n_pts = surface.GetNumberOfPoints()
        all_pts = np.array([surface.GetPoint(i) for i in range(n_pts)], dtype=np.float64)
        b_n_pts = b_output.GetNumberOfPoints()
        b_pts = np.array([b_output.GetPoint(i) for i in range(b_n_pts)], dtype=np.float64)

        loops = _chain_boundary_loops(boundary_edges_arr, points=all_pts)

        self.BoundaryLoops = []
        self.Caps = []
        self.CapCenterIds = vtk.vtkIdList()

        cap_polys = vtk.vtkCellArray()
        pts = vtk.vtkPoints()
        for i in range(n_pts):
            pts.InsertNextPoint(float(all_pts[i, 0]), float(all_pts[i, 1]), float(all_pts[i, 2]))

        existing_faces = []
        polys = surface.GetPolys()
        polys.InitTraversal()
        pt_ids = vtk.vtkIdList()
        while polys.GetNextCell(pt_ids):
            if pt_ids.GetNumberOfIds() >= 3:
                existing_faces.append([pt_ids.GetId(0), pt_ids.GetId(1), pt_ids.GetId(2)])

        other_barycenters: List[np.ndarray] = []

        for loop_idx, loop in enumerate(loops):
            loop_pts_3d = b_pts[loop]
            center = _compute_boundary_barycenter(loop_pts_3d)
            _, u, v, normal = _pca_frame(loop_pts_3d)
            uv = np.column_stack(((loop_pts_3d - center) @ u, (loop_pts_3d - center) @ v))
            perimeter = float(np.sum(np.linalg.norm(np.diff(np.vstack([uv, uv[0]]), axis=0), axis=1)))
            projected_area = _planar_polygon_area(uv)
            planarity = _planarity_score(loop_pts_3d)

            signed_area = 0.0
            for i in range(len(uv)):
                x1, y1 = uv[i]
                x2, y2 = uv[(i + 1) % len(uv)]
                signed_area += x1 * y2 - x2 * y1
            if signed_area < 0.0:
                loop = loop[::-1].copy()
                loop_pts_3d = loop_pts_3d[::-1].copy()
                uv = uv[::-1].copy()
                normal = -normal.copy()
                uv = np.column_stack(((loop_pts_3d - center) @ u, (loop_pts_3d - center) @ v))
                perimeter = float(np.sum(np.linalg.norm(np.diff(np.vstack([uv, uv[0]]), axis=0), axis=1)))
                projected_area = _planar_polygon_area(uv)

            mean_radius = float(np.mean(np.linalg.norm(loop_pts_3d - center, axis=1)))
            if self.CapDisplacement != 0.0:
                center = center + mean_radius * self.CapDisplacement * normal
            if self.InPlaneDisplacement != 0.0:
                inplane = np.cross(normal, np.array([0.0, 0.0, 1.0]))
                if np.linalg.norm(inplane) < 1e-12:
                    inplane = np.cross(normal, np.array([1.0, 0.0, 0.0]))
                inplane = inplane / np.linalg.norm(inplane)
                center = center + mean_radius * self.InPlaneDisplacement * inplane

            boundary_loop = BoundaryLoop(
                boundary_id=loop_idx,
                ordered_point_ids=loop.copy(),
                barycenter=center.copy(),
                pca_normal=normal.copy(),
                perimeter=perimeter,
                projected_area=projected_area,
                planarity=planarity,
            )
            self.BoundaryLoops.append(boundary_loop)

            convex = _is_convex(uv)
            cap_faces_flat = None
            cap_points_2d = None
            cap_center_2d = None

            if convex:
                cap_faces_flat, cap_points_2d, cap_center_2d = _create_fan_cap(loop_pts_3d, normal)
            else:
                constrained = _create_constrained_cap(loop_pts_3d, normal)
                if constrained is not None:
                    cap_faces_flat, cap_points_2d = constrained
                    cap_center_2d = cap_points_2d.mean(axis=0)

            valid = False
            errors: List[str] = []
            cap_center_3d = None

            if cap_faces_flat is not None and cap_center_2d is not None:
                cap_center_3d = center + cap_center_2d[0] * u + cap_center_2d[1] * v
                valid, errors = _validate_cap(
                    cap_points_2d,
                    cap_faces_flat,
                    uv,
                    normal,
                    cap_center_3d=cap_center_3d,
                    u=u,
                    v=v,
                    center_3d=center,
                    other_centers=other_barycenters,
                    min_separation=0.0,
                )

            if not valid or cap_faces_flat is None or cap_center_2d is None:
                cap_faces_flat, cap_points_2d, cap_center_2d = _create_fan_cap(loop_pts_3d, normal)
                cap_center_3d = center + cap_center_2d[0] * u + cap_center_2d[1] * v
                valid, errors = _validate_cap(
                    cap_points_2d,
                    cap_faces_flat,
                    uv,
                    normal,
                    cap_center_3d=cap_center_3d,
                    u=u,
                    v=v,
                    center_3d=center,
                    other_centers=other_barycenters,
                    min_separation=0.0,
                )

            cap_area = 0.0
            if cap_faces_flat is not None:
                face_array = cap_faces_flat.reshape(-1, 4)[:, 1:]
                if len(face_array) > 0:
                    cap_tri_pts = np.column_stack((cap_points_2d, np.zeros(len(cap_points_2d))))
                    cap_tris = cap_tri_pts[face_array]
                    v1 = cap_tris[:, 1] - cap_tris[:, 0]
                    v2 = cap_tris[:, 2] - cap_tris[:, 0]
                    vols = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) / 2.0
                    cap_area = float(np.sum(vols))

            cap_pd = None
            cap_center_id = -1
            if cap_faces_flat is not None and cap_center_2d is not None:
                cap_points_3d = np.column_stack((
                    center[0] + cap_points_2d[:, 0] * u[0] + cap_points_2d[:, 1] * v[0],
                    center[1] + cap_points_2d[:, 0] * u[1] + cap_points_2d[:, 1] * v[1],
                    center[2] + cap_points_2d[:, 0] * u[2] + cap_points_2d[:, 1] * v[2],
                ))
                cap_points_3d = np.vstack([cap_points_3d, cap_center_3d.reshape(1, 3)])
                for p in cap_points_3d:
                    pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))

                cap_center_id = pts.GetNumberOfPoints() - 1
                face_array = cap_faces_flat.reshape(-1, 4)
                for face in face_array:
                    if face[0] == 3 and len(face) == 4:
                        cap_polys.InsertNextCell(3)
                        cap_polys.InsertCellPoint(int(face[1]) + n_pts)
                        cap_polys.InsertCellPoint(int(face[2]) + n_pts)
                        cap_polys.InsertCellPoint(int(face[3]) + n_pts)

                self.CapCenterIds.InsertNextId(int(cap_center_id))

            cap = Cap(
                boundary_id=loop_idx,
                cap_center_id=cap_center_id,
                polydata=cap_pd if cap_pd is not None else vtk.vtkPolyData(),
                area=cap_area,
                is_valid=valid,
                validation_errors=errors,
            )
            self.Caps.append(cap)
            other_barycenters.append(center.copy())

        cap_output = vtk.vtkPolyData()
        cap_output.SetPoints(pts)
        for face in existing_faces:
            cap_polys.InsertNextCell(3)
            cap_polys.InsertCellPoint(int(face[0]))
            cap_polys.InsertCellPoint(int(face[1]))
            cap_polys.InsertCellPoint(int(face[2]))

        cap_output.SetPolys(cap_polys)

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(cap_output)
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.ComputePointNormalsOn()
        normals.SetFlipNormals(False)
        normals.Update()
        self.Output = normals.GetOutput()

        self.PrintLog(
            f"Capped surface: {self.Output.GetNumberOfPoints()} points, "
            f"{len(self.BoundaryLoops)} boundary loops, "
            f"{len(self.Caps)} caps"
        )
