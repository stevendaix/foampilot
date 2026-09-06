import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import vtk

logger = logging.getLogger(__name__)


def _circumsphere(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[np.ndarray, float]:
    center = np.zeros(3, dtype=float)
    try:
        radius_squared = vtk.vtkTetra.Circumsphere(
            tuple(p0), tuple(p1), tuple(p2), tuple(p3), center
        )
        return np.array(center, dtype=float), float(np.sqrt(max(radius_squared, 0.0)))
    except Exception:
        c = (p0 + p1 + p2 + p3) / 4.0
        return c, 0.0


@dataclass
class Tetrahedron:
    cell_id: int
    point_ids: np.ndarray
    centroid: np.ndarray
    volume: float
    quality: float
    is_internal: bool
    radius: float
    circumcenter: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    circumradius: float = 0.0


@dataclass
class InternalTetraMesh:
    tetrahedra: List[Tetrahedron]
    n_internal: int = 0
    n_total: int = 0
    connectivity: Dict[int, set] = field(default_factory=dict)
    seed_component: set = field(default_factory=set)
    warnings: List[str] = field(default_factory=list)


def _tetrahedron_quality_volume(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    volume = abs(np.linalg.det(np.vstack([p1 - p0, p2 - p0, p3 - p0]))) / 6.0
    edge_lengths = [np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p0), np.linalg.norm(p3 - p0),
                    np.linalg.norm(p2 - p1), np.linalg.norm(p3 - p1), np.linalg.norm(p3 - p2)]
    max_edge = max(edge_lengths) if edge_lengths else 1.0
    return volume / max(max_edge ** 3, 1e-12)


def _validate_tetrahedron_level2(tet_points: np.ndarray, surface: vtk.vtkPolyData, radius_floor: float = 1e-12) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    p0, p1, p2, p3 = tet_points[0], tet_points[1], tet_points[2], tet_points[3]

    try:
        cc, cr = _circumsphere(p0, p1, p2, p3)
    except Exception:
        errors.append("Degenerate circumsphere")
        return False, errors

    if cr < radius_floor:
        errors.append(f"Circumradius too small: {cr}")

    if not np.isfinite(cc).all():
        errors.append("Non-finite circumcenter")
        return False, errors

    edge_midpoints = [
        (p0 + p1) / 2.0, (p0 + p2) / 2.0, (p0 + p3) / 2.0,
        (p1 + p2) / 2.0, (p1 + p3) / 2.0, (p2 + p3) / 2.0,
    ]

    enclosed = vtk.vtkSelectEnclosedPoints()
    enclosed.SetInputData(vtk.vtkPolyData())
    enclosed.SetSurfaceData(surface)
    enclosed.Update()

    test_points = [cc] + edge_midpoints + [p0, p1, p2, p3]
    for idx, pt in enumerate(test_points):
        inside = enclosed.IsInsideSurface(float(pt[0]), float(pt[1]), float(pt[2]))
        if not inside:
            errors.append(f"Point {idx} outside surface: {pt}")
            return False, errors

    return len(errors) == 0, errors


def _extract_seed_component(connectivity: Dict[int, set], seed_id: int) -> set:
    visited = set()
    stack = [seed_id]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for neighbor in connectivity.get(current, []):
            if neighbor not in visited:
                stack.append(neighbor)
    return visited


def classify_internal_tetrahedra(
    delaunay,
    surface: vtk.vtkPolyData,
    seed_cell_id: Optional[int] = None,
    validate_level2: bool = True,
    radius_floor: float = 1e-12,
    subresolution_factor: float = 1.0,
) -> InternalTetraMesh:
    mesh = delaunay.mesh
    n_cells = mesh.GetNumberOfCells()
    n_points = mesh.GetNumberOfPoints()

    points_arr = np.zeros((n_points, 3), dtype=float)
    for i in range(n_points):
        p = mesh.GetPoint(i)
        points_arr[i] = [p[0], p[1], p[2]]

    surface.BuildLinks()
    normal_arr = surface.GetPointData().GetNormals()
    if normal_arr is None:
        normals_source = vtk.vtkPolyDataNormals()
        normals_source.SetInputData(surface)
        normals_source.AutoOrientNormalsOn()
        normals_source.ComputePointNormalsOn()
        normals_source.Update()
        normal_arr = normals_source.GetOutput().GetPointData().GetNormals()
        surface = normals_source.GetOutput()

    normals = np.array([normal_arr.GetTuple(i) for i in range(surface.GetNumberOfPoints())], dtype=float)
    surface_points = points_arr[:surface.GetNumberOfPoints()] if surface.GetNumberOfPoints() <= n_points else np.array([surface.GetPoint(i) for i in range(surface.GetNumberOfPoints())], dtype=float)
    if surface.GetNumberOfPoints() > n_points:
        surface_points = np.array([surface.GetPoint(i) for i in range(surface.GetNumberOfPoints())], dtype=float)

    cap_center_ids = set()
    if seed_cell_id is not None:
        cell = mesh.GetCell(seed_cell_id)
        for i in range(4):
            cap_center_ids.add(cell.GetPointId(i))

    tetrahedra = []
    for cell_id in range(n_cells):
        cell = mesh.GetCell(cell_id)
        pt_ids = np.array([cell.GetPointId(i) for i in range(4)], dtype=int)
        tet_points = points_arr[pt_ids]

        volume = abs(np.linalg.det(tet_points[1:] - tet_points[0])) / 6.0
        edge_lengths = [np.linalg.norm(tet_points[i] - tet_points[j]) for i in range(4) for j in range(i + 1, 4)]
        max_edge = max(edge_lengths) if edge_lengths else 1.0
        quality = volume / max(max_edge ** 3, 1e-12)

        try:
            cc, cr = _circumsphere(tet_points[0], tet_points[1], tet_points[2], tet_points[3])
        except Exception:
            cc = tet_points.mean(axis=0)
            cr = max_edge * 0.5

        is_internal = False
        boundary_tetra = False
        for pid in pt_ids:
            if pid in cap_center_ids:
                boundary_tetra = True
                break

        v = tet_points - cc
        n0, n1, n2, n3 = normals[pt_ids[0]], normals[pt_ids[1]], normals[pt_ids[2]], normals[pt_ids[3]]
        dot0, dot1, dot2, dot3 = np.dot(v[0], n0), np.dot(v[1], n1), np.dot(v[2], n2), np.dot(v[3], n3)
        all_dot_positive = (dot0 > 1e-12 and dot1 > 1e-12 and dot2 > 1e-12 and dot3 > 1e-12)
        all_but_one_positive = sum(d > 1e-12 for d in [dot0, dot1, dot2, dot3]) >= 3

        if all_dot_positive:
            is_internal = True
        elif boundary_tetra and all_but_one_positive:
            is_internal = True

        tetrahedra.append(Tetrahedron(
            cell_id=cell_id,
            point_ids=pt_ids,
            centroid=tet_points.mean(axis=0),
            volume=volume,
            quality=quality,
            is_internal=is_internal,
            radius=max_edge * 0.5,
            circumcenter=cc,
            circumradius=cr,
        ))

    if subresolution_factor > 0:
        for i, t in enumerate(tetrahedra):
            if not t.is_internal:
                continue
            if t.circumradius < 1e-12:
                continue
            pt_ids = t.point_ids
            min_surface_edge = float("inf")
            for pid in pt_ids:
                if pid < surface.GetNumberOfPoints():
                    id_list = vtk.vtkIdList()
                    surface.GetPointCells(pid, id_list)
                    for k in range(id_list.GetNumberOfIds()):
                        cid = id_list.GetId(k)
                        tri = surface.GetCell(cid)
                        if tri.GetNumberOfPoints() < 3:
                            continue
                        tp = np.array([surface.GetPoint(tri.GetPointId(j)) for j in range(3)], dtype=float)
                        edge_lengths = [np.linalg.norm(tp[k] - tp[l]) for k in range(3) for l in range(k+1, 3)]
                        if edge_lengths:
                            min_surface_edge = min(min_surface_edge, min(edge_lengths))
            if min_surface_edge < float("inf") and t.circumradius < subresolution_factor * min_surface_edge:
                tetrahedra[i].is_internal = False
                logger.debug("Removed subresolution tet %d: cr=%.4f < %.4f * min_edge=%.4f", t.cell_id, t.circumradius, subresolution_factor, min_surface_edge)

    internal = [t for t in tetrahedra if t.is_internal]
    logger.info("Classified tetrahedra: %d internal out of %d total", len(internal), len(tetrahedra))

    all_ids = {t.cell_id for t in tetrahedra}
    face_to_tets: Dict[tuple, List[int]] = {}
    for cell_id in all_ids:
        cell = mesh.GetCell(cell_id)
        for i in range(4):
            face_pids = tuple(sorted([cell.GetPointId(j) for j in range(4) if j != i]))
            face_to_tets.setdefault(face_pids, []).append(cell_id)

    connectivity: Dict[int, set] = {}
    for cell_id in all_ids:
        cell = mesh.GetCell(cell_id)
        for i in range(4):
            face_pids = tuple(sorted([cell.GetPointId(j) for j in range(4) if j != i]))
            neighbors = face_to_tets.get(face_pids, [])
            for nb in neighbors:
                if nb != cell_id and nb in all_ids:
                    connectivity.setdefault(cell_id, set()).add(nb)

    seed_component: set = set()
    if seed_cell_id is not None:
        if seed_cell_id in connectivity:
            seed_component = _extract_seed_component(connectivity, seed_cell_id)
        else:
            seed_component = {seed_cell_id}
        seed_component = {cid for cid in seed_component if cid in all_ids}
        logger.info("Seed component contains %d tetrahedra", len(seed_component))

    result = InternalTetraMesh(
        tetrahedra=tetrahedra,
        n_internal=len(internal),
        n_total=len(tetrahedra),
        connectivity=connectivity,
        seed_component=seed_component,
    )
    return result
