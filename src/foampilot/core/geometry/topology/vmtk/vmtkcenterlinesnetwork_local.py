import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import vtk

logger = logging.getLogger(__name__)


@dataclass
class CenterlineNetwork:
    points: np.ndarray
    edges: np.ndarray
    nodes: np.ndarray
    centerline_ids: np.ndarray
    group_ids: np.ndarray
    tract_ids: np.ndarray
    blanking: np.ndarray
    radius: np.ndarray
    bifurcation_nodes: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


def _detect_bifurcations(network: CenterlineNetwork, min_degree: int = 3, min_angle_separation: float = 0.5) -> np.ndarray:
    n = len(network.points)
    degree = np.zeros(n, dtype=int)
    for e in network.edges:
        degree[e[0]] += 1
        degree[e[1]] += 1

    candidates = np.where(degree >= min_degree)[0]
    if len(candidates) == 0:
        return np.array([], dtype=int)

    bifurcations = []
    for node in candidates:
        neighbors = []
        for e in network.edges:
            if e[0] == node:
                neighbors.append(e[1])
            elif e[1] == node:
                neighbors.append(e[0])
        if len(neighbors) < min_degree:
            continue

        dirs = []
        for nb in neighbors:
            d = network.points[nb] - network.points[node]
            norm = np.linalg.norm(d)
            if norm > 1e-9:
                dirs.append(d / norm)
        if len(dirs) < 2:
            continue

        ok = True
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                dot = abs(np.dot(dirs[i], dirs[j]))
                if dot > min_angle_separation:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            bifurcations.append(node)

    return np.array(bifurcations, dtype=int)


def _compute_confidence(network: CenterlineNetwork, surface: Optional[vtk.vtkPolyData] = None) -> Tuple[str, List[str]]:
    warnings = []

    if len(network.points) == 0:
        return "FAIL", ["Empty centerline network"]

    if len(network.edges) == 0:
        return "FAIL", ["No edges in network"]

    if np.any(network.radius <= 0):
        warnings.append("Zero or negative radius detected")

    if surface is not None:
        enclosed = vtk.vtkSelectEnclosedPoints()
        enclosed.SetInputData(vtk.vtkPolyData())
        enclosed.SetSurfaceData(surface)
        enclosed.Update()

        pts = vtk.vtkPoints()
        for p in network.points:
            pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        enclosed.SetInputData(pd)
        enclosed.Update()

        outside = 0
        for i in range(len(network.points)):
            if not enclosed.IsInsideSurface(float(network.points[i, 0]), float(network.points[i, 1]), float(network.points[i, 2])):
                outside += 1
        if outside > 0:
            warnings.append(f"{outside} points outside surface")

    status = "PASS"
    if warnings:
        status = "WARNING"
    if any("Empty" in w or "No edges" in w for w in warnings):
        status = "FAIL"

    return status, warnings


def build_centerline_network(
    points_list: List[np.ndarray],
    radii_list: List[np.ndarray],
    abscissas_list: List[np.ndarray],
    tangents_list: List[np.ndarray],
    curvature_list: List[np.ndarray],
    torsion_list: List[np.ndarray],
    surface: Optional[vtk.vtkPolyData] = None,
) -> CenterlineNetwork:
    from .vmtkcenterlinegeometry_local import Centerline, compute_centerline_geometry

    centerlines = []
    for pts, rad in zip(points_list, radii_list):
        centerlines.append(compute_centerline_geometry(pts, rad))

    if not centerlines:
        return CenterlineNetwork(
            points=np.array([]).reshape(0, 3),
            edges=np.array([]).reshape(0, 2),
            nodes=np.array([]).reshape(0, 3),
            centerline_ids=np.array([], dtype=int),
            group_ids=np.array([], dtype=int),
            tract_ids=np.array([], dtype=int),
            blanking=np.array([], dtype=int),
            radius=np.array([], dtype=float),
        )

    all_pts = []
    all_rad = []
    offsets = [0]
    for cl in centerlines:
        all_pts.append(cl.points)
        all_rad.append(cl.radii)
        offsets.append(offsets[-1] + len(cl.points))

    points = np.vstack(all_pts)
    radius = np.concatenate(all_rad)

    edges = []
    centerline_ids = []
    group_ids = []
    tract_ids = []
    blanking = []

    for cl_idx, cl in enumerate(centerlines):
        start = offsets[cl_idx]
        end = offsets[cl_idx + 1]
        for i in range(start, end - 1):
            edges.append([i, i + 1])
            centerline_ids.append(cl_idx)
            group_ids.append(0 if cl_idx == 0 else cl_idx)
            tract_ids.append(cl_idx)
            blanking.append(0)

    edges_arr = np.array(edges, dtype=int) if edges else np.array([]).reshape(0, 2)
    nodes = points.copy()

    network = CenterlineNetwork(
        points=points,
        edges=edges_arr,
        nodes=nodes,
        centerline_ids=np.array(centerline_ids, dtype=int) if centerline_ids else np.array([], dtype=int),
        group_ids=np.array(group_ids, dtype=int) if group_ids else np.array([], dtype=int),
        tract_ids=np.array(tract_ids, dtype=int) if tract_ids else np.array([], dtype=int),
        blanking=np.array(blanking, dtype=int) if blanking else np.array([], dtype=int),
        radius=radius,
    )

    network.bifurcation_nodes = _detect_bifurcations(network)
    status, conf_warnings = _compute_confidence(network, surface=surface)
    logger.info("Built centerline network: %d points, %d edges, %d bifurcations, status=%s", len(points), len(edges_arr), len(network.bifurcation_nodes), status)
    return network
