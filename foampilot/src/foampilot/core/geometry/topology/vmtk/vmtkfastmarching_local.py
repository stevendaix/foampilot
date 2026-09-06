import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import scipy.ndimage
import scipy.sparse
import trimesh
import vtk
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .pypes import vmtkBaseScript

try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)


def find_voronoi_seeds(
    delaunay_mesh: vtk.vtkUnstructuredGrid,
    cap_centers: np.ndarray,
    cap_normals: Optional[List[np.ndarray]],
    internal_tetrahedra: List,
) -> Tuple[List[int], List[np.ndarray]]:
    delaunay_mesh.BuildLinks()
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(delaunay_mesh)
    locator.BuildLocator()
    seed_voronoi_indices = []
    seed_positions = []
    n_points = delaunay_mesh.GetNumberOfPoints()

    for cap_idx in range(len(cap_centers)):
        baricenter = np.array(cap_centers[cap_idx], dtype=float)
        closest_point_id = locator.FindClosestPoint(baricenter)
        if closest_point_id < 0 or closest_point_id >= n_points:
            seed_voronoi_indices.append(-1)
            seed_positions.append(baricenter.copy())
            continue

        if cap_normals is not None and cap_idx < len(cap_normals) and np.linalg.norm(cap_normals[cap_idx]) > 1e-12:
            normal = cap_normals[cap_idx].copy()
        else:
            normal = np.array([0.0, 0.0, 1.0], dtype=float)

        point_cells = vtk.vtkIdList()
        delaunay_mesh.GetPointCells(closest_point_id, point_cells)

        max_radius = 0.0
        max_radius_cell_id = -1
        pole = baricenter.copy()

        for j in range(point_cells.GetNumberOfIds()):
            cell_id = point_cells.GetId(j)
            cell = delaunay_mesh.GetCell(cell_id)
            if cell.GetCellType() != vtk.VTK_TETRA:
                continue
            pts = [
                np.array(delaunay_mesh.GetPoint(cell.GetPointId(k)), dtype=float)
                for k in range(4)
            ]
            p0, p1, p2, p3 = pts
            try:
                radius_sq = vtk.vtkTetra.Circumsphere(tuple(p0), tuple(p1), tuple(p2), tuple(p3), np.zeros(3))
                tetra_radius = float(np.sqrt(max(radius_sq, 0.0)))
            except Exception:
                continue

            if tetra_radius - max_radius > 1e-12:
                max_radius = tetra_radius
                max_radius_cell_id = cell_id
                center = np.zeros(3)
                vtk.vtkTetra.Circumsphere(tuple(p0), tuple(p1), tuple(p2), tuple(p3), center)
                pole = np.array(center, dtype=float)

        if max_radius_cell_id == -1:
            max_radius_cell_id = closest_point_id
            pole = baricenter.copy()

        pole_vector = pole - baricenter
        second_max_radius = 0.0
        second_max_radius_cell_id = -1
        second_pole = pole.copy()

        for j in range(point_cells.GetNumberOfIds()):
            cell_id = point_cells.GetId(j)
            cell = delaunay_mesh.GetCell(cell_id)
            if cell.GetCellType() != vtk.VTK_TETRA:
                continue
            pts = [
                np.array(delaunay_mesh.GetPoint(cell.GetPointId(k)), dtype=float)
                for k in range(4)
            ]
            p0, p1, p2, p3 = pts
            try:
                radius_sq = vtk.vtkTetra.Circumsphere(tuple(p0), tuple(p1), tuple(p2), tuple(p3), np.zeros(3))
                tetra_radius = float(np.sqrt(max(radius_sq, 0.0)))
            except Exception:
                continue

            center = np.zeros(3)
            vtk.vtkTetra.Circumsphere(tuple(p0), tuple(p1), tuple(p2), tuple(p3), center)
            reference_vector = np.array(center, dtype=float) - baricenter

            if (tetra_radius - second_max_radius > 1e-12) and (np.dot(pole_vector, reference_vector) < 1e-12):
                second_max_radius = tetra_radius
                second_max_radius_cell_id = cell_id
                second_pole = np.array(center, dtype=float)

        if second_max_radius_cell_id == -1:
            for j in range(point_cells.GetNumberOfIds()):
                cell_id = point_cells.GetId(j)
                if cell_id == max_radius_cell_id:
                    continue
                cell = delaunay_mesh.GetCell(cell_id)
                if cell.GetCellType() != vtk.VTK_TETRA:
                    continue
                pts = [
                    np.array(delaunay_mesh.GetPoint(cell.GetPointId(k)), dtype=float)
                    for k in range(4)
                ]
                center = np.zeros(3)
                vtk.vtkTetra.Circumsphere(tuple(pts[0]), tuple(pts[1]), tuple(pts[2]), tuple(pts[3]), center)
                reference_vector = np.array(center, dtype=float) - baricenter
                if np.dot(pole_vector, reference_vector) < -1e-12:
                    second_max_radius_cell_id = cell_id
                    second_pole = np.array(center, dtype=float)
                    break

        chosen_cell_id = -1
        chosen_position = baricenter.copy()
        if np.dot(pole_vector, normal) < 1e-12:
            chosen_cell_id = max_radius_cell_id
            chosen_position = pole.copy()
        else:
            chosen_cell_id = second_max_radius_cell_id
            chosen_position = second_pole.copy()

        if chosen_cell_id == -1:
            chosen_cell_id = max_radius_cell_id
            chosen_position = pole.copy()

        seed_voronoi_indices.append(chosen_cell_id)
        seed_positions.append(chosen_position)

    return seed_voronoi_indices, seed_positions


@dataclass
class Pole:
    position: np.ndarray
    radius: float
    clearance: float
    associated_voronoi_node: int
    cap_id: int
    refined: bool = False


@dataclass
class Centerline:
    points: np.ndarray
    radii: np.ndarray
    abscissas: np.ndarray
    tangents: np.ndarray
    curvature: np.ndarray
    torsion: np.ndarray
    tortuosity: float
    frenet_tangent: np.ndarray
    parallel_transport_normals: np.ndarray
    parallel_transport_binormals: np.ndarray
    source_id: int = -1
    target_id: int = -1


@dataclass
class VoronoiGraph:
    points: np.ndarray
    radii: np.ndarray
    edges: np.ndarray
    polys: List[List[int]] = field(default_factory=list)
    polys_edges: List[Tuple[int, int]] = field(default_factory=list)
    adjacency: dict = field(default_factory=dict)


def _numba_or_numpy_edge_cost(points: np.ndarray, radii: np.ndarray, edges: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    xi = np.array([-0.7745966692, 0.0, 0.7745966692])
    wi = np.array([0.5555555556, 0.8888888889, 0.5555555556])

    if NUMBA_AVAILABLE:

        @numba.njit(cache=True)
        def _edge_costs_numba(pts, rads, edgs, fl, xi_n, wi_n):
            result = np.empty(edgs.shape[0], dtype=np.float64)
            for k in range(edgs.shape[0]):
                i, j = int(edgs[k, 0]), int(edgs[k, 1])
                p0, p1 = pts[i], pts[j]
                r0, r1 = rads[i], rads[j]
                length = np.sqrt(np.sum((p1 - p0) ** 2))
                total = 0.0
                for x, w in zip(xi_n, wi_n):
                    a = 0.5 * (x + 1.0)
                    r = (1.0 - a) * r0 + a * r1
                    total += w * max(r, fl)
                result[k] = 0.5 * length * total
            return result

        return _edge_costs_numba(points, radii, edges, floor, xi, wi)

    costs = np.empty(edges.shape[0], dtype=np.float64)
    for k in range(edges.shape[0]):
        i, j = int(edges[k, 0]), int(edges[k, 1])
        p0, p1 = points[i], points[j]
        r0, r1 = radii[i], radii[j]
        length = np.linalg.norm(p1 - p0)
        total = 0.0
        for x, w in zip(xi, wi):
            a = 0.5 * (x + 1.0)
            r = (1.0 - a) * r0 + a * r1
            total += w * max(r, floor)
        costs[k] = 0.5 * length * total
    return costs


def _build_sparse_graph(vertices: np.ndarray, radii: np.ndarray, edges: np.ndarray, floor: float = 1e-6) -> scipy.sparse.csr_matrix:
    edge_weights = _numba_or_numpy_edge_cost(vertices, radii, edges, floor)
    n = vertices.shape[0]
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([edge_weights, edge_weights])
    return scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))


def _backtrack_with_cycle_check(predecessor: np.ndarray, source: int, target: int, max_steps: int = 100000) -> Optional[List[int]]:
    path = [target]
    visited = {target}
    current = target
    steps = 0
    while current != source and steps < max_steps:
        pred = int(predecessor[current])
        if pred == -9999 or pred in visited:
            return None
        visited.add(pred)
        current = pred
        path.insert(0, current)
        steps += 1
    if current != source:
        return None
    return path


def _voxel_fmm_backend(
    mask: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    vertices: np.ndarray,
    radii: np.ndarray,
    source_point: np.ndarray,
    target_point: np.ndarray,
    floor: float = 1e-6,
) -> Optional[np.ndarray]:
    from scipy.interpolate import LinearNDInterpolator

    shape = mask.shape
    speed = np.ones(shape, dtype=np.float64)
    speed[:] = np.inf

    valid_vertices = []
    valid_radii = []
    for i in range(vertices.shape[0]):
        if np.isfinite(vertices[i]).all() and np.isfinite(radii[i]) and radii[i] > floor:
            valid_vertices.append(vertices[i])
            valid_radii.append(radii[i])
    if len(valid_vertices) < 3:
        return None

    valid_vertices = np.array(valid_vertices, dtype=np.float64)
    valid_radii = np.array(valid_radii, dtype=np.float64)

    try:
        interp = LinearNDInterpolator(valid_vertices, 1.0 / np.maximum(valid_radii, floor), fill_value=1.0 / floor)
    except Exception:
        return None

    grid_indices = np.argwhere(mask)
    if grid_indices.shape[0] == 0:
        return None

    physical_points = grid_indices * np.array(spacing) + origin
    try:
        speed_values = interp(physical_points)
    except Exception:
        return None

    speed_values = np.clip(speed_values, 1.0 / (1.0 / floor), None)
    for idx, p in zip(grid_indices, speed_values):
        speed[tuple(idx)] = p

    try:
        src_idx = np.floor((source_point - origin) / np.array(spacing)).astype(int)
        tgt_idx = np.floor((target_point - origin) / np.array(spacing)).astype(int)
        src_idx = np.clip(src_idx, 0, np.array(shape) - 1)
        tgt_idx = np.clip(tgt_idx, 0, np.array(shape) - 1)
        src_ijk = tuple(src_idx)
        tgt_ijk = tuple(tgt_idx)
    except Exception:
        return None

    if not mask[src_ijk] or not mask[tgt_ijk]:
        return None

    import heapq
    dist = np.full(shape, np.inf, dtype=np.float64)
    status = np.full(shape, 0, dtype=np.int32)
    pred = np.full(shape, -1, dtype=int)
    dist[src_ijk] = 0.0
    status[src_ijk] = 2
    heap = [(0.0, src_ijk)]

    offsets = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dz, dy, dx))
    offset_dists = np.array([np.sqrt(dx**2 + dy**2 + dz**2) for dz, dy, dx in offsets], dtype=np.float64)

    while heap:
        t, ijk = heapq.heappop(heap)
        if status[ijk] != 1:
            continue
        status[ijk] = 2
        z, y, x = ijk
        if ijk == tgt_ijk:
            break

        for idx_offset, (dz, dy, dx) in enumerate(offsets):
            nz = z + dz
            ny = y + dy
            nx = x + dx
            if nz < 0 or ny < 0 or nx < 0 or nz >= shape[0] or ny >= shape[1] or nx >= shape[2]:
                continue
            nijk = (nz, ny, nx)
            if not mask[nijk]:
                continue
            if status[nijk] == 2:
                continue
            d = offset_dists[idx_offset]
            F = speed[nijk]
            candidate_t = dist[ijk] + d / max(F, floor)
            if candidate_t < dist[nijk]:
                dist[nijk] = candidate_t
                pred[nijk] = x + y * shape[1] + z * shape[1] * shape[2]
                if status[nijk] == 0:
                    status[nijk] = 1
                    heapq.heappush(heap, (candidate_t, nijk))

    if dist[tgt_ijk] >= np.inf:
        return None

    path_ijk = [tgt_ijk]
    current = tgt_ijk
    visited = {tgt_ijk}
    while current != src_ijk:
        z, y, x = current
        flat_idx = x + y * shape[1] + z * shape[1] * shape[2]
        p_flat = int(pred[current])
        if p_flat < 0:
            return None
        pz = p_flat // (shape[1] * shape[2])
        py = (p_flat % (shape[1] * shape[2])) // shape[1]
        px = p_flat % shape[1]
        parent = (pz, py, px)
        if parent in visited:
            return None
        visited.add(parent)
        current = parent
        path_ijk.insert(0, current)

    path_points = []
    path_radii = []
    for ijk in path_ijk:
        pt = np.array(ijk, dtype=float) * np.array(spacing) + origin
        path_points.append(pt)
        path_radii.append(1.0 / max(speed[ijk], 1e-12))

    return np.array(path_points, dtype=float), np.array(path_radii, dtype=float)


def _subdivide_voronoi_edges(vertices: np.ndarray, radii: np.ndarray, edges: np.ndarray, target_length: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if edges.shape[0] == 0:
        return vertices, radii, edges

    new_pts = [vertices[i].copy() for i in range(vertices.shape[0])]
    new_rads = [float(radii[i]) for i in range(radii.shape[0])]
    new_edges = []
    next_id = vertices.shape[0]

    for k in range(edges.shape[0]):
        i, j = int(edges[k, 0]), int(edges[k, 1])
        p0, p1 = vertices[i], vertices[j]
        r0, r1 = float(radii[i]), float(radii[j])
        d = float(np.linalg.norm(p1 - p0))
        if d < 1e-12:
            new_edges.append([i, j])
            continue
        n_sub = max(1, int(np.ceil(d / target_length)))
        seg_ids = [i]
        for s in range(1, n_sub):
            alpha = s / n_sub
            new_pts.append(p0 + alpha * (p1 - p0))
            new_rads.append((1.0 - alpha) * r0 + alpha * r1)
            seg_ids.append(next_id)
            next_id += 1
        seg_ids.append(j)
        for s in range(len(seg_ids) - 1):
            new_edges.append([seg_ids[s], seg_ids[s + 1]])

    return np.array(new_pts, dtype=float), np.array(new_rads, dtype=float), np.array(new_edges, dtype=int)


def _true_fmm_backend(
    vertices: np.ndarray,
    radii: np.ndarray,
    edges: np.ndarray,
    source: int,
    target: int,
    floor: float = 1e-6,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    n = vertices.shape[0]
    dist = np.full(n, np.inf, dtype=np.float64)
    status = np.full(n, 0, dtype=np.int32)
    pred = np.full(n, -9999, dtype=np.int64)

    adj = {}
    edge_len = {}
    for k in range(edges.shape[0]):
        i, j = int(edges[k, 0]), int(edges[k, 1])
        d = float(np.linalg.norm(vertices[i] - vertices[j]))
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
        edge_len[(i, j)] = d
        edge_len[(j, i)] = d

    dist[source] = 0.0
    status[source] = 2

    import heapq
    heap = []
    for nb in adj.get(source, []):
        if status[nb] == 0:
            d = edge_len.get((source, nb), 1.0)
            r = max(radii[nb] if nb < len(radii) else floor, floor)
            dist[nb] = d / r
            pred[nb] = source
            status[nb] = 1
            heapq.heappush(heap, (dist[nb], nb))

    while heap:
        t, i = heapq.heappop(heap)
        if status[i] != 1:
            continue
        status[i] = 2
        if i == target:
            break

        for nb in adj.get(i, []):
            if status[nb] == 2:
                continue
            candidates = []
            for nb2 in adj.get(nb, []):
                if status[nb2] == 2:
                    d1 = edge_len.get((nb, nb2), 1.0)
                    d2 = edge_len.get((i, nb), 1.0)
                    cos_theta = 0.0
                    v1 = vertices[nb2] - vertices[nb]
                    v2 = vertices[i] - vertices[nb]
                    n1 = np.linalg.norm(v1)
                    n2 = np.linalg.norm(v2)
                    if n1 > 1e-12 and n2 > 1e-12:
                        cos_theta = np.dot(v1, v2) / (n1 * n2)
                    u = dist[nb2] - dist[i]
                    F = max(radii[nb] if nb < len(radii) else floor, floor)
                    a = d1**2 + d2**2 - 2.0 * d1 * d2 * cos_theta
                    b = 2.0 * d2 * u * (d1 * cos_theta - d2)
                    c = d2**2 * (u**2 - (1.0 / F**2) * d1**2 * (1.0 - cos_theta**2))
                    t_candidate = 1e32
                    if abs(a) > 1e-12:
                        disc = b * b - 4.0 * a * c
                        if disc >= -1e-12:
                            if disc < 0:
                                disc = 0.0
                            sqrt_disc = np.sqrt(disc)
                            q = -0.5 * (b + sqrt_disc) if b >= 0 else -0.5 * (b - sqrt_disc)
                            t1 = q / a
                            t2 = c / q if abs(q) > 1e-12 else 1e32
                            t_candidate = min(t1, t2)
                            t_comp = d2 * (t_candidate - u) / t_candidate if abs(t_candidate) > 1e-12 else 1e32
                            t_lower = d1 * cos_theta
                            t_upper = d1 / cos_theta if abs(cos_theta) > 1e-12 else 1e32
                            if (u - t_candidate < -1e-12) or (t_comp - t_lower <= 1e-12) or (t_comp - t_upper >= -1e-12):
                                t_candidate = 1e32
                    if t_candidate < 1e32:
                        candidates.append(t_candidate + dist[i])
                    else:
                        candidates.append(d1 / F + dist[i])
            if candidates:
                new_t = min(candidates)
                if new_t < dist[nb]:
                    dist[nb] = new_t
                    pred[nb] = i
                    if status[nb] == 0:
                        status[nb] = 1
                        heapq.heappush(heap, (dist[nb], nb))
                    elif status[nb] == 1:
                        heapq.heappush(heap, (dist[nb], nb))

    if dist[target] >= np.inf:
        return None, dist

    path = [target]
    visited = {target}
    current = target
    while current != source:
        p = int(pred[current])
        if p == -9999 or p in visited:
            return None, dist
        visited.add(p)
        current = p
        path.insert(0, current)
    return np.array(path, dtype=int), dist


def _build_poly_adjacency(polys: List[List[int]]) -> dict:
    adj = {}
    for poly in polys:
        for i, vi in enumerate(poly):
            for j, vj in enumerate(poly):
                if i != j:
                    adj.setdefault(vi, set()).add(vj)
    return {k: list(v) for k, v in adj.items()}


def _compute_poly_centroids_radii(vertices: np.ndarray, radii: np.ndarray, polys: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    n_polys = len(polys)
    centroids = np.zeros((n_polys, 3), dtype=np.float64)
    poly_radii = np.zeros(n_polys, dtype=np.float64)
    for i, poly in enumerate(polys):
        pts = vertices[poly]
        centroids[i] = pts.mean(axis=0)
        rads = radii[poly]
        valid = rads[rads > 0]
        poly_radii[i] = float(np.mean(valid)) if len(valid) > 0 else 1e-3
    return centroids, poly_radii


def _true_fmm_poly_backend(
    vertices: np.ndarray,
    radii: np.ndarray,
    polys: List[List[int]],
    source_poly: int,
    target_poly: int,
    floor: float = 1e-6,
) -> Tuple[Optional[np.ndarray], np.ndarray, dict]:
    if not polys or source_poly >= len(polys) or target_poly >= len(polys):
        return None, np.full(len(polys), np.inf, dtype=np.float64), {}

    centroids, poly_radii = _compute_poly_centroids_radii(vertices, radii, polys)

    centroids, poly_radii = _compute_poly_centroids_radii(vertices, radii, polys)

    poly_sets = [set(p) for p in polys]
    n_polys = len(polys)
    poly_adj = {}
    poly_edge_len = {}
    for i in range(n_polys):
        for t in poly_sets[i]:
            if t == i or t >= n_polys:
                continue
            if i in poly_sets[t]:
                d = float(np.linalg.norm(centroids[i] - centroids[t]))
                poly_adj.setdefault(i, set()).add(t)
                poly_edge_len[(i, t)] = d

    n = len(polys)
    dist = np.full(n, np.inf, dtype=np.float64)
    status = np.full(n, 0, dtype=np.int32)
    pred = np.full(n, -9999, dtype=np.int64)

    dist[source_poly] = 0.0
    status[source_poly] = 2

    import heapq
    heap = []
    for nb in poly_adj.get(source_poly, []):
        if status[nb] == 0:
            d = poly_edge_len.get((source_poly, nb), 1.0)
            r = max(poly_radii[nb], floor)
            dist[nb] = d / r
            pred[nb] = source_poly
            status[nb] = 1
            heapq.heappush(heap, (dist[nb], nb))

    while heap:
        t, i = heapq.heappop(heap)
        if status[i] != 1:
            continue
        status[i] = 2
        if i == target_poly:
            break

        for nb in poly_adj.get(i, []):
            if status[nb] == 2:
                continue
            candidates = []
            for nb2 in poly_adj.get(nb, []):
                if status[nb2] == 2:
                    d1 = poly_edge_len.get((nb, nb2), 1.0)
                    d2 = poly_edge_len.get((i, nb), 1.0)
                    v1 = centroids[nb2] - centroids[nb]
                    v2 = centroids[i] - centroids[nb]
                    n1 = np.linalg.norm(v1)
                    n2 = np.linalg.norm(v2)
                    cos_theta = 0.0
                    if n1 > 1e-12 and n2 > 1e-12:
                        cos_theta = np.dot(v1, v2) / (n1 * n2)
                    u = dist[nb2] - dist[i]
                    F = max(poly_radii[nb], floor)
                    a = d1**2 + d2**2 - 2.0 * d1 * d2 * cos_theta
                    b = 2.0 * d2 * u * (d1 * cos_theta - d2)
                    c = d2**2 * (u**2 - (1.0 / F**2) * d1**2 * (1.0 - cos_theta**2))
                    t_candidate = 1e32
                    if abs(a) > 1e-12:
                        disc = b * b - 4.0 * a * c
                        if disc >= -1e-12:
                            if disc < 0:
                                disc = 0.0
                            sqrt_disc = np.sqrt(disc)
                            q = -0.5 * (b + sqrt_disc) if b >= 0 else -0.5 * (b - sqrt_disc)
                            t1 = q / a
                            t2 = c / q if abs(q) > 1e-12 else 1e32
                            t_candidate = min(t1, t2)
                            t_comp = d2 * (t_candidate - u) / t_candidate if abs(t_candidate) > 1e-12 else 1e32
                            t_lower = d1 * cos_theta
                            t_upper = d1 / cos_theta if abs(cos_theta) > 1e-12 else 1e32
                            if (u - t_candidate < -1e-12) or (t_comp - t_lower <= 1e-12) or (t_comp - t_upper >= -1e-12):
                                t_candidate = 1e32
                    if t_candidate < 1e32:
                        candidates.append(t_candidate + dist[i])
                    else:
                        candidates.append(d1 / F + dist[i])
            if candidates:
                new_t = min(candidates)
                if new_t < dist[nb]:
                    dist[nb] = new_t
                    pred[nb] = i
                    if status[nb] == 0:
                        status[nb] = 1
                        heapq.heappush(heap, (dist[nb], nb))
                    elif status[nb] == 1:
                        heapq.heappush(heap, (dist[nb], nb))

    path = None
    if dist[target_poly] < np.inf:
        path = [target_poly]
        visited = {target_poly}
        current = target_poly
        while current != source_poly:
            p = int(pred[current])
            if p == -9999 or p in visited:
                path = None
                break
            visited.add(p)
            current = p
            path.insert(0, current)

    return path, dist, pred


def _trace_centerline_steepest_descent(
    dist: np.ndarray,
    vertices: np.ndarray,
    radii: np.ndarray,
    edges: np.ndarray,
    source: int,
    target: int,
    step_size: float = 0.2,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if dist[target] >= np.inf:
        return None

    adj = {}
    for k in range(edges.shape[0]):
        i, j = int(edges[k, 0]), int(edges[k, 1])
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    pts = [vertices[target].copy()]
    rads = [float(radii[target]) if target < len(radii) else 1e-3]
    current = target
    max_steps = 50000
    steps = 0
    visited = {target}

    while current != source and steps < max_steps:
        steps += 1
        neighbors = adj.get(current, [])
        if not neighbors:
            break

        downhill = []
        for nb in neighbors:
            if nb in visited and nb != source:
                continue
            t0 = dist[current]
            t1 = dist[nb]
            if t1 >= t0:
                continue
            downhill.append((t1, nb))

        if not downhill:
            break

        downhill.sort(key=lambda x: x[0])
        best_nb = downhill[0][1]

        p0 = vertices[current]
        p1 = vertices[best_nb]
        d = float(np.linalg.norm(p1 - p0))
        if d < 1e-12:
            current = best_nb
            visited.add(current)
            pts.append(vertices[current].copy())
            rads.append(float(radii[current]) if current < len(radii) else 1e-3)
            continue

        n_samples = max(2, int(np.ceil(d / step_size)))
        for s in range(1, n_samples + 1):
            alpha = s / n_samples
            p = p0 + alpha * (p1 - p0)
            r0 = float(radii[current]) if current < len(radii) else 1e-3
            r1 = float(radii[best_nb]) if best_nb < len(radii) else 1e-3
            r = (1.0 - alpha) * r0 + alpha * r1
            pts.append(p.copy())
            rads.append(r)

        current = best_nb
        visited.add(current)
        if current != source:
            pts.append(vertices[current].copy())
            rads.append(float(radii[current]) if current < len(radii) else 1e-3)

    if current != source:
        return None

    return np.array(pts, dtype=float), np.array(rads, dtype=float)


def _trace_centerline_poly_continuous(
    dist: np.ndarray,
    pred: dict,
    vertices: np.ndarray,
    radii: np.ndarray,
    polys: List[List[int]],
    source_poly: int,
    target_poly: int,
    n_subdivisions: int = 250,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not polys or source_poly >= len(polys) or target_poly >= len(polys):
        return None
    if dist[target_poly] >= np.inf:
        return None

    path_polys = [target_poly]
    visited = {target_poly}
    current = target_poly
    while current != source_poly:
        p = int(pred[current]) if current < len(pred) else -9999
        if p == -9999 or p in visited:
            return None
        visited.add(p)
        current = p
        path_polys.insert(0, current)

    all_pts = []
    all_rads = []

    for idx in range(len(path_polys) - 1):
        pi = path_polys[idx]
        pj = path_polys[idx + 1]
        poly_i = polys[pi]
        poly_j = polys[pj]
        shared = list(set(poly_i) & set(poly_j))
        if not shared:
            ci = vertices[poly_i].mean(axis=0)
            cj = vertices[poly_j].mean(axis=0)
            d = np.linalg.norm(cj - ci)
            for s in range(n_subdivisions):
                alpha = s / n_subdivisions
                pt = ci + alpha * (cj - ci)
                all_pts.append(pt)
                ri = float(np.mean(radii[poly_i])) if len(poly_i) > 0 else 1e-3
                rj = float(np.mean(radii[poly_j])) if len(poly_j) > 0 else 1e-3
                all_rads.append((1.0 - alpha) * ri + alpha * rj)
            continue

        bridge_v = shared[0]
        ci = vertices[poly_i].mean(axis=0)
        cj = vertices[poly_j].mean(axis=0)
        t_i = dist[pi]
        t_j = dist[pj]

        n_seg = n_subdivisions
        for s in range(n_seg + 1):
            alpha = s / n_seg
            pt = ci + alpha * (cj - ci)
            all_pts.append(pt)
            ri = float(np.mean(radii[poly_i])) if len(poly_i) > 0 else 1e-3
            rj = float(np.mean(radii[poly_j])) if len(poly_j) > 0 else 1e-3
            all_rads.append((1.0 - alpha) * ri + alpha * rj)

    if len(all_pts) < 2:
        return None

    return np.array(all_pts, dtype=float), np.array(all_rads, dtype=float)


def _find_best_poly_for_vertex(vertex_idx: int, graph: VoronoiGraph) -> int:
    if not graph.polys:
        return -1
    best_poly = -1
    best_score = -np.inf
    for i, poly in enumerate(graph.polys):
        if vertex_idx in poly:
            rads = graph.radii[poly]
            valid = rads[rads > 0]
            score = float(np.mean(valid)) if len(valid) > 0 else 0.0
            if score > best_score:
                best_score = score
                best_poly = i
    return best_poly


def _compute_centerline_geometry(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = points.shape[0]
    if n < 2:
        empty = np.empty((n, 3), dtype=float)
        return empty, empty, np.zeros(n, dtype=float), empty, empty, empty, np.zeros(n, dtype=float)

    tangents = np.zeros_like(points)
    for i in range(n):
        if i == 0:
            t = points[1] - points[0]
        elif i == n - 1:
            t = points[-1] - points[-2]
        else:
            t = points[i + 1] - points[i - 1]
        norm = np.linalg.norm(t)
        if norm > 1e-9:
            t = t / norm
        tangents[i] = t

    abscissas = np.zeros(n, dtype=float)
    for i in range(1, n):
        abscissas[i] = abscissas[i - 1] + np.linalg.norm(points[i] - points[i - 1])

    frenet_tangent = tangents.copy()
    curvature = np.zeros(n, dtype=float)
    torsion = np.zeros(n, dtype=float)
    parallel_transport_normals = np.zeros_like(points)
    parallel_transport_binormals = np.zeros_like(points)

    if n >= 3:
        normal = np.array([0.0, 1.0, 0.0], dtype=float)
        if np.linalg.norm(np.cross(tangents[0], normal)) < 1e-6:
            normal = np.array([1.0, 0.0, 0.0], dtype=float)
        n0 = np.cross(tangents[0], normal)
        n0_norm = np.linalg.norm(n0)
        if n0_norm > 1e-12:
            n0 = n0 / n0_norm
        else:
            n0 = np.array([0.0, 0.0, 1.0], dtype=float)
        parallel_transport_normals[0] = n0
        parallel_transport_binormals[0] = np.cross(tangents[0], n0)

        for i in range(1, n):
            t_prev = tangents[i - 1]
            t_curr = tangents[i]
            n_prev = parallel_transport_normals[i - 1]
            b_prev = parallel_transport_binormals[i - 1]

            cos_angle = np.dot(t_prev, t_curr)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = np.arccos(cos_angle)
            curvature[i] = angle / max(abscissas[i] - abscissas[i - 1], 1e-9)

            if angle > 1e-9:
                axis = np.cross(t_prev, t_curr)
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-9:
                    axis = axis / axis_norm
                    n_curr = n_prev * np.cos(angle) + np.cross(axis, n_prev) * np.sin(angle) + axis * np.dot(axis, n_prev) * (1 - np.cos(angle))
                    n_curr = n_curr / np.linalg.norm(n_curr)
                else:
                    n_curr = n_prev
            else:
                n_curr = n_prev

            parallel_transport_normals[i] = n_curr
            parallel_transport_binormals[i] = np.cross(t_curr, n_curr)

            if i < n - 1:
                t_next = tangents[i + 1]
                d1 = np.cross(t_curr, t_next)
                d2 = np.cross(t_prev, t_curr)
                denom = np.linalg.norm(d1) * np.linalg.norm(d2)
                if denom > 1e-9:
                    torsion[i] = np.arctan2(np.dot(d1, n_curr), denom)

    tortuosity = float(abscissas[-1]) if abscissas[-1] > 1e-9 else 0.0
    if n >= 2:
        straight = np.linalg.norm(points[-1] - points[0])
        if straight > 1e-9:
            tortuosity = tortuosity / straight

    return tangents, curvature, torsion, parallel_transport_normals, parallel_transport_binormals, frenet_tangent, abscissas


class vmtkFastMarchingLocal(vmtkBaseScript):
    def __init__(self):
        super().__init__()
        self.VoronoiDiagram: Optional[VoronoiGraph] = None
        self.CapCenters: Optional[np.ndarray] = None
        self.CapNormals: Optional[np.ndarray] = None
        self.SeedPositions: Optional[np.ndarray] = None
        self.SeedVoronoiIds: Optional[List[int]] = None
        self.InternalVolumeMask: Optional[np.ndarray] = None
        self.VoxelSpacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.VolumeOrigin: Optional[np.ndarray] = None
        self.SourceIds: List[int] = []
        self.TargetIds: List[int] = []
        self.Poles: Optional[List[Pole]] = None
        self.Centerlines: Optional[List[Centerline]] = None
        self.Backend: str = "dijkstra"
        self.RadiusFloor: float = 1e-12
        self.EikonalRelaxationIters: int = 50

    def Execute(self) -> None:
        if self.VoronoiDiagram is None:
            self.PrintError("Error: No Voronoi diagram.")
            return
        if self.SeedPositions is None and (self.CapCenters is None or self.CapNormals is None):
            self.PrintError("Error: Seed positions or cap centers and normals must be set.")
            return
        if self.InternalVolumeMask is None or self.VolumeOrigin is None:
            self.PrintError("Error: Internal volume mask and origin must be set.")
            return

        poles = self._compute_edt_poles()
        if not poles:
            self.PrintError("No poles computed.")
            return
        self.Poles = poles

        points = self.VoronoiDiagram.points
        radii = self.VoronoiDiagram.radii
        edges = self.VoronoiDiagram.edges
        voronoi_polys = self.VoronoiDiagram.polys
        voronoi_polys_edges = self.VoronoiDiagram.polys_edges

        if self.Backend != "python_fmm_poly" or not voronoi_polys:
            points, radii, edges = _subdivide_voronoi_edges(points, radii, edges, target_length=0.5)

        graph = VoronoiGraph(
            points=points,
            radii=radii,
            edges=edges,
            polys=voronoi_polys,
            polys_edges=voronoi_polys_edges,
        )
        sparse_graph = _build_sparse_graph(graph.points, graph.radii, graph.edges, self.RadiusFloor)
        tree = cKDTree(graph.points)

        centerlines = []
        centerline_pairs = []
        seed_positions = self.SeedPositions if self.SeedPositions is not None else self.CapCenters
        if seed_positions is not None and len(seed_positions) == 0 and poles:
            for i in range(len(poles) - 1):
                src_pole = poles[i]
                tgt_pole = poles[i + 1]
                src_vor = int(src_pole.associated_voronoi_node)
                tgt_vor = int(tgt_pole.associated_voronoi_node)
                if src_vor >= graph.points.shape[0] or tgt_vor >= graph.points.shape[0]:
                    centerlines.append(self._build_centerline(
                        np.vstack([src_pole.position, tgt_pole.position]),
                        np.array([src_pole.radius, tgt_pole.radius], dtype=float),
                    ))
                    centerline_pairs.append((i, i + 1))
                    continue

                dists: Optional[np.ndarray] = None
                predecessor: Optional[np.ndarray] = None

                if self.Backend == "dijkstra":
                    dists, predecessor = dijkstra(sparse_graph, indices=src_vor, directed=False, return_predecessors=True)
                    path = _backtrack_with_cycle_check(predecessor, src_vor, tgt_vor)
                elif self.Backend == "python_fmm":
                    fmm_path, fmm_dists = _true_fmm_backend(graph.points, graph.radii, graph.edges, src_vor, tgt_vor, self.RadiusFloor)
                    dists, predecessor = dijkstra(sparse_graph, indices=src_vor, directed=False, return_predecessors=True)
                    path = fmm_path if fmm_path is not None else _backtrack_with_cycle_check(predecessor, src_vor, tgt_vor)
                elif self.Backend == "python_fmm_poly" and len(graph.polys) > 0:
                    src_poly = _find_best_poly_for_vertex(src_vor, graph)
                    tgt_poly = _find_best_poly_for_vertex(tgt_vor, graph)
                    if src_poly >= 0 and tgt_poly >= 0:
                        path_polys, dists_poly, pred_poly = _true_fmm_poly_backend(graph.points, graph.radii, graph.polys, src_poly, tgt_poly, self.RadiusFloor)
                        if path_polys is not None:
                            result = _trace_centerline_poly_continuous(dists_poly, pred_poly, graph.points, graph.radii, graph.polys, src_poly, tgt_poly)
                            if result is not None:
                                pts, rads = result
                                centerlines.append(self._build_centerline(pts, rads))
                                continue
                    logger.warning("Poly FMM failed for pole pair, falling back to dijkstra")
                    dists, predecessor = dijkstra(sparse_graph, indices=src_vor, directed=False, return_predecessors=True)
                    path = _backtrack_with_cycle_check(predecessor, src_vor, tgt_vor)
                elif self.Backend == "voxel_fmm":
                    result = _voxel_fmm_backend(self.InternalVolumeMask, self.VolumeOrigin, self.VoxelSpacing, graph.points, graph.radii, src_pole.position, tgt_pole.position, self.RadiusFloor)
                    if result is not None:
                        pts, rads = result
                        centerlines.append(self._build_centerline(pts, rads))
                        continue
                    dists, predecessor = dijkstra(sparse_graph, indices=src_vor, directed=False, return_predecessors=True)
                    path = _backtrack_with_cycle_check(predecessor, src_vor, tgt_vor)
                elif self.Backend == "python_eikonal":
                    dists, predecessor = dijkstra(sparse_graph, indices=src_vor, directed=False, return_predecessors=True)
                    path = _backtrack_with_cycle_check(predecessor, src_vor, tgt_vor)
                else:
                    dists, predecessor = dijkstra(sparse_graph, indices=src_vor, directed=False, return_predecessors=True)
                    path = _backtrack_with_cycle_check(predecessor, src_vor, tgt_vor)

                if path is None:
                    centerlines.append(self._build_centerline(
                        np.vstack([src_pole.position, tgt_pole.position]),
                        np.array([src_pole.radius, tgt_pole.radius], dtype=float),
                    ))
                    continue

                if self.Backend in ("python_eikonal", "python_fmm") and len(path) >= 2:
                    result = _trace_centerline_steepest_descent(dists, graph.points, graph.radii, graph.edges, src_vor, tgt_vor, step_size=0.2)
                    if result is not None:
                        pts, rads = result
                    else:
                        logger.warning("Steepest descent tracing failed for pole pair, falling back to discrete path")
                        pts = graph.points[path]
                        rads = graph.radii[path]
                else:
                    pts = graph.points[path]
                    rads = graph.radii[path]
                centerlines.append(self._build_centerline(pts, rads))
        else:
            use_seed_voronoi = self.SeedVoronoiIds is not None and len(self.SeedVoronoiIds) == len(seed_positions) if seed_positions is not None else False
            for src_id in self.SourceIds:
                for tgt_id in self.TargetIds:
                    if src_id == tgt_id:
                        continue
                    if use_seed_voronoi and src_id < len(self.SeedVoronoiIds) and tgt_id < len(self.SeedVoronoiIds):
                        src_vor = self.SeedVoronoiIds[src_id]
                        tgt_vor = self.SeedVoronoiIds[tgt_id]
                        if src_vor < 0 or tgt_vor < 0 or src_vor >= graph.points.shape[0] or tgt_vor >= graph.points.shape[0]:
                            centerlines.append(self._fallback_centerline(src_id, tgt_id))
                            continue
                    else:
                        src_pt = seed_positions[src_id] if seed_positions is not None and src_id < len(seed_positions) else graph.points[src_id]
                        tgt_pt = seed_positions[tgt_id] if seed_positions is not None and tgt_id < len(seed_positions) else graph.points[tgt_id]
                        src_dist, src_idx = tree.query(src_pt)
                        tgt_dist, tgt_idx = tree.query(tgt_pt)
                        src_vor = int(src_idx)
                        tgt_vor = int(tgt_idx)

                    dists: Optional[np.ndarray] = None
                    predecessor: Optional[np.ndarray] = None

                    if self.Backend == "dijkstra":
                        dists, predecessor = dijkstra(sparse_graph, indices=src_vor, directed=False, return_predecessors=True)
                        path = _backtrack_with_cycle_check(predecessor, src_vor, tgt_vor)
                    elif self.Backend == "python_fmm":
                        fmm_path, fmm_dists = _true_fmm_backend(graph.points, graph.radii, graph.edges, src_vor, tgt_vor, self.RadiusFloor)
                        dists, predecessor = dijkstra(sparse_graph, indices=src_vor, directed=False, return_predecessors=True)
                        path = fmm_path if fmm_path is not None else _backtrack_with_cycle_check(predecessor, src_vor, tgt_vor)
                    elif self.Backend == "python_fmm_poly" and len(graph.polys) > 0:
                        src_poly = _find_best_poly_for_vertex(src_vor, graph)
                        tgt_poly = _find_best_poly_for_vertex(tgt_vor, graph)
                        if src_poly >= 0 and tgt_poly >= 0:
                            path_polys, dists_poly, pred_poly = _true_fmm_poly_backend(graph.points, graph.radii, graph.polys, src_poly, tgt_poly, self.RadiusFloor)
                            if path_polys is not None:
                                result = _trace_centerline_poly_continuous(dists_poly, pred_poly, graph.points, graph.radii, graph.polys, src_poly, tgt_poly)
                                if result is not None:
                                    pts, rads = result
                                    centerlines.append(self._build_centerline(pts, rads))
                                    continue
                        logger.warning("Poly FMM failed for source %d target %d, falling back to dijkstra", src_id, tgt_id)
                        dists, predecessor = dijkstra(sparse_graph, indices=src_vor, directed=False, return_predecessors=True)
                        path = _backtrack_with_cycle_check(predecessor, src_vor, tgt_vor)
                    elif self.Backend == "voxel_fmm":
                        src_pt = seed_positions[src_id] if seed_positions is not None and src_id < len(seed_positions) else graph.points[src_vor]
                        tgt_pt = seed_positions[tgt_id] if seed_positions is not None and tgt_id < len(seed_positions) else graph.points[tgt_vor]
                        result = _voxel_fmm_backend(self.InternalVolumeMask, self.VolumeOrigin, self.VoxelSpacing, graph.points, graph.radii, src_pt, tgt_pt, self.RadiusFloor)
                        if result is not None:
                            pts, rads = result
                            centerlines.append(self._build_centerline(pts, rads))
                            continue
                        dists, predecessor = dijkstra(sparse_graph, indices=src_vor, directed=False, return_predecessors=True)
                        path = _backtrack_with_cycle_check(predecessor, src_vor, tgt_vor)
                    elif self.Backend == "python_eikonal":
                        dists, predecessor = dijkstra(sparse_graph, indices=src_vor, directed=False, return_predecessors=True)
                        path = _backtrack_with_cycle_check(predecessor, src_vor, tgt_vor)
                    else:
                        self.PrintLog(f"Unknown backend {self.Backend}, falling back to dijkstra")
                        dists, predecessor = dijkstra(sparse_graph, indices=src_vor, directed=False, return_predecessors=True)
                        path = _backtrack_with_cycle_check(predecessor, src_vor, tgt_vor)

                    if path is None:
                        logger.warning("No path found between source %d and target %d, using fallback", src_id, tgt_id)
                        centerlines.append(self._fallback_centerline(src_id, tgt_id))
                        continue

                    if self.Backend in ("python_eikonal", "python_fmm") and len(path) >= 2:
                        result = _trace_centerline_steepest_descent(dists, graph.points, graph.radii, graph.edges, src_vor, tgt_vor, step_size=0.2)
                        if result is not None:
                            pts, rads = result
                        else:
                            logger.warning("Steepest descent tracing failed for source %d target %d, falling back to discrete path", src_id, tgt_id)
                            pts = graph.points[path]
                            rads = graph.radii[path]
                    else:
                        pts = graph.points[path]
                        rads = graph.radii[path]
                    centerlines.append(self._build_centerline(pts, rads))

        if not centerlines:
            self.PrintError("No centerline paths found.")
            return
        self.Centerlines = centerlines
        self.PrintLog(f"Centerlines computed: {len(centerlines)} paths")

    def _compute_edt_poles(self) -> List[Pole]:
        mask = self.InternalVolumeMask
        spacing = self.VoxelSpacing
        origin = self.VolumeOrigin
        points = self.VoronoiDiagram.points
        voronoi_radii = self.VoronoiDiagram.radii
        cap_centers = self.CapCenters
        cap_normals = self.CapNormals

        clearance = scipy.ndimage.distance_transform_edt(mask, sampling=spacing)
        local_maxima = (clearance == scipy.ndimage.maximum_filter(clearance, size=5)) & mask

        maxima_coords = np.argwhere(local_maxima)
        if maxima_coords.shape[0] == 0:
            self.PrintLog("No local maxima found in EDT, using cap centers as fallback poles")
            poles = []
            for cap_id in range(cap_centers.shape[0]):
                pos = cap_centers[cap_id]
                vor_idx = int(np.argmin(np.sum((points - pos) ** 2, axis=1)))
                poles.append(Pole(position=pos.copy(), radius=voronoi_radii[vor_idx] if vor_idx < len(voronoi_radii) else 1e-3, clearance=0.0, associated_voronoi_node=vor_idx, cap_id=cap_id, refined=False))
            return poles

        maxima_physical = maxima_coords * np.array(spacing) + origin
        tree = cKDTree(maxima_physical)
        poles = []
        if cap_centers.shape[0] == 0:
            for idx in range(maxima_physical.shape[0]):
                pos = maxima_physical[idx]
                cl = float(clearance[maxima_coords[idx, 0], maxima_coords[idx, 1], maxima_coords[idx, 2]])
                vor_idx = int(np.argmin(np.sum((points - pos) ** 2, axis=1)))
                radius = voronoi_radii[vor_idx] if vor_idx < len(voronoi_radii) else 1e-3
                poles.append(Pole(position=pos.copy(), radius=float(radius), clearance=float(cl), associated_voronoi_node=vor_idx, cap_id=-1, refined=False))
            if len(poles) >= 2:
                return poles
            self.PrintLog("No EDT maxima usable as poles")
            return []
        for cap_id in range(cap_centers.shape[0]):
            cap_pos = cap_centers[cap_id]
            cap_normal = cap_normals[cap_id]
            query_center = cap_pos - 2.0 * cap_normal
            corridor_radius = 3.0 * np.max(spacing)
            candidates = tree.query_ball_point(query_center, r=corridor_radius * 5.0)
            if not candidates:
                dists, idxs = tree.query(query_center, k=min(3, maxima_physical.shape[0]))
                if np.isscalar(idxs):
                    idxs = [int(idxs)]
                else:
                    idxs = [int(i) for i in idxs]
                candidates = idxs

            best_pole = None
            best_score = -np.inf
            for idx in candidates:
                pos = maxima_physical[idx]
                cl = clearance[maxima_coords[idx, 0], maxima_coords[idx, 1], maxima_coords[idx, 2]]
                inward = np.dot(pos - cap_pos, -cap_normal)
                if inward < 0:
                    continue
                dist_to_vor = np.min(np.sum((points - pos) ** 2, axis=1))
                vor_idx = int(np.argmin(np.sum((points - pos) ** 2, axis=1)))
                radius = voronoi_radii[vor_idx] if vor_idx < len(voronoi_radii) else 1e-3
                score = cl + 0.1 * inward - 0.01 * dist_to_vor
                if score > best_score:
                    best_score = score
                    best_pole = Pole(position=pos.copy(), radius=float(radius), clearance=float(cl), associated_voronoi_node=vor_idx, cap_id=cap_id, refined=False)

            if best_pole is None:
                vor_idx = int(np.argmin(np.sum((points - cap_pos) ** 2, axis=1)))
                best_pole = Pole(position=cap_pos.copy(), radius=float(voronoi_radii[vor_idx]) if vor_idx < len(voronoi_radii) else 1e-3, clearance=0.0, associated_voronoi_node=vor_idx, cap_id=cap_id, refined=False)
            poles.append(best_pole)

        return poles

    def _build_centerline(self, pts: np.ndarray, rads: np.ndarray, source_id: int = -1, target_id: int = -1) -> Centerline:
        tangents, curvature, torsion, parallel_transport_normals, parallel_transport_binormals, frenet_tangent, abscissas = _compute_centerline_geometry(pts)
        tortuosity = float(abscissas[-1]) if abscissas[-1] > 1e-9 else 0.0
        if pts.shape[0] >= 2:
            straight = np.linalg.norm(pts[-1] - pts[0])
            if straight > 1e-9:
                tortuosity = tortuosity / straight
        return Centerline(points=pts, radii=rads, abscissas=abscissas, tangents=tangents, curvature=curvature, torsion=torsion, tortuosity=tortuosity, frenet_tangent=frenet_tangent, parallel_transport_normals=parallel_transport_normals, parallel_transport_binormals=parallel_transport_binormals, source_id=source_id, target_id=target_id)

    def _fallback_centerline(self, src_id: int, tgt_id: int) -> Centerline:
        logger.warning("Using fallback straight-line centerline between cap %d and cap %d", src_id, tgt_id)
        p0 = self.CapCenters[src_id]
        p1 = self.CapCenters[tgt_id]
        pts = np.vstack([p0, p1])
        rads = np.array([1e-3, 1e-3], dtype=float)
        return self._build_centerline(pts, rads, source_id=src_id, target_id=tgt_id)
