import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import vtk
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


@dataclass
class VoronoiGraph:
    points: np.ndarray
    radii: np.ndarray
    edges: np.ndarray
    polys: List[List[int]] = field(default_factory=list)
    polys_edges: List[Tuple[int, int]] = field(default_factory=list)
    n_points: int = 0
    n_edges: int = 0
    warnings: List[str] = field(default_factory=list)


def build_voronoi_from_tetrahedra(tetrahedra: List, surface: Optional[vtk.vtkPolyData] = None, acceleration: str = "auto", internal_only: bool = False) -> VoronoiGraph:
    if not tetrahedra:
        return VoronoiGraph(points=np.array([]), radii=np.array([]), edges=np.array([]).reshape(0, 2))

    max_cell_id = max(t.cell_id for t in tetrahedra)
    centers = np.full((max_cell_id + 1, 3), np.nan, dtype=float)
    radii = np.full(max_cell_id + 1, np.nan, dtype=float)
    internal_mask = np.zeros(max_cell_id + 1, dtype=bool)
    for t in tetrahedra:
        centers[t.cell_id] = t.circumcenter
        radii[t.cell_id] = t.circumradius
        if not internal_only or t.is_internal:
            internal_mask[t.cell_id] = True

    tets = np.array([t.point_ids for t in tetrahedra], dtype=int)
    tets_sorted = np.sort(tets, axis=1)

    faces = np.stack([
        tets_sorted[:, [0, 1, 2]],
        tets_sorted[:, [0, 1, 3]],
        tets_sorted[:, [0, 2, 3]],
        tets_sorted[:, [1, 2, 3]],
    ], axis=1).reshape(-1, 3)

    faces = np.sort(faces, axis=1)
    unique_faces, inverse, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)

    adj = {}
    for i, face in enumerate(unique_faces):
        if counts[i] == 2:
            idx = np.where(inverse == i)[0]
            if len(idx) == 2:
                t0_cell_id = tetrahedra[idx[0] // 4].cell_id
                t1_cell_id = tetrahedra[idx[1] // 4].cell_id
                if internal_mask[t0_cell_id] and internal_mask[t1_cell_id]:
                    adj.setdefault(t0_cell_id, set()).add(t1_cell_id)
                    adj.setdefault(t1_cell_id, set()).add(t0_cell_id)

    edges_list = []
    for src, tgts in adj.items():
        for tgt in tgts:
            if src < tgt:
                edges_list.append((src, tgt))

    if not edges_list:
        edges = np.array([]).reshape(0, 2)
    else:
        edges = np.array(edges_list, dtype=int)

    valid = internal_mask & (~np.isnan(radii))
    n_valid = int(np.sum(valid))
    mean_r = float(np.nanmean(radii[valid])) if np.any(valid) else 0.0
    logger.info("Built Voronoi graph: %d points, %d edges, mean_radius=%.4f", n_valid, len(edges), mean_r)
    result = VoronoiGraph(points=centers, radii=radii, edges=edges, n_points=max_cell_id + 1, n_edges=len(edges))

    polys, polys_edges = _build_voronoi_polys(tetrahedra, internal_mask)
    result.polys = polys
    result.polys_edges = polys_edges
    return result


def _build_voronoi_polys(tetrahedra: List, cell_id_to_idx: Dict[int, int]) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    n = len(tetrahedra)
    if n == 0:
        return [], []

    tet_point_ids = np.array([t.point_ids for t in tetrahedra], dtype=int)
    cell_ids = np.array([t.cell_id for t in tetrahedra], dtype=int)

    point_to_tets: Dict[int, List[int]] = {}
    for i in range(n):
        for pid in tet_point_ids[i]:
            point_to_tets.setdefault(int(pid), []).append(i)

    face_to_tets: Dict[tuple, List[int]] = {}
    for i in range(n):
        pts = sorted(tet_point_ids[i].tolist())
        for a in range(4):
            for b in range(a + 1, 4):
                for c in range(b + 1, 4):
                    f = (pts[a], pts[b], pts[c])
                    face_to_tets.setdefault(f, []).append(i)

    polys: List[List[int]] = []
    polys_edges: List[Tuple[int, int]] = []

    for i in range(n):
        neighbor_faces = []
        pts = sorted(tet_point_ids[i].tolist())
        for a in range(4):
            for b in range(a + 1, 4):
                for c in range(b + 1, 4):
                    f = (pts[a], pts[b], pts[c])
                    tets_on_face = face_to_tets.get(f, [])
                    if len(tets_on_face) == 2:
                        other = tets_on_face[0] if tets_on_face[1] == i else tets_on_face[1]
                        neighbor_faces.append(other)

        if not neighbor_faces:
            continue

        neighbors = sorted(set(neighbor_faces))
        poly = [cell_ids[i]] + [cell_ids[nb] for nb in neighbors]
        if len(poly) >= 3:
            polys.append(poly)
            for k in range(len(poly)):
                e = (poly[k], poly[(k + 1) % len(poly)])
                polys_edges.append(e)

    return polys, polys_edges


def filter_voronoi_by_clearance(voronoi: VoronoiGraph, surface: vtk.vtkPolyData, clearance_threshold: float = 0.0, radius_floor: float = 1e-12) -> VoronoiGraph:
    if len(voronoi.points) == 0:
        return voronoi

    keep_mask = np.ones(len(voronoi.points), dtype=bool)
    if radius_floor > 0:
        keep_mask &= (voronoi.radii >= radius_floor) & (~np.isnan(voronoi.radii))

    new_points = voronoi.points.copy()
    new_radii = voronoi.radii.copy()
    new_points[~keep_mask] = 0.0
    new_radii[~keep_mask] = -1.0

    logger.info("Filtered Voronoi: %d points, %d edges, %d polys", int(np.sum(keep_mask)), len(voronoi.edges), len(voronoi.polys))
    return VoronoiGraph(
        points=new_points,
        radii=new_radii,
        edges=voronoi.edges,
        n_points=voronoi.n_points,
        n_edges=voronoi.n_edges,
        polys=voronoi.polys,
        polys_edges=voronoi.polys_edges,
    )


def extract_seed_component(voronoi: VoronoiGraph, seed_index: int) -> VoronoiGraph:
    if voronoi.n_points == 0 or seed_index < 0 or seed_index >= voronoi.n_points:
        return voronoi

    adj = {}
    for i in range(voronoi.n_points):
        adj[i] = []
    for e in voronoi.edges:
        adj[int(e[0])].append(int(e[1]))
        adj[int(e[1])].append(int(e[0]))

    visited = set()
    stack = [seed_index]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                stack.append(neighbor)

    index_map = {}
    new_idx = 0
    for old in range(voronoi.n_points):
        if old in visited:
            index_map[old] = new_idx
            new_idx += 1

    new_edges = []
    for e in voronoi.edges:
        if e[0] in index_map and e[1] in index_map:
            new_edges.append([index_map[e[0]], index_map[e[1]]])

    new_polys = []
    for poly in voronoi.polys:
        new_poly = [index_map[i] for i in poly if i in index_map]
        if len(new_poly) >= 3:
            new_polys.append(new_poly)

    new_polys_edges = []
    for e in voronoi.polys_edges:
        if e[0] in index_map and e[1] in index_map:
            new_polys_edges.append((index_map[e[0]], index_map[e[1]]))

    new_edges_arr = np.array(new_edges, dtype=int) if new_edges else np.array([]).reshape(0, 2)
    logger.info("Extracted seed component: %d points, %d edges, %d polys", len(visited), len(new_edges_arr), len(new_polys))
    return VoronoiGraph(
        points=voronoi.points[list(visited)],
        radii=voronoi.radii[list(visited)],
        edges=new_edges_arr,
        n_points=len(visited),
        n_edges=len(new_edges_arr),
        polys=new_polys,
        polys_edges=new_polys_edges,
    )


def simplify_voronoi(voronoi: VoronoiGraph, unremovable_indices: Optional[List[int]] = None) -> VoronoiGraph:
    if voronoi.n_points == 0:
        return voronoi

    unremovable = set(unremovable_indices or [])
    adj = {}
    for e in voronoi.edges:
        adj.setdefault(int(e[0]), set()).add(int(e[1]))
        adj.setdefault(int(e[1]), set()).add(int(e[0]))

    to_remove = []
    for i in range(voronoi.n_points):
        if i in unremovable:
            continue
        neighbors = adj.get(i, set())
        if len(neighbors) == 2:
            to_remove.append(i)

    if not to_remove:
        return voronoi

    kept = [i for i in range(voronoi.n_points) if i not in to_remove]
    index_map = {old: new for new, old in enumerate(kept)}

    new_edges = []
    for e in voronoi.edges:
        a, b = int(e[0]), int(e[1])
        if a in to_remove or b in to_remove:
            continue
        new_edges.append([index_map[a], index_map[b]])

    new_polys = []
    for poly in voronoi.polys:
        new_poly = [index_map[i] for i in poly if i not in to_remove]
        if len(new_poly) >= 3:
            new_polys.append(new_poly)

    new_polys_edges = []
    for e in voronoi.polys_edges:
        a, b = int(e[0]), int(e[1])
        if a in to_remove or b in to_remove:
            continue
        new_polys_edges.append((index_map[a], index_map[b]))

    new_edges_arr = np.array(new_edges, dtype=int) if new_edges else np.array([]).reshape(0, 2)
    logger.info("Simplified Voronoi: %d points, %d edges, %d polys", len(kept), len(new_edges), len(new_polys))
    return VoronoiGraph(
        points=voronoi.points[kept],
        radii=voronoi.radii[kept],
        edges=new_edges_arr,
        n_points=len(kept),
        n_edges=len(new_edges),
        polys=new_polys,
        polys_edges=new_polys_edges,
    )
