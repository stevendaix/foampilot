import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh
import vtk

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


def _compute_voronoi(pts: np.ndarray):
    from scipy.spatial import Voronoi
    vor = Voronoi(pts)
    return vor.vertices, vor.ridge_points


class VmtkCenterlines:
    def __init__(self, resampling_step_length: float = 1.0, simplify_voronoi: bool = False):
        self.resampling_step_length = resampling_step_length
        self.simplify_voronoi = simplify_voronoi

    def execute(self, surface: trimesh.Trimesh, source_ids: List[int], target_ids: List[int]) -> trimesh.Trimesh:
        logger.info("Computing centerlines (VMTK-style)")
        pd = _trimesh_to_vtk_polydata(surface)

        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(pd)
        clean.Update()

        tri = vtk.vtkTriangleFilter()
        tri.SetInputConnection(clean.GetOutputPort())
        tri.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(tri.GetOutputPort())
        normals.SplittingOff()
        normals.AutoOrientNormalsOn()
        normals.ConsistencyOn()
        normals.Update()

        surf = normals.GetOutput()
        n_pts = surf.GetNumberOfPoints()
        pts = np.array([surf.GetPoint(i) for i in range(n_pts)], dtype=float)

        from scipy.spatial import Delaunay
        delaunay = Delaunay(pts)
        vertices, ridge_points = _compute_voronoi(pts)
        if vertices is None or ridge_points is None or len(vertices) == 0:
            raise RuntimeError("Voronoi diagram failed")

        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        dists, _ = tree.query(vertices, k=1)
        with np.errstate(divide='ignore'):
            cost = np.where(dists > 1e-9, 1.0 / dists, 1e9)

        adj = {}
        for p1, p2 in ridge_points:
            p1, p2 = int(p1), int(p2)
            adj.setdefault(p1, []).append(p2)
            adj.setdefault(p2, []).append(p1)

        all_nodes = sorted(set(adj.keys()) | set(source_ids) | set(target_ids))
        node_index = {n: i for i, n in enumerate(all_nodes)}
        n_nodes = len(all_nodes)
        row, col, data = [], [], []
        for p1, neighbors in adj.items():
            if p1 not in node_index:
                continue
            for p2 in neighbors:
                if p2 not in node_index:
                    continue
                w = 0.5 * (cost[node_index[p1]] + cost[node_index[p2]])
                row.append(node_index[p1])
                col.append(node_index[p2])
                data.append(w)

        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        A = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

        centerline_points = []
        for src_surf, tgt_surf in zip(source_ids, target_ids):
            src_vor = int(np.argmin(np.sum((vertices - pts[src_surf]) ** 2, axis=1)))
            tgt_vor = int(np.argmin(np.sum((vertices - pts[tgt_surf]) ** 2, axis=1)))
            if src_vor not in node_index or tgt_vor not in node_index:
                continue
            dists_src = dijkstra(A, directed=False, indices=node_index[src_vor])
            d_tgt = dists_src[node_index[tgt_vor]]
            if not np.isfinite(d_tgt):
                continue
            p_src = vertices[src_vor]
            p_tgt = vertices[tgt_vor]
            n_steps = max(2, int(np.linalg.norm(p_tgt - p_src) / max(self.resampling_step_length, 1e-6)) + 1)
            for t in np.linspace(0, 1, n_steps):
                centerline_points.append(p_src + t * (p_tgt - p_src))

        if not centerline_points:
            raise RuntimeError("No centerline path found")
        centerline = np.array(centerline_points)
        logger.info("Centerline computed: %d points", len(centerline))
        return trimesh.Trimesh(vertices=centerline, process=False)


class VmtkCenterlineSections:
    def execute(self, surface: trimesh.Trimesh, centerlines: trimesh.Trimesh) -> trimesh.Trimesh:
        cl_pts = centerlines.vertices
        sections = []
        for i in range(len(cl_pts) - 1):
            center = cl_pts[i]
            direction = cl_pts[i + 1] - cl_pts[i]
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            direction = direction / norm
            try:
                sec = surface.section(plane_origin=center, plane_normal=direction)
            except Exception:
                continue
            if sec is None or len(sec.discrete) == 0:
                continue
            sections.append(sec.discrete[0])
        if not sections:
            raise RuntimeError("No centerline sections computed")
        all_pts = np.vstack(sections)
        logger.info("Centerline sections: %d sections, %d points", len(sections), len(all_pts))
        return trimesh.Trimesh(vertices=all_pts, process=False)


class VmtkBranchSections:
    def __init__(self, number_of_distance_spheres: int = 1):
        self.number_of_distance_spheres = number_of_distance_spheres

    def execute(self, surface: trimesh.Trimesh, centerlines: trimesh.Trimesh) -> trimesh.Trimesh:
        return VmtkCenterlineSections().execute(surface, centerlines)
