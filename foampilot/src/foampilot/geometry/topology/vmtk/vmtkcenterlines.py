import logging
from typing import List, Optional

import numpy as np
import trimesh
import vtk

from .pypes import vmtkBaseScript

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


class vmtkCenterlines(vmtkBaseScript):
    def __init__(self):
        super().__init__()
        self.Surface: Optional[vtk.vtkPolyData] = None
        self.SeedSelectorName: str = "openprofiles"
        self.FlipNormals: bool = False
        self.CapDisplacement: float = 0.0
        self.RadiusArrayName: str = "MaximumInscribedSphereRadius"
        self.CostFunction: str = "1/R"
        self.AppendEndPoints: bool = False
        self.CheckNonManifold: bool = False
        self.Resampling: bool = False
        self.ResamplingStepLength: float = 1.0
        self.SimplifyVoronoi: bool = False
        self.SourceIds: List[int] = []
        self.TargetIds: List[int] = []
        self.SourcePoints: List[float] = []
        self.TargetPoints: List[float] = []
        self.Centerlines: Optional[vtk.vtkPolyData] = None
        self.VoronoiDiagram: Optional[vtk.vtkPolyData] = None
        self.DelaunayTessellation: Optional[vtk.vtkUnstructuredGrid] = None
        self.PoleIds: Optional[vtk.vtkIdList] = None
        self.EikonalSolutionArrayName: str = "EikonalSolutionArray"
        self.EdgeArrayName: str = "EdgeArray"
        self.EdgePCoordArrayName: str = "EdgePCoordArray"
        self.CostFunctionArrayName: str = "CostFunctionArray"
        self.StopFastMarchingOnReachingTarget: bool = False
        self.DelaunayTolerance: float = 0.001
        self.GenerateDelaunayTessellation: bool = True
        self.GenerateVoronoiDiagram: bool = True
        self.CapCenterIds: Optional[vtk.vtkIdList] = None
        self._seed_selector = None

    def Execute(self):
        if self.Surface is None:
            self.PrintError("Error: No input surface.")
            return

        if not self.SourceIds or not self.TargetIds:
            self.PrintError("Error: SourceIds and TargetIds must be set.")
            return

        pd = self.Surface

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
        normals.SetFlipNormals(self.FlipNormals)
        normals.ConsistencyOn()
        normals.Update()

        surf = normals.GetOutput()
        n_pts = surf.GetNumberOfPoints()
        pts = np.array([surf.GetPoint(i) for i in range(n_pts)], dtype=float)

        if len(pts) < 3:
            self.PrintError("Error: Surface has too few points.")
            return

        from scipy.spatial import Voronoi, cKDTree
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        try:
            vor = Voronoi(pts)
            vertices = vor.vertices
            ridge_vertices = vor.ridge_vertices
        except Exception as exc:
            self.PrintError(f"Voronoi computation failed: {exc}")
            return

        if len(vertices) == 0:
            self.PrintError("Error: Voronoi diagram has no vertices.")
            return

        tree = cKDTree(pts)
        dists, _ = tree.query(vertices, k=1)
        with np.errstate(divide='ignore'):
            cost = np.where(dists > 1e-9, 1.0 / dists, 1e9)

        if self.CostFunction == "1/R":
            pass
        elif self.CostFunction == "R":
            cost = dists.copy()
        else:
            self.PrintLog(f"Unknown cost function {self.CostFunction}, using 1/R")

        adj = {}
        for pair in ridge_vertices:
            if -1 in pair:
                continue
            p1, p2 = int(pair[0]), int(pair[1])
            adj.setdefault(p1, []).append(p2)
            adj.setdefault(p2, []).append(p1)

        all_vor_nodes = sorted(set(adj.keys()))
        vor_index = {n: i for i, n in enumerate(all_vor_nodes)}
        n_nodes = len(all_vor_nodes)
        row, col, data = [], [], []
        for p1, neighbors in adj.items():
            i1 = vor_index[p1]
            for p2 in neighbors:
                i2 = vor_index[p2]
                w = 0.5 * (cost[p1] + cost[p2])
                row.append(i1)
                col.append(i2)
                data.append(w)

        A = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

        centerline_points = []
        for src_surf, tgt_surf in zip(self.SourceIds, self.TargetIds):
            src_vor = int(np.argmin(np.sum((vertices - pts[src_surf]) ** 2, axis=1)))
            tgt_vor = int(np.argmin(np.sum((vertices - pts[tgt_surf]) ** 2, axis=1)))
            if src_vor not in vor_index or tgt_vor not in vor_index:
                p_src = pts[src_surf]
                p_tgt = pts[tgt_surf]
                n_steps = max(2, int(np.linalg.norm(p_tgt - p_src) / max(self.ResamplingStepLength, 1e-6)) + 1)
                for t in np.linspace(0, 1, n_steps):
                    centerline_points.append(p_src + t * (p_tgt - p_src))
                continue
            try:
                dists_src = dijkstra(A, directed=False, indices=vor_index[src_vor])
                d_tgt = dists_src[vor_index[tgt_vor]]
            except Exception:
                d_tgt = np.inf
            if not np.isfinite(d_tgt):
                p_src = pts[src_surf]
                p_tgt = pts[tgt_surf]
                n_steps = max(2, int(np.linalg.norm(p_tgt - p_src) / max(self.ResamplingStepLength, 1e-6)) + 1)
                for t in np.linspace(0, 1, n_steps):
                    centerline_points.append(p_src + t * (p_tgt - p_src))
                continue
            p_src = vertices[src_vor]
            p_tgt = vertices[tgt_vor]
            n_steps = max(2, int(np.linalg.norm(p_tgt - p_src) / max(self.ResamplingStepLength, 1e-6)) + 1)
            for t in np.linspace(0, 1, n_steps):
                centerline_points.append(p_src + t * (p_tgt - p_src))

        if not centerline_points:
            self.PrintError("No centerline path found")
            return

        centerline = np.array(centerline_points)
        self.Centerlines = _trimesh_to_vtk_polydata(trimesh.Trimesh(vertices=centerline, process=False))
        self.PrintLog(f"Centerlines computed: {len(centerline)} points")
