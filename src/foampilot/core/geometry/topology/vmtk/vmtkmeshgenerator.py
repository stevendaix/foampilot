import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import vtk

from .pypes import vmtkBaseScript

logger = logging.getLogger(__name__)


class vmtkMeshGenerator(vmtkBaseScript):
    def __init__(self):
        super().__init__()
        self.Surface: Optional[vtk.vtkPolyData] = None
        self.Centerlines: Optional[vtk.vtkPolyData] = None
        self.Tetrahedralize: bool = True
        self.BoundaryLayer: bool = False
        self.BoundaryLayerThicknessFactor: float = 0.5
        self.BoundaryLayerSublayers: int = 2
        self.BoundaryLayerSublayerRatio: float = 0.5
        self.ElementSizeMode: str = "edgelength"
        self.TargetEdgeLength: float = 1.0
        self.EdgeLengthArrayName: str = "DistanceToCenterlines"
        self.EdgeLengthFactor: float = 0.3
        self.MaximumEdgeLength: float = 1e9
        self.MinimumEdgeLength: float = 0.0
        self.Mesh: Optional[vtk.vtkUnstructuredGrid] = None

    def Execute(self):
        if self.Surface is None:
            self.PrintError("Error: No input surface.")
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
        normals.ConsistencyOn()
        normals.Update()

        surf = normals.GetOutput()

        if self.ElementSizeMode == "edgelengtharray" and self.Centerlines is not None:
            size_arr = _compute_adaptive_size(surf, self.Centerlines, self.EdgeLengthArrayName)
            if size_arr is not None:
                _apply_size_array(surf, size_arr, self.EdgeLengthFactor, self.MinimumEdgeLength, self.MaximumEdgeLength)

        try:
            import gmsh
        except ImportError:
            self.PrintError("gmsh is required for volume meshing")
            return

        gmsh.initialize()
        gmsh.model.add("vmtk_mesh")
        try:
            self._build_gmsh_mesh(surf)
            types, elems, nodes = gmsh.model.mesh.getElements(dim=3)
            if not elems:
                self.PrintError("Volume meshing failed: no 3D elements")
                return
            node_tags, coords, _ = gmsh.model.mesh.getNodes()
            node_tags = [int(t) for t in node_tags]
            coords = list(coords)
            tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
            points = np.array([(coords[3*i], coords[3*i+1], coords[3*i+2]) for i in range(len(node_tags))], dtype=float)

            cells = []
            for elem_list in elems:
                arr = np.asarray(elem_list, dtype=int).reshape(-1, 4)
                arr = np.vectorize(tag_to_idx.get)(arr)
                cells.extend(arr.tolist())

            self.Mesh = _numpy_to_vtu(points, cells, vtk.VTK_TETRA)
            self.PrintLog(f"Mesh generated: {len(cells)} tetrahedra, {len(points)} nodes")
        finally:
            gmsh.finalize()

    def _build_gmsh_mesh(self, surf: vtk.vtkPolyData):
        pts = _vtk_to_numpy(surf.GetPoints())
        polys = surf.GetPolys()
        polys.InitTraversal()
        faces = []
        pt_ids = vtk.vtkIdList()
        while polys.GetNextCell(pt_ids):
            faces.append([pt_ids.GetId(0), pt_ids.GetId(1), pt_ids.GetId(2)])

        node_map = {}
        next_tag = 1
        for face in faces:
            tags = []
            for pid in face:
                if pid not in node_map:
                    p = pts[pid]
                    node_map[pid] = gmsh.model.occ.addPoint(float(p[0]), float(p[1]), float(p[2]))
                    next_tag += 1
                tags.append(node_map[pid])
            gmsh.model.occ.addPolygon(tags)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSizeFactor(0.5)
        gmsh.model.mesh.generate(3)


def _vtk_to_numpy(points: vtk.vtkPoints) -> np.ndarray:
    n = points.GetNumberOfPoints()
    out = np.empty((n, 3), dtype=float)
    for i in range(n):
        p = points.GetPoint(i)
        out[i] = [p[0], p[1], p[2]]
    return out


def _compute_adaptive_size(
    surf: vtk.vtkPolyData, centerlines: vtk.vtkPolyData, array_name: str
) -> Optional[np.ndarray]:
    try:
        cl_pts = _vtk_to_numpy(centerlines.GetPoints())
        surf_pts = _vtk_to_numpy(surf.GetPoints())
        if cl_pts.size == 0 or surf_pts.size == 0:
            return None
        from scipy.spatial import cKDTree
        dists, _ = cKDTree(cl_pts).query(surf_pts, k=1)
        return np.asarray(dists, dtype=float)
    except Exception:
        return None


def _apply_size_array(
    surf: vtk.vtkPolyData,
    size_values: np.ndarray,
    factor: float,
    min_size: float,
    max_size: float,
):
    arr = vtk.vtkFloatArray()
    arr.SetNumberOfComponents(1)
    arr.SetName("MeshSize")
    for v in size_values:
        s = float(np.clip(v * factor, min_size, max_size))
        arr.InsertNextTuple1(s)
    surf.GetPointData().AddArray(arr)
    surf.GetPointData().SetActiveScalars("MeshSize")


def _numpy_to_vtu(points: np.ndarray, cells: List[List[int]], cell_type: int) -> vtk.vtkUnstructuredGrid:
    grid = vtk.vtkUnstructuredGrid()
    pts = vtk.vtkPoints()
    for p in points:
        pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    grid.SetPoints(pts)
    for cell in cells:
        id_list = vtk.vtkIdList()
        for c in cell:
            id_list.InsertNextId(int(c))
        grid.InsertNextCell(cell_type, id_list)
    return grid
