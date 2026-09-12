import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import vtk

from .pypes import vmtkBaseScript

logger = logging.getLogger(__name__)


class vmtkSurfaceRemesher(vmtkBaseScript):
    def __init__(self):
        super().__init__()
        self.Surface: Optional[vtk.vtkPolyData] = None
        self.TargetEdgeLength: float = 1.0
        self.MaximumEdgeLength: float = 1e9
        self.MinimumEdgeLength: float = 0.0
        self.TriangleSplitFactor: float = 2.0
        self.CellEntityIdsArrayName: str = "CellEntityIds"
        self.Surface: Optional[vtk.vtkPolyData] = None

    def Execute(self):
        if self.Surface is None:
            self.PrintError("Error: No input surface.")
            return

        mesh = _vtk_to_trimesh(self.Surface)
        if mesh.is_empty:
            self.PrintError("Error: Empty surface mesh")
            return

        edge_length = float(np.clip(self.TargetEdgeLength, self.MinimumEdgeLength, self.MaximumEdgeLength))
        try:
            remeshed = mesh.subdivide_to_size(max_edge=edge_length)
        except Exception as exc:
            self.PrintError(f"Surface remeshing failed: {exc}")
            return

        self.Surface = _trimesh_to_vtk(remeshed)
        self.PrintLog(f"Surface remeshed: {len(remeshed.faces)} triangles, {len(remeshed.vertices)} vertices")


class vmtkMeshQuality(vmtkBaseScript):
    def __init__(self):
        super().__init__()
        self.Mesh: Optional[vtk.vtkUnstructuredGrid] = None
        self.QualityMeasureName: str = "Quality"
        self.SaveCellQuality: bool = True
        self.TargetQuality: float = 0.1

    def Execute(self):
        if self.Mesh is None:
            self.PrintError("Error: No input mesh.")
            return

        quality = vtk.vtkMeshQuality()
        quality.SetInputData(self.Mesh)
        quality.SetTriangleQualityMeasure(vtk.vtkMeshQuality.TRIANGLE_EDGE_RATIO)
        quality.SetTetQualityMeasure(vtk.vtkMeshQuality.TET_RADIUS_RATIO)
        quality.SaveCellQualityOff()
        quality.Update()

        q = quality.GetOutput()
        if q is None:
            self.PrintError("Quality computation failed")
            return

        min_q = 1e9
        max_q = -1e9
        count = 0
        for i in range(q.GetNumberOfCells()):
            arr = q.GetCellData().GetArray("Quality")
            if arr is not None and i < arr.GetNumberOfTuples():
                v = arr.GetTuple1(i)
                min_q = min(min_q, v)
                max_q = max(max_q, v)
                count += 1

        self.PrintLog(f"Mesh quality computed: {count} cells, min={min_q:.3f}, max={max_q:.3f}")


def _vtk_to_trimesh(pd: vtk.vtkPolyData) -> trimesh.Trimesh:
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


def _trimesh_to_vtk(mesh: trimesh.Trimesh) -> vtk.vtkPolyData:
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
