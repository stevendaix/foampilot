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


class vmtkCenterlineSections(vmtkBaseScript):
    def __init__(self):
        super().__init__()
        self.Surface: Optional[vtk.vtkPolyData] = None
        self.Centerlines: Optional[vtk.vtkPolyData] = None
        self.CenterlineSections: Optional[vtk.vtkPolyData] = None
        self.CenterlineSectionAreaArrayName: str = "CenterlineSectionArea"
        self.CenterlineSectionMinSizeArrayName: str = "CenterlineSectionMinSize"
        self.CenterlineSectionMaxSizeArrayName: str = "CenterlineSectionMaxSize"
        self.CenterlineSectionShapeArrayName: str = "CenterlineSectionShape"
        self.CenterlineSectionClosedArrayName: str = "CenterlineSectionClosed"

    def Execute(self):
        if self.Surface is None:
            self.PrintError("Error: No input surface.")
            return
        if self.Centerlines is None:
            self.PrintError("Error: No input centerlines.")
            return

        cl_pts = np.array([self.Centerlines.GetPoint(i) for i in range(self.Centerlines.GetNumberOfPoints())])
        sections = []
        for i in range(len(cl_pts) - 1):
            center = cl_pts[i]
            direction = cl_pts[i + 1] - cl_pts[i]
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            direction = direction / norm
            try:
                sec = self._trimesh_section(self.Surface, center, direction)
            except Exception:
                continue
            if sec is None or len(sec) == 0:
                continue
            sections.append(sec)

        if not sections:
            self.PrintError("No centerline sections computed")
            return

        all_pts = np.vstack(sections)
        self.CenterlineSections = _trimesh_to_vtk_polydata(trimesh.Trimesh(vertices=all_pts, process=False))
        self.PrintLog(f"Centerline sections: {len(sections)} sections, {len(all_pts)} points")

    def _trimesh_section(self, surface: vtk.vtkPolyData, center: np.ndarray, direction: np.ndarray) -> Optional[np.ndarray]:
        mesh = _vtk_polydata_to_trimesh(surface)
        try:
            sec = mesh.section(plane_origin=center, plane_normal=direction)
        except Exception:
            return None
        if sec is None or len(sec.discrete) == 0:
            return None
        return np.asarray(sec.discrete[0])


class vmtkBranchSections(vmtkBaseScript):
    def __init__(self):
        super().__init__()
        self.Surface: Optional[vtk.vtkPolyData] = None
        self.Centerlines: Optional[vtk.vtkPolyData] = None
        self.BranchSections: Optional[vtk.vtkPolyData] = None
        self.NumberOfDistanceSpheres: int = 1
        self.ReverseDirection: bool = False
        self.RadiusArrayName: str = "MaximumInscribedSphereRadius"
        self.GroupIdsArrayName: str = "GroupIds"
        self.CenterlineIdsArrayName: str = "CenterlineIds"
        self.TractIdsArrayName: str = "TractIds"
        self.BlankingArrayName: str = "Blanking"
        self.BranchSectionGroupIdsArrayName: str = "BranchSectionGroupIds"
        self.BranchSectionAreaArrayName: str = "BranchSectionArea"
        self.BranchSectionMinSizeArrayName: str = "BranchSectionMinSize"
        self.BranchSectionMaxSizeArrayName: str = "BranchSectionMaxSize"
        self.BranchSectionShapeArrayName: str = "BranchSectionShape"
        self.BranchSectionClosedArrayName: str = "BranchSectionClosed"
        self.BranchSectionDistanceSpheresArrayName: str = "BranchSectionDistanceSpheres"

    def Execute(self):
        if self.Surface is None:
            self.PrintError("Error: No input surface.")
            return
        if self.Centerlines is None:
            self.PrintError("Error: No input centerlines.")
            return

        sections_script = vmtkCenterlineSections()
        sections_script.Surface = self.Surface
        sections_script.Centerlines = self.Centerlines
        sections_script.Execute()

        if sections_script.CenterlineSections is None:
            return

        self.BranchSections = sections_script.CenterlineSections
        self.PrintLog("Branch sections computed")
