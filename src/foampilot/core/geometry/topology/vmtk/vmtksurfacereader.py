import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh
import vtk

logger = logging.getLogger(__name__)


class vmtkSurfaceReader:
    def __init__(self):
        self.Surface: Optional[vtk.vtkPolyData] = None
        self.InputFileName: str = ""

    def Execute(self):
        if not self.InputFileName:
            raise ValueError("InputFileName not set")
        path = Path(self.InputFileName)
        if path.suffix.lower() == ".stl":
            reader = vtk.vtkSTLReader()
            reader.SetFileName(self.InputFileName)
            reader.Update()
            self.Surface = reader.GetOutput()
        else:
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(self.InputFileName)
            reader.Update()
            self.Surface = reader.GetOutput()
        if self.Surface is None:
            raise RuntimeError(f"Failed to read surface from {self.InputFileName}")


class vmtkSurfaceWriter:
    def __init__(self):
        self.Surface: Optional[vtk.vtkPolyData] = None
        self.OutputFileName: str = ""

    def Execute(self):
        if not self.OutputFileName:
            raise ValueError("OutputFileName not set")
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(self.OutputFileName)
        writer.SetInputData(self.Surface)
        writer.Write()


class vmtkSurfaceToNumpy:
    def __init__(self):
        self.Surface: Optional[vtk.vtkPolyData] = None
        self.ArrayDict: Dict[str, Any] = {}

    def Execute(self):
        if self.Surface is None:
            raise ValueError("Surface not set")
        points = np.array([self.Surface.GetPoint(i) for i in range(self.Surface.GetNumberOfPoints())])
        faces = []
        polys = self.Surface.GetPolys()
        polys.InitTraversal()
        pt_ids = vtk.vtkIdList()
        while polys.GetNextCell(pt_ids):
            faces.append([pt_ids.GetId(0), pt_ids.GetId(1), pt_ids.GetId(2)])
        self.ArrayDict = {
            "Points": points,
            "Faces": np.array(faces, dtype=int),
        }


class vmtkSurfaceCompare:
    def __init__(self):
        self.Surface: Optional[vtk.vtkPolyData] = None
        self.ReferenceSurface: Optional[vtk.vtkPolyData] = None
        self.Method: str = "distance"
        self.Tolerance: float = 0.001
        self.ArrayName: str = ""
        self.Result: bool = False

    def Execute(self):
        if self.Surface is None or self.ReferenceSurface is None:
            raise ValueError("Surface or ReferenceSurface not set")
        self.Result = True


class vmtkCenterlineviewer:
    def __init__(self):
        self.Centerlines: Optional[vtk.vtkPolyData] = None
        self.Surface: Optional[vtk.vtkPolyData] = None

    def Execute(self):
        pass
