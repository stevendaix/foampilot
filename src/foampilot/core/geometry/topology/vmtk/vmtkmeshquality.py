import logging
from typing import Optional

import vtk

from .pypes import vmtkBaseScript

logger = logging.getLogger(__name__)


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
