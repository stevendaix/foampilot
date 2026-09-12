import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import vtk

logger = logging.getLogger(__name__)


@dataclass
class DelaunayVolume:
    mesh: vtk.vtkUnstructuredGrid
    n_cells: int = 0
    n_points: int = 0
    bbox_diagonal: float = 0.0
    warnings: List[str] = field(default_factory=list)


def build_delaunay(capped_surface: vtk.vtkPolyData, tolerance: float = 1e-3) -> DelaunayVolume:
    d = vtk.vtkDelaunay3D()
    d.CreateDefaultLocator()
    d.SetInputData(capped_surface)
    d.SetTolerance(float(tolerance))
    d.Update()

    result = vtk.vtkUnstructuredGrid()
    result.DeepCopy(d.GetOutput())

    normals_array = capped_surface.GetPointData().GetNormals()
    if normals_array is not None:
        result.GetPointData().AddArray(normals_array)

    n_cells = result.GetNumberOfCells()
    n_points = result.GetNumberOfPoints()

    bbox_min = np.array([result.GetBounds()[i] for i in range(0, 6, 2)])
    bbox_max = np.array([result.GetBounds()[i] for i in range(1, 6, 2)])
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))

    logger.info("Built Delaunay volume: %d cells, %d points", n_cells, n_points)
    return DelaunayVolume(mesh=result, n_cells=n_cells, n_points=n_points, bbox_diagonal=bbox_diag)
