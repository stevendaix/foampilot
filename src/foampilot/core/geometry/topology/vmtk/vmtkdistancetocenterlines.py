import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import vtk

from .pypes import vmtkBaseScript

logger = logging.getLogger(__name__)


class vmtkDistanceToCenterlines(vmtkBaseScript):
    def __init__(self):
        super().__init__()
        self.Surface: Optional[vtk.vtkPolyData] = None
        self.Centerlines: Optional[vtk.vtkPolyData] = None
        self.DistanceToCenterlinesArrayName: str = "DistanceToCenterlines"
        self.UseRadius: bool = False
        self.RadiusArrayName: str = "MaximumInscribedSphereRadius"
        self.InsideOut: bool = False
        self.Smoothing: bool = False
        self.SmoothingFactor: float = 0.0

    def Execute(self):
        if self.Surface is None:
            self.PrintError("Error: No input surface.")
            return
        if self.Centerlines is None:
            self.PrintError("Error: No input centerlines.")
            return

        cl_pts = _vtk_to_numpy(self.Centerlines.GetPoints())
        if cl_pts.size == 0:
            self.PrintError("Error: Empty centerlines.")
            return

        tree = _KDTree(cl_pts)
        surf_pts = _vtk_to_numpy(self.Surface.GetPoints())
        dists, _ = tree.query(surf_pts, k=1)
        dists = np.asarray(dists, dtype=float)

        if self.UseRadius:
            try:
                radius_arr = _vtk_get_array(self.Centerlines, self.RadiusArrayName)
                if radius_arr is not None:
                    cl_radius = np.asarray(radius_arr, dtype=float)
                    if cl_radius.size == cl_pts.shape[0]:
                        r_interp = _interpolate_along_centerline(self.Centerlines, cl_radius, surf_pts)
                        if r_interp is not None:
                            dists = np.where(r_interp > 1e-9, dists / r_interp, dists)
            except Exception as exc:
                logger.debug("Radius interpolation failed: %s", exc)

        arr = vtk.vtkFloatArray()
        arr.SetName(self.DistanceToCenterlinesArrayName)
        for d in dists:
            arr.InsertNextTuple1(float(d))
        self.Surface.GetPointData().AddArray(arr)
        self.Surface.GetPointData().SetActiveScalars(self.DistanceToCenterlinesArrayName)
        self.PrintLog(f"DistanceToCenterlines computed: min={dists.min():.3f}, max={dists.max():.3f}")


def _vtk_to_numpy(points: vtk.vtkPoints) -> np.ndarray:
    n = points.GetNumberOfPoints()
    out = np.empty((n, 3), dtype=float)
    for i in range(n):
        p = points.GetPoint(i)
        out[i, 0] = p[0]
        out[i, 1] = p[1]
        out[i, 2] = p[2]
    return out


def _vtk_get_array(polydata: vtk.vtkPolyData, name: str) -> Optional[np.ndarray]:
    arr = polydata.GetPointData().GetArray(name)
    if arr is None:
        return None
    n = arr.GetNumberOfTuples()
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = arr.GetTuple1(i)
    return out


class _KDTree:
    def __init__(self, pts: np.ndarray):
        from scipy.spatial import cKDTree
        self._tree = cKDTree(pts)

    def query(self, pts: np.ndarray, k: int = 1):
        return self._tree.query(pts, k=k)


def _interpolate_along_centerline(
    centerlines: vtk.vtkPolyData, values: np.ndarray, query_pts: np.ndarray
) -> Optional[np.ndarray]:
    try:
        from scipy.interpolate import interp1d

        pts = _vtk_to_numpy(centerlines.GetPoints())
        if pts.shape[0] != values.shape[0]:
            return None
        arc = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        arc = np.insert(arc, 0, 0.0)
        if arc[-1] < 1e-9:
            return None
        f = interp1d(arc, values, kind="linear", fill_value="extrapolate")
        q_arc = np.full(query_pts.shape[0], np.nan, dtype=float)
        for i, q in enumerate(query_pts):
            idx = np.argmin(np.sum((pts - q) ** 2, axis=1))
            q_arc[i] = arc[idx]
        return f(q_arc)
    except Exception:
        return None
