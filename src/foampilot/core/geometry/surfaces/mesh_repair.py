import pyvista as pv
import numpy as np
from typing import Optional


def remove_duplicates(mesh: pv.PolyData, tolerance: float = 1e-6) -> pv.PolyData:
    return mesh.clean(tolerance=tolerance)


def fix_normals(mesh: pv.PolyData, flip: bool = False) -> pv.PolyData:
    result = mesh.compute_normals(point_normals=False, inplace=False)
    if flip:
        result.flip_normals(inplace=True)
    return result


def compute_normals(mesh: pv.PolyData) -> pv.PolyData:
    return mesh.compute_normals(point_normals=True, inplace=False)


class MeshRepairer:
    def __init__(self, mesh: pv.PolyData):
        self.mesh = mesh.copy()

    def repair(self, remove_duplicates_tol: float = 1e-6) -> pv.PolyData:
        self.mesh = self.mesh.clean(tolerance=remove_duplicates_tol)
        self.mesh = self.mesh.remove_degenerate_cells()
        self.mesh = self.mesh.remove_unused_points()
        return self.mesh

    def close_holes(self, max_hole_size: Optional[float] = None) -> pv.PolyData:
        if max_hole_size is not None:
            return self.mesh.fill_holes(hole_size=max_hole_size)
        return self.mesh.fill_holes()

    def orient_normals_outward(self, center: Optional[np.ndarray] = None) -> pv.PolyData:
        self.mesh.compute_normals(point_normals=False, inplace=True)
        if center is None:
            center = self.mesh.center
        normals = self.mesh["Normals"] if "Normals" in self.mesh.array_names else None
        if normals is not None:
            vectors_to_center = center - self.mesh.points
            dots = np.sum(normals * vectors_to_center, axis=1)
            if np.any(dots < 0):
                self.mesh.flip_normals(inplace=True)
        return self.mesh