import pyvista as pv
import numpy as np
from typing import Optional


class NormalCalculator:
    @staticmethod
    def compute_point_normals(mesh: pv.PolyData) -> pv.PolyData:
        return mesh.compute_normals(point_normals=True, inplace=False)

    @staticmethod
    def compute_face_normals(mesh: pv.PolyData) -> pv.PolyData:
        return mesh.compute_normals(point_normals=False, inplace=False)

    @staticmethod
    def compute_all_normals(mesh: pv.PolyData) -> pv.PolyData:
        return mesh.compute_normals(point_normals=True, inplace=False)


def fix_face_orientations(mesh: pv.PolyData, outward_point: Optional[np.ndarray] = None) -> pv.PolyData:
    result = mesh.copy()
    normals = result.compute_normals(point_normals=False, inplace=False)
    if "Normals" not in normals.array_names:
        return result
    face_normals = normals["Normals"]
    if outward_point is None:
        outward_point = result.center
    centroids = result.center
    vectors_to_point = outward_point - centroids
    dots = np.sum(face_normals * vectors_to_point, axis=1)
    if np.any(dots < 0):
        result.flip_normals(inplace=True)
    return result