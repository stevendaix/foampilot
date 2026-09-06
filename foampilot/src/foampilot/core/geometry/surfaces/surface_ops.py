import pyvista as pv
import numpy as np
from typing import List, Optional, Tuple


class SurfaceCleaner:
    def __init__(self, mesh: pv.PolyData):
        self.mesh = mesh.copy()

    def remove_duplicates(self, tolerance: float = 1e-6) -> pv.PolyData:
        return self.mesh.clean(tolerance=tolerance)

    def remove_degenerate(self) -> pv.PolyData:
        return self.mesh.remove_degenerate_cells()

    def remove_unused_points(self) -> pv.PolyData:
        return self.mesh.remove_unused_points()

    def fill_holes(self, hole_size: Optional[float] = None) -> pv.PolyData:
        if hole_size is not None:
            return self.mesh.fill_holes(hole_size=hole_size)
        return self.mesh.fill_holes()

    def compute_hausdorff_distance(self, other: pv.PolyData) -> dict:
        tree = __import__('scipy.spatial').spatial.cKDTree(self.mesh.points)
        distances, _ = tree.query(other.points)
        return {
            "max": np.max(distances),
            "mean": np.mean(distances),
            "local_dist": distances
        }

    def is_manifold(self) -> bool:
        non_manifold = self.mesh.extract_feature_edges(
            boundary_edges=False,
            non_manifold_edges=True,
            feature_edges=False
        )
        return non_manifold.n_points == 0

    def is_watertight(self) -> bool:
        boundaries = self.mesh.extract_feature_edges(boundary_edges=True)
        return boundaries.n_points == 0


class SurfaceMerger:
    @staticmethod
    def merge(mesh_list: List[pv.PolyData], merge_threshold: float = 1e-6) -> pv.PolyData:
        if not mesh_list:
            raise ValueError("No meshes provided")
        result = mesh_list[0]
        for mesh in mesh_list[1:]:
            result = result.merge(mesh)
        return result.clean(tolerance=merge_threshold)

    @staticmethod
    def merge_with_tags(mesh_list: List[pv.PolyData], tag_array: List[int]) -> pv.PolyData:
        combined = SurfaceMerger.merge(mesh_list)
        n_points = combined.n_points
        tags = np.zeros(n_points, dtype=int)
        offset = 0
        for mesh, tag in zip(mesh_list, tag_array):
            n = mesh.n_points
            tags[offset:offset + n] = tag
            offset += n
        combined["region_id"] = tags
        return combined


class SurfaceSplitter:
    @staticmethod
    def by_connectivity(mesh: pv.PolyData) -> List[pv.PolyData]:
        connected = mesh.connectivity()
        regions = np.unique(connected["RegionId"])
        return [connected.extract_points(connected["RegionId"] == r) for r in regions]

    @staticmethod
    def largest_region(mesh: pv.PolyData) -> pv.PolyData:
        connected = mesh.connectivity()
        regions, counts = np.unique(connected["RegionId"], return_counts=True)
        largest_idx = regions[np.argmax(counts)]
        return connected.extract_points(connected["RegionId"] == largest_idx)