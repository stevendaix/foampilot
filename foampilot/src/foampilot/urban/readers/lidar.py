from pathlib import Path
from typing import Optional, Dict, Any
import logging

import numpy as np

from foampilot.urban.readers.base_reader import BaseReader
from foampilot.urban.model.urban_model import UrbanModel, Building
from foampilot.urban.model.terrain import CFDTerrain
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


class LiDARReader(BaseReader):
    """Read building footprints and heights from LiDAR data."""

    def __init__(self, resolution: float = 1.0, min_building_area: float = 10.0):
        self.resolution = resolution
        self.min_building_area = min_building_area

    def read(self, source: Path) -> UrbanModel:
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"LiDAR file not found: {source}")

        if source.suffix.lower() == ".las" or source.suffix.lower() == ".laz":
            points = self._read_las(source)
        elif source.suffix.lower() == ".csv":
            points = self._read_csv(source)
        else:
            raise ValueError(f"Unsupported LiDAR format: {source.suffix}")

        urban = self._extract_buildings(points)
        return urban

    def _read_las(self, path: Path) -> np.ndarray:
        try:
            import laspy
        except ImportError:
            raise ImportError("laspy is required for LiDAR reading")

        las = laspy.read(path)
        points = np.column_stack([
            las.x, las.y, las.z, las.classification
        ])
        return points

    def _read_csv(self, path: Path) -> np.ndarray:
        import csv
        points = []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row.get("x", row.get("X", 0)))
                y = float(row.get("y", row.get("Y", 0)))
                z = float(row.get("z", row.get("Z", 0)))
                cls = int(row.get("classification", row.get("class", 2)))
                points.append((x, y, z, cls))
        return np.array(points)

    def _extract_buildings(self, points: np.ndarray) -> UrbanModel:
        try:
            from scipy.spatial import KDTree
        except ImportError:
            raise ImportError("scipy is required for LiDAR building extraction")

        building_points = points[points[:, 3] == 6]
        if len(building_points) == 0:
            building_points = points[points[:, 3] == 2]

        if len(building_points) == 0:
            return UrbanModel()

        xs = building_points[:, 0]
        ys = building_points[:, 1]
        zs = building_points[:, 2]

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        zmin, zmax = zs.min(), zs.max()

        urban = UrbanModel()
        terrain = CFDTerrain(source="lidar")

        nx = int(np.ceil((xmax - xmin) / self.resolution)) + 1
        ny = int(np.ceil((ymax - ymin) / self.resolution)) + 1

        for i in range(nx):
            for j in range(ny):
                x = xmin + i * self.resolution
                y = ymin + j * self.resolution
                terrain.add_point(x, y, zmin)

        urban.add_terrain(terrain)

        grid_x, grid_y = np.meshgrid(
            np.linspace(xmin, xmax, max(2, nx // 10)),
            np.linspace(ymin, ymax, max(2, ny // 10))
        )
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

        tree = KDTree(building_points[:, :2])
        distances, indices = tree.query(grid_points, k=min(50, len(building_points)))

        from sklearn.cluster import DBSCAN
        eps = self.resolution * 2
        min_samples = 10
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(building_points[:, :2])

        labels = clustering.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)

        count = 0
        for label in unique_labels:
            mask = labels == label
            cluster_points = building_points[mask]
            if len(cluster_points) < 10:
                continue

            cx = cluster_points[:, 0].mean()
            cy = cluster_points[:, 1].mean()
            cz_min = cluster_points[:, 2].min()
            cz_max = cluster_points[:, 2].max()
            height = float(cz_max - cz_min)

            if height < 1.0:
                continue

            footprint = Polygon([
                (cluster_points[:, 0].min(), cluster_points[:, 1].min()),
                (cluster_points[:, 0].max(), cluster_points[:, 1].min()),
                (cluster_points[:, 0].max(), cluster_points[:, 1].max()),
                (cluster_points[:, 0].min(), cluster_points[:, 1].max()),
            ])

            if footprint.area < self.min_building_area:
                continue

            urban.add_building(Building(
                id=f"lidar_{count}",
                footprint=footprint,
                ground_z=float(cz_min),
                roof_z=float(cz_max),
                source="lidar",
                confidence=0.9,
                attributes={
                    "n_points": int(len(cluster_points)),
                    "height_std": float(cluster_points[:, 2].std()),
                },
            ))
            count += 1

        logger.info("Extracted %d buildings from LiDAR", count)
        return urban
