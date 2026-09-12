from pathlib import Path
from typing import Optional, Dict, Any
import logging

from foampilot.urban.readers.base_reader import BaseReader
from foampilot.urban.model.urban_model import UrbanModel, Building
from shapely.geometry import Polygon

import logging

logger = logging.getLogger(__name__)


class OSMReader(BaseReader):
    """Read building footprints from OpenStreetMap via osmnx."""

    def __init__(self, distance: int = 500, tags: dict = None, center: tuple = None):
        self.distance = distance
        self.tags = tags or {"building": True}
        self.center = center

    def read(self, source: str) -> UrbanModel:
        import osmnx as ox

        if self.center is not None:
            logger.info("Downloading OSM data around point %s (distance=%sm)", self.center, self.distance)
            gdf = ox.features_from_point(self.center, tags=self.tags, dist=self.distance)
        else:
            logger.info("Geocoding and downloading OSM data for: %s", source)
            point = ox.geocode(source)
            logger.info("Downloading OSM data around %s (distance=%sm)", point, self.distance)
            gdf = ox.features_from_point(point, tags=self.tags, dist=self.distance)

        # Project to a local projected CRS (meters)
        try:
            from osmnx.projection import project_gdf
            gdf = project_gdf(gdf)
        except Exception as exc:
            logger.warning("Could not project OSM data: %s", exc)

        logger.info("Found %d OSM features", len(gdf))

        urban = UrbanModel()
        count = 0

        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            if geom.geom_type == "Polygon":
                coords = list(geom.exterior.coords)
            elif geom.geom_type == "MultiPolygon":
                largest = max(geom.geoms, key=lambda g: g.area)
                coords = list(largest.exterior.coords)
            else:
                continue

            if len(coords) < 4:
                continue

            building_id = str(idx)
            if isinstance(idx, tuple):
                building_id = f"{idx[0]}/{idx[1]}"

            height = self._extract_height(row)
            levels = row.get("building:levels")
            if levels is not None:
                try:
                    level_value = float(levels)
                    if level_value == level_value:
                        height = max(height, level_value * 3.0)
                except (ValueError, TypeError):
                    pass

            if height <= 1.0:
                height = 10.0

            footprint = Polygon(coords)
            if not footprint.is_valid:
                footprint = footprint.buffer(0)
                if not footprint.is_valid:
                    continue

            urban.add_building(Building(
                id=building_id,
                footprint=footprint,
                ground_z=0.0,
                roof_z=height,
                source="osm",
                confidence=0.8,
                attributes={
                    "osm_tags": {k: str(v) for k, v in row.items() if not k.startswith("geometry")},
                },
            ))
            count += 1

        logger.info("Converted %d valid buildings", count)
        return urban

    def _extract_height(self, row) -> float:
        for field in ["height", "building:height"]:
            if field in row and row[field] is not None:
                try:
                    value = float(row[field])
                    if value == value:
                        return value
                except (ValueError, TypeError):
                    continue
        return 10.0
