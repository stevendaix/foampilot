from __future__ import annotations

from pathlib import Path
import logging
from typing import Optional, Tuple
import numpy as np
import shapely.ops

from foampilot.urban.model.urban_model import UrbanModel, Building
from foampilot.urban.model.terrain import CFDTerrain
from foampilot.urban.readers.base_reader import BaseReader

logger = logging.getLogger(__name__)


class VoxCityReader(BaseReader):
    """Read building footprints and terrain from VoxCity voxel data.

    VoxCity aggregates multiple open data sources (OSM, Overture,
    Microsoft, EUBUCCO, DEM, canopy) into a voxel city model.
    This reader extracts buildings and terrain from it.
    """

    def __init__(
        self,
        meshsize: float = 5.0,
        building_source: Optional[str] = None,
        dem_source: Optional[str] = None,
        land_cover_source: Optional[str] = None,
        canopy_height_source: Optional[str] = None,
    ):
        self.meshsize = meshsize
        self.building_source = building_source
        self.dem_source = dem_source
        self.land_cover_source = land_cover_source
        self.canopy_height_source = canopy_height_source

    def read(self, source: Path | list) -> Tuple[UrbanModel, CFDTerrain]:
        """Generate UrbanModel and CFDTerrain from VoxCity data.

        Parameters
        ----------
        source : Path or list
            Path to a JSON input defining the AOI rectangle,
            or a list of (lon, lat) vertices.

        Returns
        -------
        Tuple[UrbanModel, CFDTerrain]
        """
        try:
            import ee
            from voxcity.generator import get_voxcity
        except ImportError as exc:
            raise RuntimeError(
                "VoxCity and Google Earth Engine are required for VoxCityReader. "
                "Install them with: pip install voxcity && earthengine authenticate"
            ) from exc

        rectangle_vertices = self._load_rectangle(source)
        kwargs = {}
        if self.building_source is not None:
            kwargs["building_source"] = self.building_source
        if self.dem_source is not None:
            kwargs["dem_source"] = self.dem_source
        if self.land_cover_source is not None:
            kwargs["land_cover_source"] = self.land_cover_source
        if self.canopy_height_source is not None:
            kwargs["canopy_height_source"] = self.canopy_height_source

        logger.info("Generating VoxCity model for %s", rectangle_vertices)
        voxcity = get_voxcity(rectangle_vertices, self.meshsize, **kwargs)

        urban = self._extract_buildings(voxcity)
        terrain = self._extract_terrain(voxcity)
        return urban, terrain

    def _load_rectangle(self, source: Path | list) -> list:
        """Load AOI rectangle vertices from a JSON file, a list, or a Path."""
        if isinstance(source, list):
            return source

        source = Path(source)
        if source.exists() and source.is_file():
            import json
            data = json.loads(source.read_text())
            if "rectangle_vertices" in data:
                return data["rectangle_vertices"]
            if "vertices" in data:
                return data["vertices"]
            raise ValueError(
                f"Input file {source} must contain 'rectangle_vertices' or 'vertices'."
            )
        raise FileNotFoundError(f"Source file not found: {source}")

    def _extract_buildings(self, voxcity) -> UrbanModel:
        """Convert VoxCity building GeoDataFrame to UrbanModel."""
        urban = UrbanModel()
        count = 0

        if "building_gdf" not in voxcity.extras or voxcity.extras["building_gdf"] is None:
            logger.warning("No building_gdf found in VoxCity output.")
            return urban

        gdf = voxcity.extras["building_gdf"]
        if len(gdf) == 0:
            logger.warning("Empty building_gdf in VoxCity output.")
            return urban

        logger.info("Extracting %d buildings from VoxCity", len(gdf))

        try:
            from pyproj import CRS, Transformer
            source_crs = getattr(gdf, "crs", None)
            if source_crs is not None:
                source_crs = CRS.from_user_input(source_crs)
            else:
                minx, miny, maxx, maxy = gdf.total_bounds
                looks_like_lonlat = max(abs(minx), abs(maxx)) <= 180 and max(abs(miny), abs(maxy)) <= 90
                source_crs = CRS.from_epsg(4326) if looks_like_lonlat else None
            target_crs = CRS.from_epsg(32631)
            if source_crs is not None and source_crs != target_crs:
                transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
                def project_geom(geom):
                    if geom is None or geom.is_empty:
                        return geom
                    return shapely.ops.transform(lambda x, y: transformer.transform(x, y), geom)
            else:
                project_geom = lambda geom: geom
            if source_crs is None:
                logger.warning("VoxCity building CRS is unknown; preserving native coordinates")
        except Exception as exc:
            logger.warning("Could not normalize VoxCity CRS: %s", exc)
            project_geom = lambda geom: geom

        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            if geom.geom_type == "Polygon":
                footprints = [geom]
            elif geom.geom_type == "MultiPolygon":
                footprints = list(geom.geoms)
            else:
                continue

            for footprint in footprints:
                projected = project_geom(footprint)
                if projected is None or projected.is_empty:
                    continue
                area_m2 = projected.area
                if area_m2 < 1.0:
                    continue
                footprint = projected

                height = self._extract_height(row)
                ground_z = float(getattr(row, "ground_z", 0.0) or 0.0)

                building_id = (
                    f"vox_{idx}_{count}"
                    if len(footprints) > 1
                    else f"vox_{idx}"
                )

                urban.add_building(Building(
                    id=building_id,
                    footprint=footprint,
                    ground_z=ground_z,
                    roof_z=ground_z + height,
                    source="voxcity",
                    confidence=0.7,
                    attributes={
                        "voxcity_source": getattr(voxcity, "metadata", {}).get("building_source", "unknown"),
                        "original_id": str(idx),
                    },
                ))
                count += 1

        logger.info("Created %d buildings in UrbanModel", count)
        return urban

    def _extract_terrain(self, voxcity) -> CFDTerrain:
        """Convert VoxCity DEM to CFDTerrain."""
        dem = getattr(voxcity, "dem", None)
        if dem is None:
            logger.warning("No DEM found in VoxCity output, using flat terrain.")
            return CFDTerrain.flat(z=0.0)

        if hasattr(dem, "shape") and hasattr(dem, "dims"):
            try:
                import xarray as xr
                if isinstance(dem, xr.DataArray):
                    dem_array = dem.values
                    xs = dem.coords["east"].values if "east" in dem.coords else dem.coords["x"].values
                    ys = dem.coords["north"].values if "north" in dem.coords else dem.coords["y"].values
                    return CFDTerrain.from_grid(
                        dem_array=dem_array,
                        x_min=float(xs.min()),
                        x_max=float(xs.max()),
                        y_min=float(ys.min()),
                        y_max=float(ys.max()),
                    )
            except Exception as exc:
                logger.warning("Could not extract DEM as xarray: %s", exc)

        if isinstance(dem, np.ndarray):
            h, w = dem.shape
            return CFDTerrain.from_grid(
                dem_array=dem,
                x_min=0.0,
                x_max=float(w * self.meshsize),
                y_min=0.0,
                y_max=float(h * self.meshsize),
            )

        logger.warning("Unsupported DEM format, using flat terrain.")
        return CFDTerrain.flat(z=0.0)

    def _extract_height(self, row) -> float:
        """Extract building height from VoxCity row with fallbacks."""
        height = getattr(row, "height", None)
        if height is not None:
            try:
                h = float(height)
                if h == h and h > 0:
                    return h
            except (TypeError, ValueError):
                pass

        levels = getattr(row, "building:levels", None)
        if levels is not None:
            try:
                level_value = float(levels)
                if level_value == level_value and level_value > 0:
                    return level_value * 3.0
            except (TypeError, ValueError):
                pass

        roof_height = getattr(row, "roof_height", None)
        if roof_height is not None:
            try:
                rh = float(roof_height)
                if rh == rh and rh > 0:
                    return rh
            except (TypeError, ValueError):
                pass

        return 9.0
