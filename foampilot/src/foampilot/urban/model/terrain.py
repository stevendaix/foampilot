from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import numpy as np

from foampilot.utilities.manageunits import ValueWithUnit


@dataclass
class TerrainPoint:
    x: float
    y: float
    z: float


@dataclass
class CFDTerrain:
    """Terrain elevation data for CFD domain."""

    points: List[TerrainPoint] = field(default_factory=list)
    grid_resolution: Optional[ValueWithUnit] = None
    source: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_point(self, x: float, y: float, z: float) -> None:
        self.points.append(TerrainPoint(x, y, z))

    def get_elevation(self, x: float, y: float) -> float:
        if not self.points:
            return 0.0

        xs = np.array([p.x for p in self.points])
        ys = np.array([p.y for p in self.points])
        zs = np.array([p.z for p in self.points])

        if len(self.points) == 1:
            return float(zs[0])

        if len(self.points) == 2:
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
            dist = np.hypot(dx, dy)
            if dist < 1e-12:
                return float(np.mean(zs))
            t = ((x - xs[0]) * dx + (y - ys[0]) * dy) / (dist * dist)
            t = max(0.0, min(1.0, t))
            return float(zs[0] + t * (zs[1] - zs[0]))

        if len(self.points) == 3:
            try:
                from scipy.interpolate import LinearNDInterpolator
                points = np.column_stack((xs, ys))
                interp = LinearNDInterpolator(points, zs, fill_value=float(np.mean(zs)))
                result = interp([[x, y]])
                return float(result[0, 0])
            except Exception:
                idx = np.argmin((xs - x)**2 + (ys - y)**2)
                return float(zs[idx])

        try:
            from scipy.interpolate import griddata
            points = np.column_stack((xs, ys))
            result = griddata(points, zs, (x, y), method="linear")
            if result is None or np.isnan(result):
                idx = np.argmin((xs - x)**2 + (ys - y)**2)
                return float(zs[idx])
            return float(result)
        except ImportError:
            idx = np.argmin((xs - x)**2 + (ys - y)**2)
            return float(zs[idx])

    def get_bounds(self) -> Tuple[float, float, float, float, float, float]:
        if not self.points:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        zs = [p.z for p in self.points]

        return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

    @classmethod
    def from_geotiff(cls, path: Path, band: int = 1) -> "CFDTerrain":
        terrain = cls(source="geotiff")
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio is required for GeoTIFF terrain import")

        with rasterio.open(path) as src:
            band_data = src.read(band)
            transform = src.transform

            rows, cols = band_data.shape
            for row in range(0, rows, max(1, rows // 100)):
                for col in range(0, cols, max(1, cols // 100)):
                    x, y = rasterio.transform.xy(transform, row, col)
                    z = float(band_data[row, col])
                    if np.isfinite(z):
                        terrain.add_point(x, y, z)

        terrain.metadata["source_path"] = str(path)
        terrain.metadata["band"] = band
        return terrain

    @classmethod
    def from_grid(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> "CFDTerrain":
        terrain = cls(source="grid")
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                terrain.add_point(float(x[i, j]), float(y[i, j]), float(z[i, j]))
        return terrain

    @classmethod
    def flat(cls, z: float = 0.0, extent: Tuple[float, float, float, float] = None) -> "CFDTerrain":
        terrain = cls(source="flat")
        if extent is not None:
            xmin, ymin, xmax, ymax = extent
            terrain.add_point(xmin, ymin, z)
            terrain.add_point(xmax, ymax, z)
        else:
            terrain.add_point(0.0, 0.0, z)
        terrain.metadata["elevation"] = z
        return terrain

    @classmethod
    def slope(cls, slope_x: float = 0.0, slope_y: float = 0.0,
              origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
              extent: Tuple[float, float, float, float] = None) -> "CFDTerrain":
        terrain = cls(source="slope")
        ox, oy, oz = origin

        if extent is not None:
            xmin, ymin, xmax, ymax = extent
            corners = [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)]
        else:
            corners = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)]

        for x, y in corners:
            z = oz + slope_x * (x - ox) + slope_y * (y - oy)
            terrain.add_point(x, y, z)

        terrain.metadata["slope_x"] = slope_x
        terrain.metadata["slope_y"] = slope_y
        terrain.metadata["origin"] = origin
        return terrain


@dataclass
class Road:
    pass
