from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

import numpy as np

from foampilot.urban.model.terrain import CFDTerrain

logger = logging.getLogger(__name__)


@dataclass
class TerrainConfig:
    dem_resolution: float = 5.0
    horizontal_extension: float = 50.0
    bottom_offset: float = 20.0
    smoothing_iterations: int = 1
    simplify_tolerance: Optional[float] = 0.5
    fill_nodata: bool = True
    nodata_threshold: float = -9999.0


class TerrainProcessor:
    """Convert CFDTerrain into a closed STL surface for snappyHexMesh."""

    def __init__(self, terrain: CFDTerrain, config: TerrainConfig):
        self.terrain = terrain
        self.config = config

    def build_closed_surface(self):
        """Build a closed terrain surface as a PyVista PolyData."""
        try:
            import pyvista as pv
        except ImportError as exc:
            raise RuntimeError("pyvista is required for TerrainProcessor") from exc

        bounds = self.terrain.get_bounds()
        if not bounds:
            xmin, ymin, zmin, xmax, ymax, zmax = -10.0, -10.0, 0.0, 10.0, 10.0, 0.0
        else:
            xmin, ymin, zmin, xmax, ymax, zmax = bounds

        xmin -= self.config.horizontal_extension
        xmax += self.config.horizontal_extension
        ymin -= self.config.horizontal_extension
        ymax += self.config.horizontal_extension

        res = self.config.dem_resolution
        nx = max(4, int((xmax - xmin) / res) + 1)
        ny = max(4, int((ymax - ymin) / res) + 1)
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)

        z_grid = np.zeros((ny, nx))
        for i in range(ny):
            for j in range(nx):
                z = self.terrain.get_elevation(float(xs[j]), float(ys[i]))
                if z is None or np.isnan(z) or (self.config.fill_nodata and z < self.config.nodata_threshold):
                    z = 0.0
                z_grid[i, j] = z

        z_min_surface = float(np.min(z_grid))
        z_bottom = z_min_surface - self.config.bottom_offset

        xx, yy = np.meshgrid(xs, ys)
        top_z = z_grid

        grid = pv.StructuredGrid(xx, yy, top_z)
        top_surface = grid.extract_surface()

        side_surfaces = []
        side_surfaces.append(self._build_vertical_face(xx, yy, top_z, "south", ymin, z_bottom))
        side_surfaces.append(self._build_vertical_face(xx, yy, top_z, "north", ymax, z_bottom))
        side_surfaces.append(self._build_vertical_face(xx, yy, top_z, "west", xmin, z_bottom))
        side_surfaces.append(self._build_vertical_face(xx, yy, top_z, "east", xmax, z_bottom))

        bottom_surface = self._build_bottom(xx, yy, z_bottom)

        all_surfaces = [top_surface] + side_surfaces + [bottom_surface]
        valid_surfaces = [s for s in all_surfaces if s is not None and s.n_points > 0]

        if not valid_surfaces:
            raise RuntimeError("No valid terrain surfaces generated")

        combined = valid_surfaces[0]
        for surf in valid_surfaces[1:]:
            combined = combined + surf

        return combined

    def _build_vertical_face(self, xx, yy, top_z, side, fixed_coord, z_bottom):
        try:
            import pyvista as pv
        except ImportError:
            return None

        if side == "south":
            x_edge = xx[0, :]
            z_edge = top_z[0, :]
            y_edge = np.full_like(x_edge, fixed_coord)
        elif side == "north":
            x_edge = xx[-1, :]
            z_edge = top_z[-1, :]
            y_edge = np.full_like(x_edge, fixed_coord)
        elif side == "west":
            y_edge = yy[:, 0]
            z_edge = top_z[:, 0]
            x_edge = np.full_like(y_edge, fixed_coord)
        elif side == "east":
            y_edge = yy[:, -1]
            z_edge = top_z[:, -1]
            x_edge = np.full_like(y_edge, fixed_coord)
        else:
            return None

        n = len(x_edge)
        if n < 2:
            return None

        bottom = np.full_like(z_edge, z_bottom)
        x_edge = np.asarray(x_edge, dtype=float)
        y_edge = np.asarray(y_edge, dtype=float)
        z_edge = np.asarray(z_edge, dtype=float)
        bottom = np.asarray(bottom, dtype=float)

        points_top = np.column_stack((x_edge, y_edge, z_edge))
        points_bottom = np.column_stack((x_edge, y_edge, bottom))

        all_points = np.vstack((points_top, points_bottom))
        n_top = n

        faces = []
        for i in range(n - 1):
            p0 = i
            p1 = i + 1
            p2 = n_top + i + 1
            p3 = n_top + i
            faces.extend([4, int(p0), int(p1), int(p2), int(p3)])

        faces = np.array(faces, dtype=np.int32)
        return pv.PolyData(all_points, faces)

    def _build_bottom(self, xx, yy, z_bottom):
        try:
            import pyvista as pv
        except ImportError:
            return None

        nx = xx.shape[1]
        ny = yy.shape[0]
        z = np.full_like(xx, z_bottom)
        grid = pv.StructuredGrid(xx, yy, z)
        return grid.extract_surface()

    def export_stl(self, output_path: Path) -> Path:
        """Export closed terrain surface to STL."""
        surface = self.build_closed_surface()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        surface.save(str(output_path))
        logger.info("Terrain STL exported to %s", output_path)
        return output_path
