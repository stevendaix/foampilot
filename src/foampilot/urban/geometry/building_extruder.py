from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import logging

import numpy as np
from shapely.geometry import Polygon

from foampilot.urban.model.urban_model import Building
from foampilot.urban.model.terrain import CFDTerrain
from foampilot.urban.snappy_config import BuildingConfig

logger = logging.getLogger(__name__)


@dataclass
class BuildingConfig:
    min_area: float = 10.0
    simplify_tolerance: float = 0.25
    default_height: float = 9.0
    level_height: float = 3.0
    foundation_depth: float = 1.0


class BuildingExtruder:
    """Convert CFD building footprints into closed STL solids for snappyHexMesh."""

    def __init__(self, buildings: List[Building], terrain: CFDTerrain, config: BuildingConfig):
        self.buildings = buildings
        self.terrain = terrain
        self.config = config

    def build_solids(self) -> list:
        """Build closed building solids as PyVista PolyData objects."""
        try:
            import pyvista as pv
        except ImportError as exc:
            raise RuntimeError("pyvista is required for BuildingExtruder") from exc

        solids = []
        for building in self.buildings:
            try:
                solid = self._build_single(building)
                if solid is not None and solid.n_points > 0:
                    solids.append(solid)
            except Exception as exc:
                logger.warning("Failed to build building %s: %s", building.id, exc)
        return solids

    def _build_single(self, building: Building):
        try:
            import pyvista as pv
        except ImportError:
            return None

        footprint = building.footprint
        if footprint is None or footprint.is_empty or footprint.area < self.config.min_area:
            return None

        if not footprint.is_valid:
            try:
                footprint = footprint.buffer(0)
            except Exception:
                return None

        coords = np.array(list(footprint.exterior.coords))
        if len(coords) < 4:
            return None

        coords = coords[:4]

        base_z = float(building.ground_z) - self.config.foundation_depth
        roof_z = float(building.roof_z)

        height = max(roof_z - base_z, self.config.default_height)
        if height <= 0:
            return None

        base_points = np.column_stack((coords[:, 0], coords[:, 1], np.full(len(coords), base_z)))
        roof_points = np.column_stack((coords[:, 0], coords[:, 1], np.full(len(coords), roof_z)))
        all_points = np.vstack((base_points, roof_points))
        n_base = len(coords)

        faces = []
        for i in range(n_base):
            nxt = (i + 1) % n_base
            b0, b1 = i, nxt
            r0, r1 = n_base + i, n_base + nxt
            faces.extend([4, int(b0), int(b1), int(r1), int(r0)])

        faces.extend([n_base, *range(n_base, 2 * n_base)])
        faces.extend([n_base, *range(0, n_base)[::-1]])

        faces = np.array(faces, dtype=np.int32)
        return pv.PolyData(all_points, faces)

    def export_stl(self, output_path: Path) -> Path:
        """Export all building solids into a single STL file."""
        try:
            import pyvista as pv
        except ImportError as exc:
            raise RuntimeError("pyvista is required for BuildingExtruder") from exc

        solids = self.build_solids()
        if not solids:
            raise RuntimeError("No valid building solids to export")

        combined = solids[0]
        for solid in solids[1:]:
            combined = combined + solid

        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.save(str(output_path))
        logger.info("Buildings STL exported to %s (%d solids)", output_path, len(solids))
        return output_path
