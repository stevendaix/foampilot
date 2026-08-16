from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import logging

import numpy as np

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


@dataclass
class BuildingConfig:
    min_area: float = 10.0
    simplify_tolerance: float = 0.25
    default_height: float = 9.0
    level_height: float = 3.0
    foundation_depth: float = 1.0


@dataclass
class DomainConfig:
    margin_x: float = 100.0
    margin_y: float = 100.0
    top_margin: float = 100.0
    bottom_margin: float = 20.0
    base_cell_size: float = 5.0


@dataclass
class SnappyMeshConfig:
    base_cell_size: float = 5.0
    terrain_refinement_level: int = 2
    building_refinement_level: int = 3
    n_cells_between_walls: int = 4
    max_global_cells: int = 5_000_000
    add_layers: bool = False
