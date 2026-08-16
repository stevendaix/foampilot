from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any

from foampilot.utilities.manageunits import ValueWithUnit


@dataclass
class CFDBuilding:
    id: str
    footprint_local: Any  # shapely.geometry.Polygon
    ground_z_local: float
    roof_z_local: float
    height: float
    source_building_id: str
    attributes: dict = field(default_factory=dict)


@dataclass
class CFDTerrain:
    pass


@dataclass
class CFDGeometry:
    buildings: List[CFDBuilding]
    terrain: Optional[CFDTerrain]
    domain_box: Tuple[float, float, float, float, float, float]
    lod: str
    metadata: dict = field(default_factory=dict)
