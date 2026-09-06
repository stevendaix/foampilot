from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


class BoundaryRole(Enum):
    INLET = "inlet"
    OUTLET = "outlet"
    WALL = "wall"
    UNKNOWN = "unknown"
    SYMMETRY = "symmetry"
    INTERFACE = "interface"


@dataclass
class OpenProfile:
    id: int
    vertex_ids: set = field(default_factory=set)
    edge_ids: set = field(default_factory=set)
    adjacent_face_ids: set = field(default_factory=set)
    centroid: Optional[np.ndarray] = None
    normal: Optional[np.ndarray] = None
    area: float = 0.0
    perimeter: float = 0.0
    equivalent_radius: float = 0.0
    planarity: float = 0.0
    circularity: float = 0.0
    confidence: float = 0.0
    role: BoundaryRole = BoundaryRole.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.centroid is None:
            self.centroid = np.array([0.0, 0.0, 0.0])
        if self.normal is None:
            self.normal = np.array([0.0, 0.0, 1.0])
        if self.equivalent_radius == 0.0 and self.area > 0.0:
            self.equivalent_radius = float(np.sqrt(self.area / np.pi))
