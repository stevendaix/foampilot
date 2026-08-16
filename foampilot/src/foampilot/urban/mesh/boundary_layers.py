from dataclasses import dataclass, field
from typing import Optional, List

from foampilot.utilities.manageunits import ValueWithUnit


@dataclass
class BoundaryLayerConfig:
    first_layer_height: ValueWithUnit = ValueWithUnit(0.05, "m")
    growth_rate: float = 1.2
    num_layers: int = 5
    patches: List[str] = field(default_factory=lambda: ["ground", "buildings"])
