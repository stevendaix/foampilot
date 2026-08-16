from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Union

from foampilot.utilities.manageunits import ValueWithUnit


def _to_vwu(value: Union[ValueWithUnit, float], default_unit: str = "m") -> ValueWithUnit:
    if isinstance(value, ValueWithUnit):
        return value
    return ValueWithUnit(float(value), default_unit)


@dataclass
class RefinementRegion:
    center: Tuple[float, float, float]
    size: Any = field(default_factory=lambda: _to_vwu(2.0))
    radius: Optional[Any] = None

    def __post_init__(self):
        self.size = _to_vwu(self.size)
        if self.radius is not None:
            self.radius = _to_vwu(self.radius)


@dataclass
class WakeRefinement:
    length: float = 10.0      # × H
    width: float = 4.0        # × H
    height: float = 2.0       # × H
    target_size: Any = field(default_factory=lambda: _to_vwu(2.0))
    distance_threshold: Optional[Any] = None

    def __post_init__(self):
        self.target_size = _to_vwu(self.target_size)
        if self.distance_threshold is not None:
            self.distance_threshold = _to_vwu(self.distance_threshold)


@dataclass
class BoundaryLayerConfig:
    first_layer_height: Any = field(default_factory=lambda: _to_vwu(0.05))
    growth_rate: float = 1.2
    num_layers: int = 5
    patches: List[str] = field(default_factory=lambda: ["ground", "buildings"])

    def __post_init__(self):
        self.first_layer_height = _to_vwu(self.first_layer_height)


@dataclass
class MeshConfig:
    global_size: Any = field(default_factory=lambda: _to_vwu(15.0))
    building_size: Any = field(default_factory=lambda: _to_vwu(2.0))
    wake_size: Any = field(default_factory=lambda: _to_vwu(4.0))
    ground_size: Any = field(default_factory=lambda: _to_vwu(2.0))
    top_size: Optional[Any] = None
    side_size: Optional[Any] = None
    min_size: Any = field(default_factory=lambda: _to_vwu(0.1))
    max_size: Any = field(default_factory=lambda: _to_vwu(50.0))
    grading_factor: float = 1.2
    wake_refinement: Optional[WakeRefinement] = None
    refinement_regions: List[RefinementRegion] = field(default_factory=list)
    boundary_layers: Optional[BoundaryLayerConfig] = None
    algorithm_2d: int = 6
    algorithm_3d: int = 1

    def __post_init__(self):
        self.global_size = _to_vwu(self.global_size)
        self.building_size = _to_vwu(self.building_size)
        self.wake_size = _to_vwu(self.wake_size)
        self.ground_size = _to_vwu(self.ground_size)
        self.min_size = _to_vwu(self.min_size)
        self.max_size = _to_vwu(self.max_size)
        if self.top_size is not None:
            self.top_size = _to_vwu(self.top_size)
        if self.side_size is not None:
            self.side_size = _to_vwu(self.side_size)
