from dataclasses import dataclass, field
from typing import Optional

from foampilot.utilities.manageunits import ValueWithUnit


@dataclass
class CleanupOptions:
    tolerance: ValueWithUnit = ValueWithUnit(0.05, "m")
    simplify_tolerance: Optional[ValueWithUnit] = None
    min_building_area: ValueWithUnit = ValueWithUnit(1.0, "m^2")
    min_building_height: ValueWithUnit = ValueWithUnit(0.5, "m")
    min_gap: ValueWithUnit = ValueWithUnit(0.5, "m")
    merge_overlapping_buildings: bool = True
    make_valid: bool = True
    remove_holes_below_area: ValueWithUnit = ValueWithUnit(0.5, "m^2")


class GeometryCleanup:
    def __init__(self, options: Optional[CleanupOptions] = None):
        self.options = options or CleanupOptions()

    def clean(self, polygon):
        return polygon
