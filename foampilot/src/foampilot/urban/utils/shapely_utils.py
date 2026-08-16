from typing import Optional
from foampilot.utilities.manageunits import ValueWithUnit


def snap_to_grid(value: float, step: float) -> float:
    return round(value / step) * step


def simplify_polygon(polygon, tolerance: Optional[ValueWithUnit] = None):
    if tolerance is None:
        return polygon
    tol = tolerance.get_in("m")
    return polygon.simplify(tol, preserve_topology=True)
