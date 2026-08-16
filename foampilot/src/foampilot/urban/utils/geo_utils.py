from typing import Tuple, Optional
from foampilot.utilities.manageunits import ValueWithUnit


def ensure_metric(value: ValueWithUnit, target_unit: str = "m") -> float:
    """Convert any ValueWithUnit to the target unit, return magnitude."""
    return value.get_in(target_unit)


def bbox_center(bbox: Tuple[float, float, float, float, float, float]) -> Tuple[float, float, float]:
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)


def bbox_contains(
    bbox: Tuple[float, float, float, float, float, float],
    point: Tuple[float, float, float],
    tol: float = 0.0,
) -> bool:
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    x, y, z = point
    return (
        (xmin - tol) <= x <= (xmax + tol)
        and (ymin - tol) <= y <= (ymax + tol)
        and (zmin - tol) <= z <= (zmax + tol)
    )
