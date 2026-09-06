"""Unit normalization for the physiology/OpenFOAM boundary.

The project-wide :mod:`foampilot.utilities.manageunits` module owns the Pint
registry. This adapter keeps the physiology API array-friendly while ensuring
that every boundary value is checked and converted before numerical work.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..utilities.manageunits import Q_, ValueWithUnit


def as_magnitude(value: Any, unit: str, *, name: str = "value") -> np.ndarray:
    """Return ``value`` converted to ``unit`` as a finite float array.

    ``value`` may be a raw scalar/array, a ``(magnitude, unit)`` pair, a Pint
    quantity, or the project ``ValueWithUnit`` wrapper. Raw values are assumed
    to already use ``unit``; this preserves the numerical API while making
    explicit quantities available at integration boundaries.
    """
    source = value
    if isinstance(value, ValueWithUnit):
        source = value.quantity
    elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
        source = Q_(value[0], value[1])
    if hasattr(source, "to") and hasattr(source, "magnitude"):
        try:
            source = source.to(unit).magnitude
        except Exception as error:
            raise ValueError(f"{name} est incompatible avec {unit}") from error
    try:
        result = np.asarray(source, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} doit être numérique et exprimé en {unit}") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contient une valeur non finie")
    return result


def scalar(value: Any, unit: str, *, name: str = "value") -> float:
    """Return one finite scalar converted to ``unit``."""
    result = as_magnitude(value, unit, name=name)
    if result.ndim != 0:
        raise ValueError(f"{name} doit être scalaire")
    return float(result)
