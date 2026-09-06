"""Urban climate workflows for OpenFOAM.

.. deprecated::
    Use ``foampilot.workflows.urban_climate.urbanclimate`` instead.
"""
import warnings
warnings.warn(
    "foampilot.openfoam13.urbanclimate is deprecated, "
    "use foampilot.workflows.urban_climate.urbanclimate instead",
    DeprecationWarning,
    stacklevel=2
)

from foampilot.workflows.urban_climate.urbanclimate import (
    PROFILES,
    UrbanClimateCase,
    UrbanClimateProfile,
    materialize_all,
)

__all__ = [
    "PROFILES",
    "UrbanClimateCase",
    "UrbanClimateProfile",
    "materialize_all",
]