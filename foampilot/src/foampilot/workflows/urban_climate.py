"""Urban climate workflows.

.. deprecated::
    Use ``foampilot.workflows.urban`` instead.
"""
import warnings
warnings.warn(
    "foampilot.workflows.urban_climate is deprecated, "
    "use foampilot.workflows.urban instead",
    DeprecationWarning,
    stacklevel=2
)

from foampilot.workflows.urban import (
    PROFILES,
    UrbanClimateCase,
    UrbanClimateProfile,
    materialize_all,
    RegionSpec,
    UrbanClimateNativeCaseBuilder,
)

__all__ = [
    "PROFILES",
    "UrbanClimateCase",
    "UrbanClimateProfile",
    "materialize_all",
    "RegionSpec",
    "UrbanClimateNativeCaseBuilder",
]