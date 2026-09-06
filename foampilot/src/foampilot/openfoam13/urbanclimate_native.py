"""Native urban climate case builder.

.. deprecated::
    Use ``foampilot.workflows.urban_climate.urbanclimate_native`` instead.
"""
import warnings
warnings.warn(
    "foampilot.openfoam13.urbanclimate_native is deprecated, "
    "use foampilot.workflows.urban_climate.urbanclimate_native instead",
    DeprecationWarning,
    stacklevel=2
)

from foampilot.workflows.urban_climate.urbanclimate_native import (
    RegionSpec,
    UrbanClimateNativeCaseBuilder,
)

__all__ = [
    "RegionSpec",
    "UrbanClimateNativeCaseBuilder",
]