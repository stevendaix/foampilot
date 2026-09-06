"""Windkessel boundary condition implementation.

.. deprecated::
    Use ``foampilot.workflows.medical.windkessel`` instead.
"""
import warnings
warnings.warn(
    "foampilot.model_addon.windkessel is deprecated, "
    "use foampilot.workflows.medical.windkessel instead",
    DeprecationWarning,
    stacklevel=2
)

from foampilot.workflows.medical.windkessel import Windkessel, WindkesselModel

__all__ = ["Windkessel", "WindkesselModel"]