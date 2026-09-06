"""Windkessel boundary condition models.

.. deprecated::
    Use ``foampilot.workflows.medical.windkessel`` instead.
"""
import warnings
warnings.warn(
    "foampilot.model_addon is deprecated, "
    "use foampilot.workflows.medical.windkessel instead",
    DeprecationWarning,
    stacklevel=2
)

from foampilot.workflows.medical.windkessel import WindkesselModel

__all__ = ["WindkesselModel"]