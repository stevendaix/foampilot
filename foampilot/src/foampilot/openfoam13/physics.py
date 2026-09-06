"""OpenFOAM 13 physics configurations.

.. deprecated::
    Use ``foampilot.core.physics.openfoam13`` instead.
"""
import warnings
warnings.warn(
    "foampilot.openfoam13.physics is deprecated, "
    "use foampilot.core.physics.openfoam13 instead",
    DeprecationWarning,
    stacklevel=2
)

from foampilot.core.physics.openfoam13 import (
    DEFAULT_MODULES,
    ExternalModule,
    PhysicsConfig,
    SUPPORTED_MODULES,
    check_openfoam13_case,
    module_catalog,
)

__all__ = [
    "DEFAULT_MODULES",
    "ExternalModule",
    "PhysicsConfig",
    "SUPPORTED_MODULES",
    "check_openfoam13_case",
    "module_catalog",
]