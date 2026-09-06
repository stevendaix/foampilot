"""OpenFOAM 13 integrations for FoamPilot.

.. deprecated::
    This module is deprecated. Use:
    - ``foampilot.core.physics.openfoam13`` for physics configs
    - ``foampilot.workflows.urban_climate`` for urban climate workflows
"""
import warnings
warnings.warn(
    "foampilot.openfoam13 is deprecated, use foampilot.core.physics.openfoam13 "
    "or foampilot.workflows.urban_climate instead",
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

_URBAN_EXPORTS = {
    "PROFILES",
    "UrbanClimateCase",
    "UrbanClimateProfile",
    "materialize_all",
    "RegionSpec",
    "UrbanClimateNativeCaseBuilder",
}


def __getattr__(name: str):
    if name in {"PROFILES", "UrbanClimateCase", "UrbanClimateProfile", "materialize_all"}:
        from foampilot.workflows.urban_climate.urbanclimate import (
            PROFILES,
            UrbanClimateCase,
            UrbanClimateProfile,
            materialize_all,
        )
        return locals()[name]
    if name in {"RegionSpec", "UrbanClimateNativeCaseBuilder"}:
        from foampilot.workflows.urban_climate.urbanclimate_native import (
            RegionSpec,
            UrbanClimateNativeCaseBuilder,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_MODULES",
    "ExternalModule",
    "PhysicsConfig",
    "SUPPORTED_MODULES",
    "check_openfoam13_case",
    "module_catalog",
    "PROFILES",
    "UrbanClimateCase",
    "UrbanClimateProfile",
    "materialize_all",
    "RegionSpec",
    "UrbanClimateNativeCaseBuilder",
]