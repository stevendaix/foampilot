"""OpenFOAM 13 integrations for FoamPilot."""
from .physics import (
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
