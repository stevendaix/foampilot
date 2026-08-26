"""OpenFOAM 13 integrations for FoamPilot."""
from .urbanclimate import PROFILES, UrbanClimateCase, UrbanClimateProfile, materialize_all
from .urbanclimate_native import RegionSpec, UrbanClimateNativeCaseBuilder
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
    "PROFILES",
    "UrbanClimateCase",
    "UrbanClimateProfile",
    "materialize_all",
    "RegionSpec",
    "UrbanClimateNativeCaseBuilder",
]
