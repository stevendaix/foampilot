"""OpenFOAM 13 integrations for FoamPilot.

The lightweight physics catalog is imported eagerly. Urban-climate builders are
loaded lazily so users can inspect or configure OpenFOAM modules without first
installing the optional geometry and fluid-property stack.
"""
from __future__ import annotations

from .physics import (
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
        from .urbanclimate import PROFILES, UrbanClimateCase, UrbanClimateProfile, materialize_all
        return locals()[name]
    if name in {"RegionSpec", "UrbanClimateNativeCaseBuilder"}:
        from .urbanclimate_native import RegionSpec, UrbanClimateNativeCaseBuilder
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
