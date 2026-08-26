"""Case preparation helpers for the solids4foam toolbox."""

from .gmsh_regions import GmshRegionError, create_fsi_physical_groups
from .case import (
    Solids4FoamCase,
    Solids4FoamConfigurationError,
    SolidMaterial,
    write_solids4foam_case,
)

__all__ = [
    "Solids4FoamCase",
    "Solids4FoamConfigurationError",
    "SolidMaterial",
    "write_solids4foam_case",
    "GmshRegionError",
    "create_fsi_physical_groups",
]
