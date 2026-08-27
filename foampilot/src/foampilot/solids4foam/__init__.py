"""Case preparation helpers for the solids4foam toolbox."""

from .gmsh_regions import GmshRegionError, create_fsi_physical_groups
from .examples import build_beam_in_cross_flow, build_partition_validation, solids4foam_workflow
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
    "build_beam_in_cross_flow",
    "build_partition_validation",
    "solids4foam_workflow",
]
