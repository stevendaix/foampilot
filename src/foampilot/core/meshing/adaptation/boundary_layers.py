"""Configure Gmsh boundary layer options for near-wall inflation layers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def create_boundary_layers(
    wall_surfaces: List[int],
    case_dir: str | Path,
    n_layers: int = 3,
    thickness_factor: float = 0.5,
) -> None:
    """Configure Gmsh boundary layer options for near-wall inflation layers.

    Attempts to set Gmsh boundary layer options.  If the Gmsh build does
    not support them, falls back to the Distance+Threshold field which
    already provides wall-adjacent mesh refinement.

    Args:
        wall_surfaces: Surface tags for the wall patches.
        case_dir: Case directory (unused for Gmsh but kept for API
            consistency).
        n_layers: Number of boundary layer sublayers.
        thickness_factor: Fraction of local mesh size for total BL
            thickness.
    """
    import gmsh

    if n_layers > 0:
        options_set = 0
        for opt, val in [
            ("Mesh.BoundaryLayerElements", n_layers),
            ("Mesh.BoundaryLayerFactor", thickness_factor),
            ("Mesh.BoundaryLayerMaxThickness", thickness_factor * 2.0),
            ("Mesh.BoundaryLayers", 1),  # some versions use this boolean
        ]:
            try:
                gmsh.option.setNumber(opt, val)
                options_set += 1
            except Exception:
                pass

        if options_set > 0:
            logger.info("Boundary layers configured (%d options set)", options_set)
        else:
            logger.info(
                "Boundary layer options not available in this Gmsh build; "
                "relying on Distance+Threshold field for near-wall refinement"
            )
