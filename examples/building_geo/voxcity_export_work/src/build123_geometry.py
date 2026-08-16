#!/usr/bin/env python3
"""
build123d geometry construction for VoxCity CFD domain.

This module builds the fluid solid with buildings subtracted using build123d.
It is independent from Gmsh and returns a build123d.Solid ready for import.
"""
from __future__ import annotations

from typing import Optional

from build123d import Box, BuildPart, Compound, Location, Part, Solid

from foampilot.urban.model.urban_model import UrbanModel
from foampilot.urban.model.terrain import CFDTerrain


def build_fluid_solid(
    urban: UrbanModel,
    terrain: CFDTerrain,
    margin: Optional[float] = None,
    bottom_offset: float = 5.0,
) -> Solid:
    """Build the CFD fluid domain with buildings subtracted.

    Args:
        urban: Urban model containing buildings.
        terrain: CFD terrain model.
        margin: Domain margin around buildings. If None, follows
            building_aero rules: upstream=4*Hmax, downstream=7.5*Hmax,
            lateral=2*D, top=1.25*Hmax.
        bottom_offset: Extra depth below ground level.

    Returns:
        build123d.Solid of the fluid domain.

    Raises:
        RuntimeError: If no buildings are available.
    """
    if not urban.buildings():
        raise RuntimeError("No buildings to build")

    bbox = urban.bbox()
    xmin, ymin, zmin, xmax, ymax, zmax = bbox

    if margin is None:
        heights = [b.roof_z - b.ground_z for b in urban.buildings()]
        h_max = max(heights) if heights else 10.0
        d = xmax - xmin
        upstream = 4.0 * h_max
        downstream = 7.5 * h_max
        lateral = 2.0 * max(d, 1.0)
        top = 1.25 * h_max
        domain_xmin = xmin - upstream
        domain_ymin = ymin - lateral
        domain_zmin = zmin - bottom_offset
        domain_xmax = xmax + downstream
        domain_ymax = ymax + lateral
        domain_zmax = zmax + top
    else:
        domain_xmin = xmin - margin
        domain_ymin = ymin - margin
        domain_zmin = zmin - bottom_offset
        domain_xmax = xmax + margin
        domain_ymax = ymax + margin
        domain_zmax = zmax + margin

    dx = domain_xmax - domain_xmin
    dy = domain_ymax - domain_ymin
    dz = domain_zmax - domain_zmin

    # Build in local coordinates to avoid OCC precision issues
    local_origin = (domain_xmin, domain_ymin, domain_zmin)
    dx = domain_xmax - domain_xmin
    dy = domain_ymax - domain_ymin
    dz = domain_zmax - domain_zmin

    # Create fluid box in local coords, spanning exactly [0,dx] x [0,dy] x [0,dz]
    with BuildPart() as fluid_builder:
        Box(dx, dy, dz)
        fluid_local = fluid_builder.part.moved(Location((dx / 2, dy / 2, dz / 2)))

    # Create all buildings as a single compound in local coords
    bldg_parts = []
    for building in urban.buildings():
        base_z = building.ground_z if building.ground_z is not None else 0.0
        roof_z = building.roof_z if building.roof_z is not None else (base_z + 10.0)
        height = roof_z - base_z
        ll = building.footprint.bounds
        bdx = ll[2] - ll[0]
        bdy = ll[3] - ll[1]
        cx_local = (ll[0] + ll[2]) / 2.0 - domain_xmin
        cy_local = (ll[1] + ll[3]) / 2.0 - domain_ymin
        cz_local = base_z - domain_zmin

        with BuildPart() as bldg_builder:
            Box(bdx, bdy, height)
            bldg_builder.part.location = Location((cx_local, cy_local, cz_local))
            bldg_parts.append(bldg_builder.part)

    buildings_compound = Compound(bldg_parts)

    # Single cut with all buildings
    fluid_cut = fluid_local.cut(buildings_compound)

    # Move back to global coordinates
    fluid = fluid_cut.moved(Location(local_origin))

    return fluid
