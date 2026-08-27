"""Gmsh physical-group helpers for solids4foam regional cases."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class GmshRegionError(ValueError):
    """Raised when Gmsh entities cannot define a valid FSI partition."""


def _entity_tags(values: Iterable[int], name: str) -> list[int]:
    tags = [int(value) for value in values]
    if not tags:
        raise GmshRegionError(f"{name} must contain at least one Gmsh entity tag")
    if any(value <= 0 for value in tags):
        raise GmshRegionError(f"{name} contains an invalid Gmsh entity tag")
    return list(dict.fromkeys(tags))


def create_fsi_physical_groups(
    *,
    fluid_volumes: Iterable[int],
    solid_volumes: Iterable[int],
    interface_surfaces: Iterable[int],
    fluid_name: str = "FLUID",
    solid_name: str = "SOLID",
    interface_name: str = "interface",
    gmsh_module: Any | None = None,
) -> dict[str, int]:
    """Create the physical groups required by ``Solids4FoamCase``.

    The caller supplies CAD entity tags returned by ``gmsh.model.occ``. The
    interface surfaces must be the common CAD faces of both volumes, normally
    obtained after ``occ.fragment`` or another conformal boolean operation.
    """
    if gmsh_module is None:
        try:
            import gmsh as gmsh_module
        except ImportError as error:  # pragma: no cover - optional dependency
            raise GmshRegionError("create_fsi_physical_groups requires gmsh") from error
    fluid = _entity_tags(fluid_volumes, "fluid_volumes")
    solid = _entity_tags(solid_volumes, "solid_volumes")
    interface = _entity_tags(interface_surfaces, "interface_surfaces")
    if set(fluid) & set(solid):
        raise GmshRegionError("fluid and solid volume tags must be disjoint")
    created: dict[str, int] = {}
    for dim, tags, name in (
        (3, fluid, fluid_name),
        (3, solid, solid_name),
        (2, interface, interface_name),
    ):
        if not name or any(char.isspace() for char in name):
            raise GmshRegionError(f"{name!r} is not a valid physical-group name")
        group_tag = gmsh_module.model.addPhysicalGroup(dim, tags)
        gmsh_module.model.setPhysicalName(dim, group_tag, name)
        created[name] = int(group_tag)
    return created
