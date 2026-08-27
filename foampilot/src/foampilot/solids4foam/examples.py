"""Ready-to-run solids4foam examples built entirely through Foampilot APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from foampilot.workflows import OpenFOAMWorkflow
from .case import SolidMaterial, Solids4FoamCase


def _require_gmsh() -> Any:
    try:
        import gmsh
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install the optional 'gmsh' dependency to build examples") from error
    return gmsh


def _fragmented_two_region_geometry(gmsh: Any) -> tuple[list[int], list[int], list[int]]:
    """Create adjacent conformal boxes and return fluid/solid/interface tags."""
    fluid = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 0.4, 0.4)
    solid = gmsh.model.occ.addBox(1.0, 0.0, 0.0, 0.12, 0.4, 0.4)
    result, _ = gmsh.model.occ.fragment([(3, fluid)], [(3, solid)], removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()
    volumes = [tag for dim, tag in result if dim == 3]
    fluid_volumes = [tag for tag in volumes if gmsh.model.occ.getCenterOfMass(3, tag)[0] < 1.0]
    solid_volumes = [tag for tag in volumes if gmsh.model.occ.getCenterOfMass(3, tag)[0] >= 1.0]
    if not fluid_volumes or not solid_volumes:
        raise RuntimeError("fragment did not produce separate FLUID and SOLID volumes")
    def boundary_surfaces(volume: int) -> set[int]:
        return {tag for dim, tag in gmsh.model.getBoundary([(3, volume)], oriented=False) if dim == 2}
    shared = set.intersection(*(boundary_surfaces(tag) for tag in fluid_volumes)) & set.intersection(*(boundary_surfaces(tag) for tag in solid_volumes))
    if not shared:
        shared = set.intersection(boundary_surfaces(fluid_volumes[0]), boundary_surfaces(solid_volumes[0]))
    if not shared:
        raise RuntimeError("unable to identify the conformal fluid-solid interface")
    return fluid_volumes, solid_volumes, sorted(shared)


def build_beam_in_cross_flow(
    case_path: str | Path,
    *,
    coupling: str = "IQNILS",
    parallel: bool = False,
    generate_mesh: bool = True,
) -> tuple[Solids4FoamCase, OpenFOAMWorkflow]:
    """Build the Foampilot tutorial case and return its declarative workflow."""
    gmsh = _require_gmsh()
    root = Path(case_path)
    gmsh.initialize()
    try:
        gmsh.model.add("beamInCrossFlow")
        fluid, solid, interface = _fragmented_two_region_geometry(gmsh)
        if generate_mesh:
            gmsh.model.mesh.generate(3)
        case = Solids4FoamCase(
            root,
            fluid_patch="interface",
            solid_patch="interface",
            coupling=coupling,
            material=SolidMaterial(
                name="rubber", law="neoHookeanElastic", density=1000.0,
                young_modulus=1.0e4, poisson_ratio=0.4,
            ),
        )
        case.prepare_from_gmsh_entities(
            fluid_volumes=fluid,
            solid_volumes=solid,
            interface_surfaces=interface,
        )
    finally:
        gmsh.finalize()
    workflow = solids4foam_workflow(root, parallel=parallel)
    return case, workflow


def build_partition_validation(
    case_path: str | Path,
    *,
    generate_mesh: bool = True,
) -> tuple[Solids4FoamCase, OpenFOAMWorkflow]:
    """Build a small deterministic two-region partition validation case."""
    case, workflow = build_beam_in_cross_flow(
        case_path, coupling="fixedRelaxation", parallel=False, generate_mesh=generate_mesh
    )
    return case, workflow


def solids4foam_workflow(case_path: str | Path, *, parallel: bool = False) -> OpenFOAMWorkflow:
    """Return a Foampilot workflow; no shell script or RunFunctions is used."""
    root = Path(case_path)
    workflow = OpenFOAMWorkflow(root, name="solids4foam-fsi")
    workflow.add_command("solids4foam", "solids4Foam", *( ["-parallel"] if parallel else [] ))
    return workflow
