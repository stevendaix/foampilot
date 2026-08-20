#!/usr/bin/env python3
"""
Phase 1.4 — Minimal complete OpenFOAM case from foampilot.urban.

Produces a complete OpenFOAM case for 1 building so that checkMesh passes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.urban import (
    Building,
    UrbanModel,
    CFDDomain,
    WindFrame,
    CFDLOD,
    CFDSimplifier,
    GmshQuarterBuilder,
    MeshConfig,
)
from foampilot.solver import Solver
from shapely.geometry import Polygon


def main():
    case_path = Path("cases/phase1_one_building")
    case_path.mkdir(parents=True, exist_ok=True)

    urban = UrbanModel()
    urban.add_building(Building(
        id="B001",
        footprint=Polygon([(0, 0), (42, 0), (42, 18), (0, 18)]),
        ground_z=0.0,
        roof_z=12.5,
        source="manual",
    ))

    wind_frame = WindFrame(direction_deg=0.0, origin=urban.center_xy())
    domain = CFDDomain(
        upstream=8.0,
        downstream=15.0,
        lateral=4.0,
        top=2.5,
        extent_units="href",
        reference_height_method="Hmax",
    )

    geometry = CFDSimplifier(urban, lod=CFDLOD.LOD1).simplify(
        wind_frame=wind_frame,
        domain=domain,
    )

    builder = GmshQuarterBuilder(case_path, geometry)
    builder.build()
    builder.assign_patches()
    builder.build_mesh(MeshConfig(
        global_size=15.0,
        building_size=2.0,
        wake_size=4.0,
        ground_size=2.0,
    ))
    builder.export_openfoam()

    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = "kOmegaSST"
    solver.transient = False
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = 1.0
    solver.system.controlDict.deltaT = 1.0
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 1
    solver.system.controlDict.purgeWrite = 0
    solver.system.write()
    solver.constant.write()
    solver.setup_case()
    solver.boundary.initialize_boundary()
    solver.boundary.set_raw_condition("inlet", "U", {
        "type": "fixedValue",
        "value": "uniform (10 0 0)",
    })
    solver.boundary.set_raw_condition("outlet", "U", {
        "type": "pressureInletOutletVelocity",
        "value": "uniform (0 0 0)",
    })
    for patch_name in solver.boundary.fields["U"]:
        if patch_name in ("inlet", "outlet"):
            continue
        solver.boundary.set_raw_condition(patch_name, "U", {
            "type": "noSlip",
        })
    solver.boundary.write_boundary_conditions()

    print(f"Case generated: {case_path}")
    print(f"Domain box: {geometry.domain_box}")
    print(f"Buildings: {len(geometry.buildings)}")
    print("Run: checkMesh -case", case_path)


if __name__ == "__main__":
    main()
