#!/usr/bin/env python3
"""
OpenFOAM case setup from CAD-reconstructed TBAD geometry.
Based on run.py but using our CAD pipeline output.
"""
import shutil
from pathlib import Path

from foampilot import Meshing, ValueWithUnit, Solver


def setup_cad_case(stl_path: Path, case_dir: Path):
    case_dir.mkdir(parents=True, exist_ok=True)

    rho = ValueWithUnit(1060, "kg/m^3")
    nu = ValueWithUnit(3.77e-6, "m^2/s")

    solver = Solver(case_dir)
    solver.compressible = False
    solver.with_gravity = False
    solver.transient = False
    solver.turbulence_model = "laminar"
    solver.constant.transportProperties.nu = nu
    solver.constant.transportProperties.rho = rho

    solver.system.controlDict.application = "foamRun"
    solver.system.controlDict.startTime = 0
    solver.system.controlDict.endTime = 1
    solver.system.controlDict.deltaT = 1
    solver.system.controlDict.writeInterval = 1
    solver.system.controlDict.adjustTimeStep = False

    solver.system.fvSolution.solvers["p"]["solver"] = "smoothSolver"
    solver.system.fvSolution.solvers["p"]["smoother"] = "GAMG"
    solver.system.fvSolution.solvers["U"]["solver"] = "smoothSolver"

    solver.system.write()
    solver.constant.write()

    stl_dest = case_dir / "constant" / "triSurface"
    stl_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(stl_path, stl_dest / stl_path.name)

    mesh = Meshing(case_dir, mesher="snappy")
    snappy = mesh.mesher
    snappy.stl_file = stl_path.name
    snappy.locationInMesh = (-16, -21, -12)
    snappy.geometry = {
        "wall_aorta": {
            "type": "triSurfaceMesh",
            "file": stl_path.name,
            "name": "wall_aorta",
        }
    }
    snappy.castellatedMeshControls["refinementSurfaces"] = {
        "wall_aorta": {"level": (0, 1)}
    }
    snappy.addLayers = True
    snappy.add_layer("wall_aorta", 2)
    snappy.addLayersControls["finalLayerThickness"] = 0.2
    mesh.write()
    snappy.run()

    solver.boundary.initialize_boundary()
    solver.boundary.fields["U"]["wall_aorta"] = {"type": "noSlip"}
    solver.boundary.fields["p"]["wall_aorta"] = {"type": "zeroGradient"}
    solver.boundary.write_boundary_conditions()

    print(f"Case {case_dir} set up from {stl_path}")


if __name__ == "__main__":
    base = Path("/home/steven/foampilot/examples/coa")
    stl = base / "data_preproc" / "tbad_stl_output" / "tbad_TL_walls.stl"
    case = base / "cadtbad_test"
    setup_cad_case(stl, case)
