#!/usr/bin/env python3
"""OpenFOAM 13 fluid/cavity generated declaratively with FoamPilot."""
from pathlib import Path
import os
from foampilot import Meshing, OpenFOAMEnvironment
from foampilot.solver import Solver
from foampilot.utilities.dictonnary import OpenFOAMDictAddFile


def build_mesh(case_path: Path) -> None:
    meshing = Meshing(case_path, mesher="blockMesh")
    block = meshing.mesher
    block.scale = 0.1
    block.vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                      [0, 0, 0.1], [1, 0, 0.1], [1, 1, 0.1], [0, 1, 0.1]]
    block.blocks = ["hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1)"]
    block.edges = []
    block.boundary = {
        "movingWall": {"type": "wall", "faces": [[3, 7, 6, 2]]},
        "fixedWalls": {"type": "wall", "faces": [[0, 4, 7, 3], [2, 6, 5, 1], [1, 5, 4, 0]]},
        "frontAndBack": {"type": "empty", "faces": [[0, 3, 2, 1], [4, 5, 6, 7]]},
    }
    block.mergePatchPairs = []
    block.write(case_path / "system" / "blockMeshDict")
    block.run()


def main() -> None:
    os.environ.update(OpenFOAMEnvironment().environment())
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.compressible = True
    solver.with_gravity = False
    solver.turbulence_model = "laminar"
    solver.solver_name = "fluid"
    solver.transient = True
    solver.setup_case()
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.endTime = 1.0
    solver.system.controlDict.deltaT = 0.01
    solver.system.controlDict.writeControl = "runTime"
    solver.system.controlDict.writeInterval = 0.1
    solver.system.write()
    build_mesh(case_path)
    solver.constant.write()
    OpenFOAMDictAddFile(object_name="physicalProperties",
        thermoType={"type": "hePsiThermo", "mixture": "pureMixture", "transport": "const",
                    "thermo": "hConst", "equationOfState": "perfectGas", "specie": "specie",
                    "energy": "sensibleEnthalpy"},
        mixture={"specie": {"molWeight": 28.9}, "thermodynamics": {"Cp": 1007, "hf": 0},
                 "transport": {"mu": 1.84e-5, "Pr": 0.7}}).write(
                     "physicalProperties", solver.case_path, folder="constant")
    solver.fields_manager.register_field("T", "uniform 300")
    solver.boundary.initialize_boundary()
    solver.boundary.set_raw_condition("movingWall", "U", {"type": "fixedValue", "value": "uniform (1 0 0)"})
    solver.boundary.set_raw_condition("movingWall", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("movingWall", "T", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("fixedWalls", "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("fixedWalls", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("fixedWalls", "T", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("frontAndBack", "U", {"type": "empty"})
    solver.boundary.set_raw_condition("frontAndBack", "p", {"type": "empty"})
    solver.boundary.set_raw_condition("frontAndBack", "T", {"type": "empty"})
    solver.boundary.write_boundary_conditions({"p": "uniform 100000", "T": "uniform 300"})
    solver.run_simulation(nb_proc=1, log_filename="log.fluid")


if __name__ == "__main__":
    main()
