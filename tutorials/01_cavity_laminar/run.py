#!/usr/bin/env python3
"""Tutoriel 1 : cavité entraînée, écoulement incompressible laminaire.

Référence OpenFOAM 13 : ``$FOAM_TUTORIALS/fluid/cavity``.
Cette adaptation reconstruit la mise en données du cas source avec l'API
Foampilot uniquement.
"""

from pathlib import Path

from foampilot import Meshing
from foampilot.solver import Solver


def build_cavity_mesh(case_path: Path) -> None:
    """Construire et exécuter le maillage blockMesh de la cavité."""
    meshing = Meshing(case_path, mesher="blockMesh")
    blockmesh = meshing.mesher
    blockmesh.scale = 1.0
    blockmesh.vertices = [
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0], [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.01], [0.1, 0.0, 0.01], [0.1, 0.1, 0.01], [0.0, 0.1, 0.01],
    ]
    blockmesh.blocks = [
        "hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1)",
    ]
    blockmesh.edges = []
    blockmesh.defaultPatch = {"type": "empty"}
    blockmesh.boundary = {
        "movingWall": {"type": "wall", "faces": [[3, 7, 6, 2]]},
        "fixedWalls": {
            "type": "wall",
            "faces": [[0, 4, 5, 1], [0, 3, 7, 4], [1, 2, 6, 5]],
        },
        "frontAndBack": {
            "type": "empty",
            "faces": [[0, 1, 2, 3], [4, 7, 6, 5]],
        },
    }
    blockmesh.mergePatchPairs = []
    blockmesh.write(case_path / "system" / "blockMeshDict")
    blockmesh.run()


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.turbulence_model = "laminar"
    solver.transient = True

    solver.system.controlDict.use_solver_keyword = True
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.stopAt = "endTime"
    solver.system.controlDict.endTime = 1.0
    solver.system.controlDict.deltaT = 0.01
    solver.system.controlDict.writeControl = "runTime"
    solver.system.controlDict.writeInterval = 0.1
    solver.system.write()

    build_cavity_mesh(case_path)
    solver.constant.write()
    solver.setup_case()

    solver.boundary.initialize_boundary()
    solver.boundary.set_raw_condition(
        "movingWall", "U", {"type": "fixedValue", "value": "uniform (1 0 0)"}
    )
    solver.boundary.set_raw_condition("movingWall", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("fixedWalls", "U", {"type": "noSlip"})
    solver.boundary.set_raw_condition("fixedWalls", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("frontAndBack", "U", {"type": "empty"})
    solver.boundary.set_raw_condition("frontAndBack", "p", {"type": "empty"})
    solver.boundary.write_boundary_conditions()
    solver.run_simulation(nb_proc=1, log_filename="log.cavity")


if __name__ == "__main__":
    main()
