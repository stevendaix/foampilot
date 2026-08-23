"""OpenFOAM 13 tutorial 01: 2-D laminar channel generated with FoamPilot.

Run from this directory with::

    python run.py

The case is generated from Python; no OpenFOAM dictionary is copied or
edited by hand.
"""

from pathlib import Path

from foampilot import Meshing, ValueWithUnit
from foampilot.solver import Solver


CASE_PATH = Path(__file__).resolve().parent / "case"


def configure_mesh(case_path: Path) -> None:
    mesh = Meshing(case_path, mesher="blockMesh").mesher
    mesh.vertices = [
        [0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0],
        [0, 0, 0.05], [2, 0, 0.05], [2, 1, 0.05], [0, 1, 0.05],
    ]
    mesh.blocks = ["hex (0 1 2 3 4 5 6 7) (40 20 1) simpleGrading (1 1 1)"]
    mesh.edges = []
    mesh.defaultPatch = {"defaultFaces": "empty"}
    mesh.boundary = {
        "inlet": {"type": "patch", "faces": [[3, 0, 4, 7]]},
        "outlet": {"type": "patch", "faces": [[1, 2, 6, 5]]},
        "top": {"type": "wall", "faces": [[2, 3, 7, 6]]},
        "bottom": {"type": "wall", "faces": [[0, 1, 5, 4]]},
        "frontAndBack": {
            "type": "empty", "faces": [[0, 3, 2, 1], [4, 5, 6, 7]]
        },
    }
    mesh.mergePatchPairs = []
    mesh.write(case_path / "system" / "blockMeshDict")
    mesh.run()


def main() -> None:
    CASE_PATH.mkdir(parents=True, exist_ok=True)
    solver = Solver(CASE_PATH)
    solver.energy_activated = False
    solver.transient = False
    solver.turbulence_model = "laminar"
    solver.constant.transportProperties.nu = ValueWithUnit(1.5e-5, "m^2/s")
    solver.system.controlDict.application = "foamRun"
    solver.system.controlDict.startTime = 0
    solver.system.controlDict.endTime = 100
    solver.system.controlDict.writeControl = "timeStep"
    solver.system.controlDict.writeInterval = 100

    solver.setup_case()
    solver.system.write()
    solver.constant.write()
    configure_mesh(CASE_PATH)

    solver.boundary.initialize_boundary()
    solver.boundary.set_raw_condition("inlet", "U", {
        "type": "fixedValue", "value": "uniform (1 0 0)"
    })
    solver.boundary.set_raw_condition("outlet", "U", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("inlet", "p", {"type": "zeroGradient"})
    solver.boundary.set_raw_condition("outlet", "p", {
        "type": "fixedValue", "value": "uniform 0"
    })
    for patch in ("top", "bottom"):
        solver.boundary.set_raw_condition(patch, "U", {"type": "noSlip"})
        solver.boundary.set_raw_condition(patch, "p", {"type": "zeroGradient"})
    solver.boundary.write_boundary_conditions()

    solver.run_simulation()
    print(f"Completed FoamPilot case: {CASE_PATH}")


if __name__ == "__main__":
    main()
