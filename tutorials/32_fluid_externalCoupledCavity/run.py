"""OpenFOAM 13 fluid/externalCoupledCavity through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/fluid/externalCoupledCavity")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "fluid"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    mesh = Meshing(case_path, mesher="blockMesh")
    # Declarative generation: solver.system.write() already called in setup_case
    # Declarative generation: solver.constant.write() already called in setup_case
    # Declarative generation: solver.fields_manager.write_initial_fields()

    # Declarative generation: no files to remove
    # Declarative generation: mesh.mesher.write() generates blockMeshDict
    solver.import_reference_asset(REFERENCE / "externalSolver", "externalSolver")

    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(
        ["createExternalCoupledPatchGeometry", "T"],
        log_filename="log.createExternalCoupledPatchGeometry",
    )

    foam_process = solver.run_command_async(["foamRun"], "log.fluid")
    external_process = solver.run_command_async(["./externalSolver"], "log.externalSolver")
    solver.wait_command(foam_process)
    solver.wait_command(external_process, check=False)


if __name__ == "__main__":
    main()
