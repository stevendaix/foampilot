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
    for source in (REFERENCE / "system").iterdir():
        if source.is_file():
            solver.system.import_reference_file(source)
    for source in (REFERENCE / "constant").iterdir():
        if source.is_file():
            solver.constant.import_reference_file(source)
    for source in (REFERENCE / "0").iterdir():
        if source.is_file():
            solver.fields_manager.import_reference_field(source, case_path)

    solver.constant.remove_files(["transportProperties", "turbulenceProperties"])
    mesh.mesher.import_reference_dict(REFERENCE / "system" / "blockMeshDict")
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
