"""OpenFOAM 13 fluid/blockedChannel through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/fluid/blockedChannel")


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

    solver.constant.remove_files(["transportProperties", "turbulenceProperties", "pRef"])
    mesh.mesher.import_reference_dict(REFERENCE / "system" / "blockMeshDict")

    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(
        ["foamPostProcess", "-func", "generateAlphas"],
        log_filename="log.generateAlphas",
    )
    solver.run_simulation(nb_proc=1, log_filename="log.fluid")


if __name__ == "__main__":
    main()
