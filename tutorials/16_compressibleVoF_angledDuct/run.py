"""OpenFOAM 13 compressibleVoF/angledDuct through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/compressibleVoF/angledDuct")
BLOCK_MESH = Path("/opt/openfoam13/tutorials/resources/blockMesh/angledDuct")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "compressibleVoF"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    for source in (REFERENCE / "system").iterdir():
        if source.is_file():
            solver.system.import_reference_file(source)
    solver.system.import_reference_file(BLOCK_MESH, "blockMeshDict")
    for source in (REFERENCE / "constant").iterdir():
        if source.is_file():
            solver.constant.import_reference_file(source)
    for source in (REFERENCE / "0").iterdir():
        if source.is_file():
            solver.fields_manager.import_reference_field(source, case_path)

    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_simulation(nb_proc=1, log_filename="log.compressibleVoF")


if __name__ == "__main__":
    main()
