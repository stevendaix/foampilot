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

    # Declarative generation: solver.system.write() already called in setup_case
    solver.system.import_reference_file(BLOCK_MESH, "blockMeshDict")
    # Declarative generation: solver.constant.write() already called in setup_case
    # Declarative generation: solver.fields_manager.write_initial_fields()

    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_simulation(nb_proc=1, log_filename="log.compressibleVoF")


if __name__ == "__main__":
    main()
