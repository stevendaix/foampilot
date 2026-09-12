"""OpenFOAM 13 compressibleVoF/throttle through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/compressibleVoF/throttle")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "compressibleVoF"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    # Declarative generation: solver.system.write() already called in setup_case
    # Declarative generation: solver.constant.write() already called in setup_case
    # Declarative generation: solver.fields_manager.write_initial_fields()

    # Declarative generation: no files to remove

    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["refineMesh"], log_filename="log.refineMesh")
    solver.run_simulation(nb_proc=4, log_filename="log.compressibleVoF.parallel")


if __name__ == "__main__":
    main()
