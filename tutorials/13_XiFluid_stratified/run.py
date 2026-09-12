"""OpenFOAM 13 XiFluid/stratified through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/XiFluid/stratified")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "XiFluid"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    # Preserve the complete OF13 dictionaries through public FoamPilot APIs.
    # Declarative generation: solver.system.write() already called in setup_case
    # Declarative generation: solver.constant.write() already called in setup_case
    # Declarative generation: solver.fields_manager.write_initial_fields()

    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["setFields"], log_filename="log.setFields")
    solver.run_simulation(nb_proc=1, log_filename="log.XiFluid")


if __name__ == "__main__":
    main()
