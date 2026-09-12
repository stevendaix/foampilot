"""OpenFOAM 13 fluid/squareBend through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/fluid/squareBend")


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
    # The reference residualControl intentionally stops the run at convergence;
    # remove it in the generated validation case so the official endTime is reached.
    solver.system.remove_dictionary_entries("system/fvSolution", ["PIMPLE/residualControl"])
    # Declarative generation: mesh.mesher.write() generates blockMeshDict
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_simulation(nb_proc=1, log_filename="log.fluid")


if __name__ == "__main__":
    main()
