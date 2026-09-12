"""OpenFOAM 13 fluid/shockTube through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/fluid/shockTube")


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
    solver.import_reference_asset(REFERENCE / "createGraphs", Path("createGraphs"))

    # Declarative generation: no files to remove
    # Declarative generation: mesh.mesher.write() generates blockMeshDict
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["setFields"], log_filename="log.setFields")
    solver.run_simulation(nb_proc=1, log_filename="log.fluid")
    solver.system.run_utility("sh", ["createGraphs"], log_filename="log.createGraphs")


if __name__ == "__main__":
    main()
