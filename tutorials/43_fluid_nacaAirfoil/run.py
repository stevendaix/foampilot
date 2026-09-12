"""OpenFOAM 13 fluid/nacaAirfoil through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/fluid/nacaAirfoil")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "fluid"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    # Declarative generation: solver.system.write() already called in setup_case
    # Declarative generation: solver.constant.write() already called in setup_case
    # Declarative generation: solver.fields_manager.write_initial_fields()
    for source in (REFERENCE / "prostar").iterdir():
        if source.is_file():
            solver.import_reference_asset(source, Path("prostar") / source.name)

    # Declarative generation: no files to remove
    solver.system.run_utility("star3ToFoam", ["prostar/nacaAirfoil"], log_filename="log.star3ToFoam")
    solver.system.replace_file_text(
        "constant/polyMesh/boundary",
        "symmetry;",
        "empty;",
    )
    solver.run_simulation(nb_proc=4, log_filename="log.fluid.parallel")


if __name__ == "__main__":
    main()
