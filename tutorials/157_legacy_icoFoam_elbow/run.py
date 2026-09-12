"""OpenFOAM 13 legacy icoFoam/elbow via FoamPilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/legacy/incompressible/icoFoam/elbow")
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")


def import_reference_case(solver: Solver, case_path: Path) -> None:
    """Import all OF13 fields, dictionaries and the Fluent mesh through FoamPilot."""
    # Declarative generation: solver.fields_manager.write_initial_fields()
    # Declarative generation: solver.constant.write() already called in setup_case
    # Declarative generation: solver.system.write() already called in setup_case
    solver.import_reference_asset(REFERENCE / "elbow.msh", case_path / "elbow.msh")


def run(solver: Solver, executable: str, args: list[str], log_name: str) -> None:
    solver.run_command([str(OF13_BIN / executable), *args], log_filename=log_name)


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "icoFoam"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    run(solver, "fluentMeshToFoam", ["elbow.msh"], "log.fluentMeshToFoam")
    run(solver, "icoFoam", [], "log.icoFoam")
    run(solver, "foamMeshToFluent", [], "log.foamMeshToFluent")
    run(solver, "foamDataToFluent", [], "log.foamDataToFluent")


if __name__ == "__main__":
    main()
