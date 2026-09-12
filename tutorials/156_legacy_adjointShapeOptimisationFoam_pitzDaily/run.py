"""OpenFOAM 13 legacy adjointShapeOptimisationFoam/pitzDaily via FoamPilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path(
    "/opt/openfoam13/tutorials/legacy/incompressible/adjointShapeOptimisationFoam/pitzDaily"
)
MESH_REFERENCE = Path("/opt/openfoam13/tutorials/resources/blockMesh/pitzDaily")
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")


def import_reference_case(solver: Solver, case_path: Path) -> None:
    """Import all OF13 fields and dictionaries through FoamPilot APIs."""
    # Declarative generation: solver.fields_manager.write_initial_fields()
    # Declarative generation: solver.constant.write() already called in setup_case
    # Declarative generation: solver.system.write() already called in setup_case
    solver.system.import_reference_file(MESH_REFERENCE, "pitzDaily")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "adjointShapeOptimisationFoam"
    solver.transient = False
    solver.setup_case()
    import_reference_case(solver, case_path)

    solver.run_command(
        [str(OF13_BIN / "blockMesh"), "-dict", "system/pitzDaily"],
        log_filename="log.blockMesh",
    )
    solver.run_command(
        [str(OF13_BIN / "adjointShapeOptimisationFoam")],
        log_filename="log.adjointShapeOptimisationFoam",
    )


if __name__ == "__main__":
    main()
