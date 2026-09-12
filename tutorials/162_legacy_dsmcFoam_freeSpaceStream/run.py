"""OpenFOAM 13 legacy dsmcFoam/freeSpaceStream via FoamPilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path(
    "/opt/openfoam13/tutorials/legacy/lagrangian/dsmcFoam/freeSpaceStream"
)
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")


def import_reference_case(solver: Solver, case_path: Path) -> None:
    """Import the complete OF13 DSMC stream case through FoamPilot managers."""
    # Declarative generation: solver.fields_manager.write_initial_fields()
    # Declarative generation: solver.constant.write() already called in setup_case
    # Declarative generation: solver.system.write() already called in setup_case


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "dsmcFoam"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    solver.run_command(
        [str(OF13_BIN / "blockMesh")],
        log_filename="log.blockMesh",
    )
    solver.run_command(
        [str(OF13_BIN / "dsmcInitialise")],
        log_filename="log.dsmcInitialise",
    )
    solver.run_command(
        [str(OF13_BIN / "dsmcFoam")],
        log_filename="log.dsmcFoam",
    )


if __name__ == "__main__":
    main()
