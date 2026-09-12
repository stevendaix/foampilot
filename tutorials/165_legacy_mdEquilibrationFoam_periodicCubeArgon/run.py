"""OpenFOAM 13 legacy mdEquilibrationFoam/periodicCubeArgon via FoamPilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path(
    "/opt/openfoam13/tutorials/legacy/lagrangian/mdEquilibrationFoam/periodicCubeArgon"
)
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")


def import_reference_case(solver: Solver, case_path: Path) -> None:
    """Import the complete OF13 molecular-dynamics case through FoamPilot."""
    # Declarative generation: solver.fields_manager.write_initial_fields()
    # Declarative generation: solver.constant.write() already called in setup_case
    # Declarative generation: solver.system.write() already called in setup_case


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "mdEquilibrationFoam"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    solver.run_command(
        [str(OF13_BIN / "blockMesh")],
        log_filename="log.blockMesh",
    )
    solver.run_command(
        [str(OF13_BIN / "mdInitialise")],
        log_filename="log.mdInitialise",
    )
    solver.run_command(
        [str(OF13_BIN / "mdEquilibrationFoam")],
        log_filename="log.mdEquilibrationFoam",
    )


if __name__ == "__main__":
    main()
