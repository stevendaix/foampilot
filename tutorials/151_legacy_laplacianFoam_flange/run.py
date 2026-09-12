"""OpenFOAM 13 legacy laplacianFoam/flange through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path(
    "/opt/openfoam13/tutorials/legacy/basic/laplacianFoam/flange"
)
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")


def import_reference_case(solver: Solver, case_path: Path) -> None:
    """Import the complete OF13 case tree through FoamPilot APIs."""
    for source in REFERENCE.rglob("*"):
        if not source.is_file() or source.name in {"Allrun", "Allclean"}:
            continue
        relative = source.relative_to(REFERENCE)
        if relative.parts and relative.parts[0] == "0":
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=relative.relative_to("0")
            )
        else:
            solver.import_reference_asset(source, case_path / relative)


def run(solver: Solver, executable: str, args: list[str], log_name: str) -> None:
    solver.run_command(
        [str(OF13_BIN / executable), *args],
        log_filename=log_name,
    )


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "laplacianFoam"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    # Official OF13 Allrun: convert the supplied Ansys mesh before solving.
    run(solver, "ansysToFoam", ["flange.ans", "-scale", "0.001"], "log.ansysToFoam")
    run(solver, "laplacianFoam", [], "log.laplacianFoam")
    run(solver, "foamToEnsight", [], "log.foamToEnsight")
    run(solver, "foamToEnsightParts", [], "log.foamToEnsightParts")
    run(solver, "foamToVTK", [], "log.foamToVTK")


if __name__ == "__main__":
    main()
