"""OpenFOAM 13 incompressibleVoF/cavitatingBullet through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleVoF/cavitatingBullet")


def import_reference_case(solver: Solver, destination: Path) -> None:
    for source in REFERENCE.rglob("*"):
        if not source.is_file() or source.name in {"Allrun", "Allclean"}:
            continue
        relative = source.relative_to(REFERENCE)
        if relative.parts and relative.parts[0] == "0":
            field_name = relative.relative_to("0")
            if field_name.suffix == ".orig":
                field_name = field_name.with_suffix("")
            solver.fields_manager.import_reference_field(
                source, destination, field_name=field_name
            )
        else:
            solver.import_reference_asset(source, destination / relative)


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "incompressibleVoF"
    solver.setup_case()
    import_reference_case(solver, case_path)
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["snappyHexMesh"], log_filename="log.snappyHexMesh")
    solver.run_command(
        ["potentialFoam", "-pName", "p_rgh"], log_filename="log.potentialFoam"
    )
    solver.run_simulation(log_filename="log.incompressibleVoF")


if __name__ == "__main__":
    main()
