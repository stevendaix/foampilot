"""OpenFOAM 13 incompressibleMultiphaseVoF/mixerVessel2DMRF."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path(
    "/opt/openfoam13/tutorials/incompressibleMultiphaseVoF/mixerVessel2DMRF"
)
RESOURCE = Path("/opt/openfoam13/tutorials/resources/blockMesh/mixerVessel2D")


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
    solver.solver_name = "incompressibleMultiphaseVoF"
    solver.setup_case()
    import_reference_case(solver, case_path)
    solver.import_reference_asset(RESOURCE, case_path / "system" / "mixerVessel2D")
    solver.run_command(
        ["blockMesh", "-dict", "system/mixerVessel2D"], log_filename="log.blockMesh"
    )
    solver.run_command(["setFields"], log_filename="log.setFields")
    solver.run_simulation(log_filename="log.incompressibleMultiphaseVoF")


if __name__ == "__main__":
    main()
