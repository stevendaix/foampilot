"""OpenFOAM 13 incompressibleFluid/windAroundBuildings through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleFluid/windAroundBuildings")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "incompressibleFluid"
    solver.transient = False
    solver.turbulence_model = "kEpsilon"
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    for source in (REFERENCE / "system").rglob("*"):
        if source.is_file():
            solver.system.import_reference_file(
                source, filename=source.relative_to(REFERENCE / "system")
            )
    for source in (REFERENCE / "constant").rglob("*"):
        if source.is_file():
            solver.constant.import_reference_file(
                source, filename=source.relative_to(REFERENCE / "constant")
            )
    for source in (REFERENCE / "0").rglob("*"):
        if source.is_file():
            relative = source.relative_to(REFERENCE / "0")
            field_name = relative.with_suffix("") if relative.name.endswith(".orig") else relative
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=field_name
            )

    solver.run_command(["surfaceFeatures"], log_filename="log.surfaceFeatures")
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(
        ["snappyHexMesh", "-overwrite"], log_filename="log.snappyHexMesh"
    )
    solver.run_simulation(log_filename="log.incompressibleFluid")


if __name__ == "__main__":
    main()
