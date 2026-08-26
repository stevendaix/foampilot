"""OpenFOAM 13 incompressibleFluid/turbineSiting through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleFluid/turbineSiting")


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
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=relative
            )

    Meshing(case_path, mesher="blockMesh").mesher.import_reference_dict(
        REFERENCE / "system" / "blockMeshDict"
    )
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["decomposePar", "-copyZero"], log_filename="log.decomposePar")
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "4", "snappyHexMesh", "-parallel"],
        log_filename="log.snappyHexMesh",
    )
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "4", "createZones", "-parallel"],
        log_filename="log.createZones",
    )
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "4", "foamRun", "-solver", "incompressibleFluid", "-parallel"],
        log_filename="log.incompressibleFluid",
    )
    solver.run_command(["reconstructPar"], log_filename="log.reconstructPar")


if __name__ == "__main__":
    main()
