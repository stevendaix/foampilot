"""OpenFOAM 13 incompressibleFluid/propeller through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleFluid/propeller")
GEOMETRY = Path("/opt/openfoam13/tutorials/resources/geometry")


GEOMETRY_FILES = (
    "propeller-innerCylinder.obj.gz",
    "propeller-middleCylinder.obj.gz",
    "propeller-outerCylinder.obj.gz",
    "propeller.obj.gz",
)


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "incompressibleFluid"
    solver.transient = True
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
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=source.relative_to(REFERENCE / "0")
            )
    for filename in GEOMETRY_FILES:
        source = GEOMETRY / filename
        solver.import_reference_asset(
            source,
            case_path / "constant" / "geometry" / filename,
        )

    Meshing(case_path, mesher="blockMesh").mesher.import_reference_dict(
        REFERENCE / "system" / "blockMeshDict"
    )
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["surfaceFeatures"], log_filename="log.surfaceFeatures")
    solver.run_command(["decomposePar", "-noFields"], log_filename="log.decomposePar.mesh")
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "8", "snappyHexMesh", "-parallel"],
        log_filename="log.snappyHexMesh",
    )
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "8", "createBaffles", "-parallel"],
        log_filename="log.createBaffles",
    )
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "8", "splitBaffles", "-parallel"],
        log_filename="log.splitBaffles",
    )
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "8", "renumberMesh", "-noFields", "-parallel"],
        log_filename="log.renumberMesh",
    )
    solver.run_command(
        [
            "mpirun", "--oversubscribe", "-np", "8",
            "createNonConformalCouples", "nonCouple1", "nonCouple2", "-parallel",
        ],
        log_filename="log.createNonConformalCouples",
    )
    solver.run_command(
        ["decomposePar", "-fields", "-copyZero"], log_filename="log.decomposePar.fields"
    )
    solver.run_command(
        [
            "mpirun", "--oversubscribe", "-np", "8", "foamRun",
            "-solver", "incompressibleFluid", "-parallel",
        ],
        log_filename="log.incompressibleFluid",
    )
    solver.run_command(
        ["reconstructPar", "-constant"], log_filename="log.reconstructPar"
    )


if __name__ == "__main__":
    main()
