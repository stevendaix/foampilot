"""OpenFOAM 13 incompressibleFluid/motorBikeSteady through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleFluid/motorBikeSteady")
MOTORBIKE = Path("/opt/openfoam13/tutorials/resources/geometry/motorBike.obj.gz")


def active_name(source: Path) -> Path:
    return Path(source.name[:-5] if source.name.endswith(".orig") else source.name)


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "incompressibleFluid"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    solver.import_reference_asset(MOTORBIKE, "constant/geometry/motorBike.obj.gz")
    mesh = Meshing(case_path, mesher="blockMesh")
    for source in (REFERENCE / "system").rglob("*"):
        if source.is_file():
            rel = source.relative_to(REFERENCE / "system")
            solver.system.import_reference_file(source, filename=rel)
    for source in (REFERENCE / "constant").rglob("*"):
        if source.is_file():
            rel = source.relative_to(REFERENCE / "constant")
            solver.constant.import_reference_file(source, filename=rel)
    for source in (REFERENCE / "0").rglob("*"):
        if source.is_file():
            rel = source.relative_to(REFERENCE / "0")
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=rel.parent / active_name(source)
            )

    solver.constant.remove_files(["transportProperties", "turbulenceProperties"])
    block = Meshing(case_path, mesher="blockMesh")
    block.mesher.import_reference_dict(REFERENCE / "system" / "blockMeshDict")
    solver.run_command(["surfaceFeatures"], log_filename="log.surfaceFeatures")
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["decomposePar", "-copyZero"], log_filename="log.decomposePar")
    mpi = ["mpirun", "--oversubscribe", "-np", "6"]
    solver.run_command(mpi + ["snappyHexMesh", "-parallel"], log_filename="log.snappyHexMesh")
    solver.run_command(mpi + ["patchSummary", "-parallel"], log_filename="log.patchSummary")
    solver.run_command(mpi + ["potentialFoam", "-parallel"], log_filename="log.potentialFoam")
    solver.run_command(mpi + ["foamRun", "-solver", "incompressibleFluid", "-parallel"], log_filename="log.incompressibleFluid")
    solver.system.run_utility("reconstructPar", args=["-latestTime"], log_filename="log.reconstructPar")


if __name__ == "__main__":
    main()
