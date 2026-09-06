"""OpenFOAM 13 incompressibleVoF/mixerVessel through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleVoF/mixerVessel")
RESOURCES = Path("/opt/openfoam13/tutorials/resources/geometry")

def import_reference_case(solver: Solver, destination: Path) -> None:
    for source in REFERENCE.rglob("*"):
        if not source.is_file() or source.name in {"Allrun", "Allmesh", "Allclean"}:
            continue
        relative = source.relative_to(REFERENCE)
        if relative.parts and relative.parts[0] == "0":
            field_name = relative.relative_to("0")
            if field_name.suffix == ".orig":
                field_name = field_name.with_suffix("")
            solver.fields_manager.import_reference_field(source, destination, field_name=field_name)
        else:
            solver.import_reference_asset(source, destination / relative)
    for source in RESOURCES.glob("mixerVessel*.stl.gz"):
        solver.import_reference_asset(source, destination / "constant/geometry" / source.name)

def mpi(solver: Solver, utility: str, *args: str, log: str) -> None:
    solver.run_command(["mpirun", "--oversubscribe", "-np", "8", utility, *args, "-parallel"], log_filename=log)

def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "incompressibleVoF"
    solver.setup_case()
    import_reference_case(solver, case_path)
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["surfaceFeatures"], log_filename="log.surfaceFeatures")
    solver.run_command(["decomposePar", "-noFields"], log_filename="log.decomposePar.mesh")
    mpi(solver, "snappyHexMesh", log="log.snappyHexMesh.parallel")
    mpi(solver, "createBaffles", log="log.createBaffles.parallel")
    mpi(solver, "splitBaffles", log="log.splitBaffles.parallel")
    mpi(solver, "createNonConformalCouples", "nonCouple1", "nonCouple2", log="log.createNonConformalCouples.parallel")
    solver.run_command(["decomposePar", "-fields", "-copyZero"], log_filename="log.decomposePar.fields")
    mpi(solver, "setFields", log="log.setFields.parallel")
    mpi(solver, "foamRun", "-solver", "incompressibleVoF", log="log.foamRun.parallel")
    solver.run_command(["reconstructPar", "-constant"], log_filename="log.reconstructPar")

if __name__ == "__main__":
    main()
