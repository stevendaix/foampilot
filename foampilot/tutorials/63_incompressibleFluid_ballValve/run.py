"""OpenFOAM 13 incompressibleFluid/ballValve through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleFluid/ballValve")
MESH_REFERENCE = Path("/opt/openfoam13/tutorials/resources/blockMesh/ballValve")
GEOMETRY = Path("/opt/openfoam13/tutorials/resources/geometry/ballValve-torus.obj.gz")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "incompressibleFluid"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    mesh = Meshing(case_path, mesher="blockMesh")
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

    solver.constant.remove_files(["transportProperties", "turbulenceProperties"])
    solver.import_reference_asset(GEOMETRY, Path("constant/geometry/ballValve-torus.obj.gz"))
    mesh.mesher.import_reference_dict(MESH_REFERENCE)
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.system.run_utility("createZones", log_filename="log.createZones")
    solver.system.run_utility(
        "transformPoints", args=["-pointZone", "ball", "Rz=-45"], log_filename="log.transformPoints"
    )
    solver.system.run_utility(
        "createNonConformalCouples",
        args=["pipeNonCouple", "ballNonCouple"],
        log_filename="log.createNonConformalCouples",
    )
    solver.run_command(["decomposePar", "-cellProc"], log_filename="log.decomposePar")
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "8", "setFields", "-parallel"],
        log_filename="log.setFields",
    )
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "8", "foamRun", "-solver", "incompressibleFluid", "-parallel"],
        log_filename="log.incompressibleFluid.parallel",
    )
    solver.run_command(["reconstructPar", "-newTimes"], log_filename="log.reconstructPar")
    solver.system.run_utility("createGraphs", log_filename="log.createGraphs")


if __name__ == "__main__":
    main()
