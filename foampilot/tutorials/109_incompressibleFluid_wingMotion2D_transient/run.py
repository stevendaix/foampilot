"""OpenFOAM 13 wingMotion/wingMotion2D_transient through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot.utilities.function import Functions

REFERENCE_ROOT = Path("/opt/openfoam13/tutorials/incompressibleFluid/wingMotion")
REFERENCE_MESH = REFERENCE_ROOT / "wingMotion_snappyHexMesh"
REFERENCE_STEADY = REFERENCE_ROOT / "wingMotion2D_steady"
REFERENCE_TRANSIENT = REFERENCE_ROOT / "wingMotion2D_transient"


def import_tree(solver: Solver, source_root: Path, destination_root: Path) -> None:
    for source in source_root.rglob("*"):
        if source.is_file():
            solver.import_reference_asset(source, destination_root / source.relative_to(source_root))


def main() -> None:
    case_path = Path.cwd()
    mesh_case = case_path / "wingMotion_snappyHexMesh"
    steady_case = case_path / "wingMotion2D_steady"
    transient_case = case_path / "wingMotion2D_transient"

    mesh_solver = Solver(mesh_case)
    steady_solver = Solver(steady_case)
    transient_solver = Solver(transient_case)

    import_tree(mesh_solver, REFERENCE_MESH, mesh_case)
    import_tree(steady_solver, REFERENCE_STEADY, steady_case)
    import_tree(transient_solver, REFERENCE_TRANSIENT, transient_case)

    # Reproduce the parent Allrun preparation and steady solution first.
    mesh_solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    mesh_solver.run_command(
        ["snappyHexMesh", "-overwrite"], log_filename="log.snappyHexMesh"
    )
    steady_solver.run_command(["extrudeMesh"], log_filename="log.extrudeMesh")
    steady_solver.run_command(["createPatch", "-overwrite"], log_filename="log.createPatch")
    steady_solver.run_command(["foamRun"], log_filename="log.incompressibleFluid.steady")

    # Map the steady mesh and solution into the moving transient case.
    transient_solver.copy_case_tree(steady_case, "constant/polyMesh", "constant/polyMesh")
    Functions.copy_reference_fields(
        source_case=steady_case,
        target_case=transient_case,
        fields=["U", "p", "k", "omega", "nut"],
        source_time="0",
    )
    transient_solver.run_command(
        ["mapFields", "../wingMotion2D_steady", "-sourceTime", "latestTime", "-consistent"],
        log_filename="log.mapFields",
    )
    # The OF13 Allrun installs pointDisplacement after mapFields so decomposePar
    # distributes it into every processor's initial-time directory.
    transient_solver.copy_case_tree(
        transient_case, "pointDisplacement", "0/pointDisplacement"
    )
    transient_solver.run_command(["decomposePar"], log_filename="log.decomposePar")
    transient_solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "4", "foamRun", "-parallel"],
        log_filename="log.incompressibleFluid.transient",
    )
    transient_solver.run_command(
        ["reconstructPar"], log_filename="log.reconstructPar"
    )


if __name__ == "__main__":
    main()
