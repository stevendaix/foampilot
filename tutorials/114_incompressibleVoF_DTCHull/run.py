"""OpenFOAM 13 incompressibleVoF/DTCHull through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

BASE = Path("/opt/openfoam13/tutorials/incompressibleVoF")
REFERENCE = BASE / "DTCHull"
REFERENCE_MESH = BASE / "DTCHullMoving"
SURFACE = Path("/opt/openfoam13/tutorials/resources/geometry/DTC-scaled.stl.gz")


def import_tree(solver: Solver, source_root: Path, destination: Path) -> None:
    for source in source_root.rglob("*"):
        if source.is_file() and source.name not in {"Allrun", "Allclean", "Allmesh"}:
            solver.import_reference_asset(source, destination / source.relative_to(source_root))


def main() -> None:
    root = Path.cwd()
    mesh_case = root / "DTCHullMoving"
    case = root / "DTCHull"
    mesh_solver = Solver(mesh_case)
    solver = Solver(case)
    mesh_solver.solver_name = "incompressibleVoF"
    solver.solver_name = "incompressibleVoF"
    mesh_solver.setup_case()
    solver.setup_case()
    import_tree(mesh_solver, REFERENCE_MESH, mesh_case)
    import_tree(solver, REFERENCE, case)
    mesh_solver.import_reference_asset(SURFACE, mesh_case / "constant/geometry/DTC-scaled.stl.gz")

    mesh_solver.run_command(["surfaceFeatures"], log_filename="log.surfaceFeatures")
    mesh_solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    mesh_solver.run_command(["refineMesh"], log_filename="log.refineMesh")
    mesh_solver.run_command(["snappyHexMesh", "-overwrite"], log_filename="log.snappyHexMesh")
    mesh_solver.run_command(["renumberMesh", "-noFields"], log_filename="log.renumberMesh")

    solver.copy_case_tree(mesh_case, "constant/polyMesh", "constant/polyMesh")
    solver.run_command(["setFields"], log_filename="log.setFields")
    solver.run_command(["decomposePar"], log_filename="log.decomposePar")
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "8", "foamRun", "-parallel"],
        log_filename="log.incompressibleVoF",
    )
    solver.run_command(["reconstructPar"], log_filename="log.reconstructPar")


if __name__ == "__main__":
    main()
