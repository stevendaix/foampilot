"""OpenFOAM 13 wingMotion/wingMotion2D_steady through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE_ROOT = Path("/opt/openfoam13/tutorials/incompressibleFluid/wingMotion")
REFERENCE_MESH = REFERENCE_ROOT / "wingMotion_snappyHexMesh"
REFERENCE_STEADY = REFERENCE_ROOT / "wingMotion2D_steady"


def import_tree(solver: Solver, source_root: Path, destination_root: Path) -> None:
    """Copy a reference case tree through FoamPilot's file-import API."""
    for source in source_root.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        solver.import_reference_asset(source, destination)


def main() -> None:
    case_path = Path.cwd()
    mesh_case_path = case_path / "wingMotion_snappyHexMesh"
    steady_case_path = case_path / "wingMotion2D_steady"

    mesh_solver = Solver(mesh_case_path)
    steady_solver = Solver(steady_case_path)

    # Import the official OF13 source mesh case, including wing_5degrees.obj.
    import_tree(mesh_solver, REFERENCE_MESH, mesh_case_path)
    # Import the official steady 2D case, preserving all includes and fields.
    import_tree(steady_solver, REFERENCE_STEADY, steady_case_path)

    # Official Allrun sequence for wingMotion2D_steady:
    # blockMesh -> snappyHexMesh -> extrudeMesh -> createPatch -> foamRun.
    mesh_solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    mesh_solver.run_command(
        ["snappyHexMesh", "-overwrite"], log_filename="log.snappyHexMesh"
    )
    steady_solver.run_command(["extrudeMesh"], log_filename="log.extrudeMesh")
    steady_solver.run_command(["createPatch", "-overwrite"], log_filename="log.createPatch")
    steady_solver.run_command(["foamRun"], log_filename="log.incompressibleFluid")


if __name__ == "__main__":
    main()
