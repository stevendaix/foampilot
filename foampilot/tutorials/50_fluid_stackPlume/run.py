"""OpenFOAM 13 fluid/stackPlume through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/fluid/stackPlume")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "fluid"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    mesh = Meshing(case_path, mesher="blockMesh")
    for source in (REFERENCE / "system").rglob("*"):
        if source.is_file():
            relative_name = source.relative_to(REFERENCE / "system")
            solver.system.import_reference_file(source, filename=relative_name)
    for source in (REFERENCE / "constant").rglob("*"):
        if source.is_file():
            relative_name = source.relative_to(REFERENCE / "constant")
            solver.constant.import_reference_file(source, filename=relative_name)
    for source in (REFERENCE / "0").rglob("*"):
        if source.is_file():
            relative_name = source.relative_to(REFERENCE / "0")
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=relative_name
            )

    solver.constant.remove_files(["transportProperties", "turbulenceProperties"])
    solver.system.run_utility(
        "snappyHexMeshConfig",
        [
            "-bounds", "(-50 -200 0)(950 200 200)",
            "-nCells", "(40 16 8)",
            "-clearBoundary",
            "-xMinPatch", "inletWind patch",
            "-zMinPatch", "ground wall",
            "-defaultPatch", "outlet patch",
            "-insidePoint", "(0 0 50)",
            "-refinementDists", "((stack 5 5))",
            "-refinementBoxes", "((0 -10 0) (10 10 20) 2) ((-50 -80 0) (950 80 200) 1)",
        ],
        log_filename="log.snappyHexMeshConfig",
    )
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["snappyHexMesh", "-overwrite"], log_filename="log.snappyHexMesh")
    solver.system.run_utility("setAtmBoundaryLayer", log_filename="log.setAtmBoundaryLayer")
    solver.run_parallel(4, log_filename="log.fluid.parallel", force_decompose=True)


if __name__ == "__main__":
    main()
