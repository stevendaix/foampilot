"""OpenFOAM 13 incompressibleVoF/planingHullW3 through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleVoF/planingHullW3")


def import_reference_case(solver: Solver, destination: Path) -> None:
    for source in REFERENCE.rglob("*"):
        if not source.is_file() or source.name in {"Allrun", "Allclean", "Allmesh.1", "Allmesh.2"}:
            continue
        relative = source.relative_to(REFERENCE)
        if relative.parts and relative.parts[0] == "0":
            field_name = relative.relative_to("0")
            if field_name.suffix == ".orig":
                field_name = field_name.with_suffix("")
            solver.fields_manager.import_reference_field(source, destination, field_name=field_name)
        else:
            solver.import_reference_asset(source, destination / relative)


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "incompressibleVoF"
    solver.setup_case()
    import_reference_case(solver, case_path)
    transform = "translate=(-0.586 0 -0.156), Ry=-3.485, translate=(0.586 0 0.156)"
    solver.run_command(["surfaceTransformPoints", transform, "constant/geometry/w3_orig.stl", "constant/geometry/w3.stl"], log_filename="log.surfaceTransformPoints")
    solver.run_command(["surfaceFeatures"], log_filename="log.surfaceFeatures")
    solver.run_command(["blockMesh", "-dict", "system/blockMeshDict.1"], log_filename="log.blockMesh")
    solver.run_command(["refineMesh", "-dict", "system/refineMeshDict.1"], log_filename="log.refineMesh")
    solver.run_command(["snappyHexMesh", "-dict", "system/snappyHexMeshDict.1"], log_filename="log.snappyHexMesh")
    solver.run_command(["renumberMesh", "-noFields"], log_filename="log.renumberMesh")
    solver.run_command(["setFields"], log_filename="log.setFields")
    solver.run_command(["decomposePar"], log_filename="log.decomposePar")
    solver.run_command(["mpirun", "--oversubscribe", "-np", "16", "foamRun", "-solver", "incompressibleVoF", "-parallel"], log_filename="log.foamRun.parallel")
    solver.run_command(["reconstructPar"], log_filename="log.reconstructPar")


if __name__ == "__main__":
    main()
