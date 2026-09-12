"""OpenFOAM 13 fluid/helmholtzResonance through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/fluid/helmholtzResonance")


def prepare_variant(case_path: Path, variant: str) -> Solver:
    solver = Solver(case_path)
    solver.solver_name = "fluid"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()
    mesh = Meshing(case_path, mesher="blockMesh")

    # Declarative generation: solver.system.write() already called in setup_case
    for suffix, destination in (("Blocks", "blockMeshDict.caseBlocks"), ("Boundary", "blockMeshDict.caseBoundary")):
        solver.system.import_reference_file(
            REFERENCE / "system" / f"blockMeshDict.{variant}{suffix}",
            destination,
        )
    # Declarative generation: solver.constant.write() already called in setup_case
    # Declarative generation: solver.fields_manager.write_initial_fields()

    # Declarative generation: no files to remove
    # Declarative generation: mesh.mesher.write() generates blockMeshDict
    return solver


def run_variant(case_path: Path, variant: str) -> None:
    solver = prepare_variant(case_path, variant)
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["decomposePar"], log_filename="log.decomposePar")
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "4", "foamRun", "-solver", "fluid", "-parallel"],
        log_filename="log.fluid.parallel",
    )
    solver.run_command(["reconstructPar"], log_filename="log.reconstructPar")


def main() -> None:
    base = Path.cwd()
    run_variant(base / "resolved", "resolved")
    run_variant(base / "modelled", "modelled")


if __name__ == "__main__":
    main()
