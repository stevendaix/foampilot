"""OpenFOAM 13 fluid/aerofoilNACA0012 through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/fluid/aerofoilNACA0012")
BLOCK_MESH_ASSET = Path("/opt/openfoam13/tutorials/resources/geometry/NACA0012.obj.gz")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "fluid"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    # Declarative generation: solver.system.write() already called in setup_case
    # Declarative generation: solver.constant.write() already called in setup_case
    # Declarative generation: solver.fields_manager.write_initial_fields()

    # Declarative generation: no files to remove

    mesh = Meshing(case_path, mesher="blockMesh")
    mesh.mesher.import_reference_asset(
        BLOCK_MESH_ASSET,
        case_path / "constant" / "geometry" / "NACA0012.obj",
    )
    # Declarative generation: mesh.mesher.write() generates blockMeshDict

    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["transformPoints", "scale=(1 0 1)"], log_filename="log.transformPoints")
    solver.run_command(["extrudeMesh"], log_filename="log.extrudeMesh")
    solver.run_simulation(nb_proc=1, log_filename="log.fluid")


if __name__ == "__main__":
    main()
