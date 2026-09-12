"""OpenFOAM 13 fluid/annularThermalMixer through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/fluid/annularThermalMixer")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "fluid"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    mesh = Meshing(case_path, mesher="blockMesh")

    # Declarative generation: solver.system.write() already called in setup_case
    # Declarative generation: solver.constant.write() already called in setup_case
    geometry = REFERENCE / "constant" / "geometry"
    for source in geometry.iterdir():
        if source.is_file():
            mesh.mesher.import_reference_asset(
                source, case_path / "constant" / "geometry" / source.name
            )
    # Declarative generation: solver.fields_manager.write_initial_fields()

    # Declarative generation: no files to remove

    # Declarative generation: mesh.mesher.write() generates blockMeshDict
    solver.run_command(["surfaceFeatures"], log_filename="log.surfaceFeatures")
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["snappyHexMesh"], log_filename="log.snappyHexMesh")
    solver.run_command(["createBaffles"], log_filename="log.createBaffles")
    solver.run_command(["splitBaffles"], log_filename="log.splitBaffles")
    solver.run_command(
        ["createNonConformalCouples", "nonCouple1", "nonCouple2"],
        log_filename="log.createNonConformalCouples",
    )
    solver.run_simulation(nb_proc=1, log_filename="log.fluid")


if __name__ == "__main__":
    main()
