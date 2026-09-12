"""OpenFOAM 13 compressibleVoF/depthCharge3D through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/compressibleVoF/depthCharge3D")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "compressibleVoF"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    # Declarative generation: solver.system.write() already called in setup_case
    # Declarative generation: solver.constant.write() already called in setup_case
    for source in (REFERENCE / "0").iterdir():
        if source.is_file():
            name = source.name.removesuffix(".orig")
            solver.fields_manager.import_reference_field(source, case_path, name)

    # Declarative generation: no files to remove

    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["setFields"], log_filename="log.setFields")
    solver.run_command(["decomposePar", "-force"], log_filename="log.decomposePar")
    solver.run_simulation(nb_proc=4, log_filename="log.foamRun.parallel")
    solver.run_command(["reconstructPar", "-latestTime"], log_filename="log.reconstructPar")


if __name__ == "__main__":
    main()
