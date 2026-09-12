"""OpenFOAM 13 compressibleMultiphaseVoF/damBreak4phaseLaminar."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/compressibleMultiphaseVoF/damBreak4phaseLaminar")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "compressibleMultiphaseVoF"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    for source in (REFERENCE / "system").iterdir():
        if source.is_file():
            solver.system.import_reference_file(source)
    for source in (REFERENCE / "constant").iterdir():
        if source.is_file():
            solver.constant.import_reference_file(source)
    for source in (REFERENCE / "0").iterdir():
        if source.is_file():
            name = source.name.removesuffix(".orig")
            solver.fields_manager.import_reference_field(source, case_path, name)

    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["setFields"], log_filename="log.setFields")
    solver.run_simulation(nb_proc=1, log_filename="log.compressibleMultiphaseVoF")


if __name__ == "__main__":
    main()
