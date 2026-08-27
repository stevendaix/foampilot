"""OpenFOAM 13 incompressibleVoF/parshallFlume through FoamPilot only."""
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleVoF/parshallFlume")


def import_reference_case(solver: Solver, destination: Path) -> None:
    for source in REFERENCE.rglob("*"):
        if not source.is_file() or source.name in {"Allrun", "Allclean"}:
            continue
        relative = source.relative_to(REFERENCE)
        if relative.parts and relative.parts[0] == "0":
            field_name = relative.relative_to("0")
            if field_name.suffix == ".orig":
                field_name = field_name.with_suffix("")
            solver.fields_manager.import_reference_field(source, destination, field_name=field_name)
        else:
            solver.import_reference_asset(source, destination / relative)


def number_of_subdomains(case_path: Path) -> int:
    text = (case_path / "system/decomposeParDict").read_text()
    match = re.search(r"numberOfSubdomains\s+(\d+)\s*;", text)
    if not match:
        raise RuntimeError("numberOfSubdomains is missing from the OF13 reference dictionary")
    return int(match.group(1))


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "incompressibleVoF"
    solver.setup_case()
    import_reference_case(solver, case_path)
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["createZones"], log_filename="log.createZones")
    solver.run_command(["setFields"], log_filename="log.setFields")
    solver.run_command(["decomposePar"], log_filename="log.decomposePar")
    nprocs = number_of_subdomains(case_path)
    solver.run_command(["mpirun", "--oversubscribe", "-np", str(nprocs), "foamRun", "-solver", "incompressibleVoF", "-parallel"], log_filename="log.foamRun.parallel")
    solver.run_command(["reconstructPar"], log_filename="log.reconstructPar")


if __name__ == "__main__":
    main()
