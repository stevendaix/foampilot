"""OpenFOAM 13 incompressibleVoF/rotatingCube through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleVoF/rotatingCube")
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")


def import_reference_case(solver: Solver, case_path: Path) -> None:
    """Import all OF13 case inputs while omitting shell run scripts."""
    for source in REFERENCE.rglob("*"):
        if not source.is_file() or source.name in {"Allrun", "Allclean"}:
            continue
        relative = source.relative_to(REFERENCE)
        if relative.parts and relative.parts[0] == "0":
            field_name = relative.relative_to("0")
            if field_name.suffix == ".orig":
                field_name = field_name.with_suffix("")
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=field_name
            )
        else:
            solver.import_reference_asset(source, case_path / relative)


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "incompressibleVoF"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    solver.run_command(
        [str(OF13_BIN / "blockMesh")], log_filename="log.blockMesh"
    )
    solver.run_command(
        [
            str(OF13_BIN / "createNonConformalCouples"),
            "nonCoupleStationary", "nonCoupleRotating",
        ],
        log_filename="log.createNonConformalCouples",
    )
    solver.run_command(
        [str(OF13_BIN / "setFields")], log_filename="log.setFields"
    )
    solver.run_command(
        [str(OF13_BIN / "decomposePar"), "-cellProc"],
        log_filename="log.decomposePar.cellProc",
    )
    solver.run_command(
        [
            "/usr/bin/mpirun", "--oversubscribe", "-np", "8",
            str(OF13_BIN / "foamRun"), "-solver", "incompressibleVoF", "-parallel",
        ],
        log_filename="log.incompressibleVoF",
    )
    solver.run_command(
        [str(OF13_BIN / "reconstructPar"), "-cellProc"],
        log_filename="log.reconstructPar.cellProc",
    )


if __name__ == "__main__":
    main()
