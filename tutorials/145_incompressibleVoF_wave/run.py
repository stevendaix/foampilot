"""OpenFOAM 13 incompressibleVoF/wave through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleVoF/wave")
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")


def import_reference_case(solver: Solver, case_path: Path) -> None:
    """Import the complete OF13 wave case through FoamPilot APIs."""
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

    solver.run_command([str(OF13_BIN / "blockMesh")], log_filename="log.blockMesh")
    solver.run_command([str(OF13_BIN / "extrudeMesh")], log_filename="log.extrudeMesh")
    solver.run_command([str(OF13_BIN / "refineMesh")], log_filename="log.refineMesh")
    solver.run_command([str(OF13_BIN / "setWaves")], log_filename="log.setWaves")
    solver.run_command([str(OF13_BIN / "decomposePar")], log_filename="log.decomposePar")
    solver.run_command(
        [
            "mpirun",
            "--oversubscribe",
            "-np",
            "6",
            str(OF13_BIN / "foamRun"),
            "-solver",
            "incompressibleVoF",
            "-parallel",
        ],
        log_filename="log.incompressibleVoF",
    )
    solver.run_command([str(OF13_BIN / "reconstructPar")], log_filename="log.reconstructPar")


if __name__ == "__main__":
    main()
