"""OpenFOAM 13 incompressibleVoF/sloshingTank3D6DoF through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleVoF/sloshingTank3D6DoF")
MESH_REFERENCE = Path("/opt/openfoam13/tutorials/resources/blockMesh/sloshingTank3D")
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")


def import_reference_case(solver: Solver, case_path: Path) -> None:
    """Import OF13 dictionaries, sixDoF data and fields through FoamPilot APIs."""
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
        elif relative.parts and relative.parts[0] == "gen6DoF":
            # The official Allrun consumes the pre-generated constant/6DoF.dat;
            # the auxiliary C++ generator is not part of the run sequence.
            continue
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
        [str(OF13_BIN / "blockMesh"), "-dict", str(MESH_REFERENCE)],
        log_filename="log.blockMesh",
    )
    solver.run_command(
        [str(OF13_BIN / "setFields")], log_filename="log.setFields"
    )
    solver.run_command(
        [str(OF13_BIN / "foamRun"), "-solver", "incompressibleVoF"],
        log_filename="log.incompressibleVoF",
    )


if __name__ == "__main__":
    main()
