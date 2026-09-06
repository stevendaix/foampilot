"""OpenFOAM 13 legacy dsmcFoam/wedge15Ma5 via FoamPilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path(
    "/opt/openfoam13/tutorials/legacy/lagrangian/dsmcFoam/wedge15Ma5"
)
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")
NPROCS = 4


def import_reference_case(solver: Solver, case_path: Path) -> None:
    """Import the complete OF13 DSMC wedge case through FoamPilot managers."""
    for source in (REFERENCE / "0").iterdir():
        if source.is_file():
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=source.name
            )
    for source in (REFERENCE / "constant").iterdir():
        if source.is_file():
            solver.constant.import_reference_file(source)
    for source in (REFERENCE / "system").iterdir():
        if source.is_file():
            solver.system.import_reference_file(source)


def mpi_command(executable: str, *args: str) -> list[str]:
    return [
        "/usr/bin/mpirun",
        "--oversubscribe",
        "-np",
        str(NPROCS),
        str(OF13_BIN / executable),
        *args,
        "-parallel",
    ]


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "dsmcFoam"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    solver.run_command(
        [str(OF13_BIN / "blockMesh")],
        log_filename="log.blockMesh",
    )
    solver.run_command(
        [str(OF13_BIN / "decomposePar")],
        log_filename="log.decomposePar",
    )
    solver.run_command(
        mpi_command("dsmcInitialise"),
        log_filename="log.dsmcInitialise.parallel",
    )
    solver.run_command(
        mpi_command("dsmcFoam"),
        log_filename="log.dsmcFoam.parallel",
    )
    solver.run_command(
        [str(OF13_BIN / "reconstructPar"), "-noLagrangian"],
        log_filename="log.reconstructPar",
    )


if __name__ == "__main__":
    main()
