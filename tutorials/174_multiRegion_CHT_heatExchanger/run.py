"""OpenFOAM 13 multiRegion/CHT/heatExchanger via FoamPilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/multiRegion/CHT/heatExchanger")
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")
NPROCS = 4


def import_reference_case(solver: Solver, case_path: Path) -> None:
    for source in (REFERENCE / "0").rglob("*"):
        if source.is_file():
            relative = source.relative_to(REFERENCE / "0")
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=str(relative)
            )
    for root in ("constant", "system"):
        for source in (REFERENCE / root).rglob("*"):
            if source.is_file():
                solver.import_reference_asset(
                    source, case_path / source.relative_to(REFERENCE)
                )


def command(solver: Solver, executable: str, *args: str, tag: str) -> None:
    solver.run_command(
        [str(OF13_BIN / executable), *args], log_filename=f"log.{tag}"
    )


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "foamMultiRun"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    command(solver, "blockMesh", "-region", "air", tag="blockMesh.air")
    command(solver, "blockMesh", "-region", "porous", tag="blockMesh.porous")
    command(
        solver,
        "createZones",
        "-region",
        "air",
        "-dict",
        "system/air/createZonesDict.1",
        tag="createZones.air.1",
    )
    command(
        solver,
        "createBaffles",
        "-region",
        "air",
        "-dict",
        "system/air/createBafflesDict",
        tag="createBaffles.air",
    )
    command(
        solver,
        "createZones",
        "-region",
        "air",
        "-dict",
        "system/air/createZonesDict.2",
        tag="createZones.air.2",
    )
    command(
        solver,
        "decomposePar",
        "-region",
        "air",
        tag="decomposePar.air",
    )
    command(
        solver,
        "decomposePar",
        "-region",
        "porous",
        tag="decomposePar.porous",
    )
    solver.run_command(
        [
            "/usr/bin/mpirun",
            "--oversubscribe",
            "-np",
            str(NPROCS),
            str(OF13_BIN / "foamMultiRun"),
            "-parallel",
        ],
        log_filename="log.foamMultiRun.parallel",
    )
    command(
        solver,
        "reconstructPar",
        "-latestTime",
        "-region",
        "air",
        tag="reconstructPar.air",
    )
    command(
        solver,
        "reconstructPar",
        "-latestTime",
        "-region",
        "porous",
        tag="reconstructPar.porous",
    )


if __name__ == "__main__":
    main()
