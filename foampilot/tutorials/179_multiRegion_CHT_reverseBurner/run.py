"""OpenFOAM 13 multiRegion/CHT/reverseBurner via FoamPilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/multiRegion/CHT/reverseBurner")
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")
OF13_TOOLS = Path("/opt/openfoam13/bin")
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


def parallel_command(solver: Solver, executable: str, *args: str, tag: str) -> None:
    solver.run_command(
        [
            "/usr/bin/mpirun",
            "--oversubscribe",
            "-np",
            str(NPROCS),
            str(OF13_BIN / executable),
            *args,
        ],
        log_filename=f"log.{tag}",
    )


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "foamMultiRun"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    command(solver, "blockMesh", tag="blockMesh")
    command(
        solver,
        "splitMeshRegions",
        "-cellZonesOnly",
        tag="splitMeshRegions.cellZonesOnly",
    )
    solver.run_command(
        [str(OF13_TOOLS / "paraFoam"), "-region", "gas", "-touch"],
        log_filename="log.paraFoam.gas.touch",
    )
    solver.run_command(
        [str(OF13_TOOLS / "paraFoam"), "-region", "solid", "-touch"],
        log_filename="log.paraFoam.solid.touch",
    )
    command(solver, "decomposePar", "-allRegions", tag="decomposePar.allRegions")
    parallel_command(
        solver,
        "setFields",
        "-region",
        "gas",
        tag="setFields.gas.parallel",
    )
    parallel_command(solver, "foamMultiRun", tag="foamMultiRun.parallel")
    command(
        solver,
        "reconstructPar",
        "-allRegions",
        "-newTimes",
        tag="reconstructPar.allRegions.newTimes",
    )


if __name__ == "__main__":
    main()
