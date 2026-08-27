"""OpenFOAM 13 multiRegion/CHT/notchedRoller via FoamPilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/multiRegion/CHT/notchedRoller")
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


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "foamMultiRun"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    command(solver, "blockMesh", tag="blockMesh")
    command(solver, "createBaffles", tag="createBaffles")
    command(solver, "splitBaffles", tag="splitBaffles")
    command(
        solver,
        "splitMeshRegions",
        "-cellZonesOnly",
        tag="splitMeshRegions.cellZonesOnly",
    )
    for asset in (
        "0/cellToRegion",
        "0/fluid/cellToRegion",
        "0/solid/cellToRegion",
        "0/roller/cellToRegion",
        "constant/cellToRegion",
    ):
        solver.remove_case_asset(asset)
    command(
        solver,
        "createZones",
        "-dict",
        "system/createZonesDict",
        "-region",
        "fluid",
        tag="createZones.fluid",
    )
    command(solver, "createNonConformalCouples", tag="createNonConformalCouples")
    command(
        solver,
        "decomposePar",
        "-allRegions",
        "-copyZero",
        "-cellProc",
        tag="decomposePar.allRegions",
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
    command(solver, "reconstructPar", "-allRegions", tag="reconstructPar.allRegions")
    solver.run_command(
        [str(OF13_TOOLS / "paraFoam"), "-touchAll"],
        log_filename="log.paraFoam.touchAll",
    )


if __name__ == "__main__":
    main()
