"""OpenFOAM 13 multiRegion/CHT/coolingSphere via FoamPilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/multiRegion/CHT/coolingSphere")
TEMPLATES = REFERENCE / "templates"
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")
NPROCS = 4


def import_reference_case(solver: Solver, case_path: Path) -> None:
    """Import the parent case and its OF13 regional templates through FoamPilot."""
    for source in (REFERENCE / "constant").rglob("*"):
        if source.is_file():
            solver.import_reference_asset(source, case_path / source.relative_to(REFERENCE))
    for source in (REFERENCE / "system").rglob("*"):
        if source.is_file():
            solver.import_reference_asset(source, case_path / source.relative_to(REFERENCE))
    for source in (TEMPLATES / "0").rglob("*"):
        if source.is_file():
            relative = source.relative_to(TEMPLATES / "0")
            region = relative.parts[0]
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=f"{region}/{source.name}"
            )
    for source in (TEMPLATES / "system").rglob("*"):
        if source.is_file():
            solver.import_reference_asset(source, case_path / source.relative_to(TEMPLATES))
    for source in (TEMPLATES / "constant").rglob("*"):
        if source.is_file():
            solver.import_reference_asset(source, case_path / source.relative_to(TEMPLATES))
    for source in TEMPLATES.rglob("*"):
        if source.is_file() and source.relative_to(TEMPLATES).parts[0] != "0":
            solver.import_reference_asset(
                source, case_path / "templates" / source.relative_to(TEMPLATES)
            )


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
    solver.solver_name = "foamMultiRun"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    solver.run_command([str(OF13_BIN / "blockMesh")], log_filename="log.blockMesh")
    solver.run_command(
        [str(OF13_BIN / "createZones")], log_filename="log.createZones"
    )
    solver.run_command(
        [str(OF13_BIN / "transformPoints"), "scale=(0.01 0.01 0.01)"],
        log_filename="log.transformPoints",
    )
    solver.run_command(
        [
            str(OF13_BIN / "splitMeshRegions"),
            "-cellZones",
            "-defaultRegionName",
            "fluid",
        ],
        log_filename="log.splitMeshRegions",
    )
    solver.run_command(
        [str(OF13_BIN / "foamSetupCHT")], log_filename="log.foamSetupCHT"
    )
    solver.run_command(
        [
            str(OF13_BIN / "foamDictionary"),
            "-entry",
            "internalField",
            "-set",
            "uniform 348",
            "0/solid/T",
        ],
        log_filename="log.foamDictionary.solidT",
    )
    solver.run_command(
        [str(OF13_BIN / "decomposePar"), "-allRegions"],
        log_filename="log.decomposePar.allRegions",
    )
    solver.run_command(
        mpi_command("foamMultiRun"), log_filename="log.foamMultiRun.parallel"
    )
    solver.run_command(
        [str(OF13_BIN / "reconstructPar"), "-allRegions"],
        log_filename="log.reconstructPar.allRegions",
    )


if __name__ == "__main__":
    main()
