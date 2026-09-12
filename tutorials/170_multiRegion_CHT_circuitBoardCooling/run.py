"""OpenFOAM 13 multiRegion/CHT/circuitBoardCooling via FoamPilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/multiRegion/CHT/circuitBoardCooling")
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")


def import_reference_case(solver: Solver, case_path: Path) -> None:
    """Import global, regional and initial data through FoamPilot."""
    for source in (REFERENCE / "0").rglob("*"):
        if source.is_file():
            relative = source.relative_to(REFERENCE / "0")
            region = relative.parts[0]
            field_name = source.stem if source.name.endswith(".orig") else source.name
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=f"{region}/{field_name}"
            )

    for root_name in ("constant", "system"):
        source_root = REFERENCE / root_name
        for source in source_root.rglob("*"):
            if source.is_file():
                solver.import_reference_asset(
                    source, case_path / source.relative_to(REFERENCE)
                )

    for source in (REFERENCE / "constant/geometry").iterdir():
        if source.is_file():
            solver.import_reference_asset(
                source, case_path / "constant/geometry" / source.name
            )

    include_source = REFERENCE / "include/wallPatchFields"
    solver.import_reference_asset(include_source, case_path / "include/wallPatchFields")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "foamMultiRun"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    solver.run_command(
        [str(OF13_BIN / "blockMesh"), "-region", "fluid"],
        log_filename="log.blockMesh.fluid",
    )
    solver.run_command(
        [str(OF13_BIN / "createZones"), "-region", "fluid"],
        log_filename="log.createZones.fluid",
    )
    solver.run_command(
        [
            str(OF13_BIN / "extrudeToRegionMesh"),
            "-region",
            "fluid",
            "-dict",
            "system/fluid/extrudeToRegionMeshDict.extrudeFromInternalFaces",
        ],
        log_filename="log.extrudeToRegionMesh.fluid",
    )
    solver.run_command(
        [
            str(OF13_BIN / "createBaffles"),
            "-region",
            "fluid",
            "-dict",
            "system/fluid/createBafflesDict.baffle1D",
        ],
        log_filename="log.createBaffles.baffle1D",
    )
    solver.run_command(
        [str(OF13_BIN / "foamMultiRun")],
        log_filename="log.foamMultiRun",
    )


if __name__ == "__main__":
    main()
