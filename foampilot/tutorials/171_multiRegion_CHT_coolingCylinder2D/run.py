"""OpenFOAM 13 multiRegion/CHT/coolingCylinder2D via FoamPilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/multiRegion/CHT/coolingCylinder2D")
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")


def import_reference_case(solver: Solver, case_path: Path) -> None:
    """Import global, regional dictionaries and initial fields through FoamPilot."""
    for source in (REFERENCE / "0").rglob("*"):
        if source.is_file():
            relative = source.relative_to(REFERENCE / "0")
            region = relative.parts[0]
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=f"{region}/{source.name}"
            )

    for root_name in ("constant", "system"):
        source_root = REFERENCE / root_name
        for source in source_root.rglob("*"):
            if source.is_file():
                solver.import_reference_asset(
                    source, case_path / source.relative_to(REFERENCE)
                )


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "foamMultiRun"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    solver.run_command([str(OF13_BIN / "blockMesh")], log_filename="log.blockMesh")
    solver.run_command(
        [str(OF13_BIN / "splitMeshRegions"), "-cellZones"],
        log_filename="log.splitMeshRegions",
    )
    solver.run_command(
        [str(OF13_BIN / "foamMultiRun")], log_filename="log.foamMultiRun"
    )


if __name__ == "__main__":
    main()
