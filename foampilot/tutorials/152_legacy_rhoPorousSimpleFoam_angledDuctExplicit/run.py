"""OpenFOAM 13 legacy rhoPorousSimpleFoam/angledDuctExplicit via FoamPilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path(
    "/opt/openfoam13/tutorials/legacy/compressible/rhoPorousSimpleFoam/angledDuctExplicit"
)
MESH_REFERENCE = Path("/opt/openfoam13/tutorials/resources/blockMesh/angledDuct")
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")


def import_reference_case(solver: Solver, case_path: Path) -> None:
    """Import all OF13 fields and dictionaries through dedicated FoamPilot APIs."""
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

    solver.system.import_reference_file(MESH_REFERENCE, "angledDuct")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "rhoPorousSimpleFoam"
    solver.transient = False
    solver.setup_case()
    import_reference_case(solver, case_path)

    solver.run_command(
        [str(OF13_BIN / "blockMesh"), "-dict", "system/angledDuct"],
        log_filename="log.blockMesh",
    )
    solver.run_command(
        [str(OF13_BIN / "rhoPorousSimpleFoam")],
        log_filename="log.rhoPorousSimpleFoam",
    )


if __name__ == "__main__":
    main()
