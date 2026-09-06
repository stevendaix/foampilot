"""OpenFOAM 13 movingMesh/SnakeRiverCanyon via FoamPilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/movingMesh/SnakeRiverCanyon")
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")
NPROCS = 2


def import_reference_tree(solver: Solver, source_root: Path, destination: Path) -> None:
    """Import OF13 dictionaries, fields and geometry through FoamPilot."""
    for source in source_root.rglob("*"):
        if source.is_file():
            relative = source.relative_to(source_root)
            if relative.parts[:2] == ("constant", "geometry"):
                target = destination / "constant" / "geometry" / source.name
            else:
                target = destination / relative
            if relative.parts[:1] == ("0",):
                solver.fields_manager.import_reference_field(
                    source, destination, field_name=source.name
                )
            else:
                solver.import_reference_asset(source, target)


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "movingMesh"
    solver.transient = True
    solver.setup_case()
    import_reference_tree(solver, REFERENCE, case_path)

    solver.run_command([str(OF13_BIN / "blockMesh")], log_filename="log.blockMesh")
    solver.run_command(
        [str(OF13_BIN / "decomposePar")], log_filename="log.decomposePar"
    )
    solver.run_command(
        [
            "/usr/bin/mpirun",
            "--oversubscribe",
            "-np",
            str(NPROCS),
            str(OF13_BIN / "foamRun"),
            "-solver",
            "movingMesh",
            "-parallel",
        ],
        log_filename="log.foamRun.movingMesh.parallel",
    )
    solver.run_command(
        [str(OF13_BIN / "reconstructPar")], log_filename="log.reconstructPar"
    )


if __name__ == "__main__":
    main()
