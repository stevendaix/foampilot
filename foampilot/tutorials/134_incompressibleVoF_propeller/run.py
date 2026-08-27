"""OpenFOAM 13 incompressibleVoF/propeller through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleVoF/propeller")
MESH_REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleFluid/propeller")
GEOMETRY = Path("/opt/openfoam13/tutorials/resources/geometry")
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")


GEOMETRY_FILES = (
    "propeller-innerCylinder.obj.gz",
    "propeller-middleCylinder.obj.gz",
    "propeller-outerCylinder.obj.gz",
    "propeller.obj.gz",
)


def import_reference_case(solver: Solver, case_path: Path) -> None:
    """Import the OF13 reference dictionaries and fields without shell copying."""
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
        else:
            solver.import_reference_asset(source, case_path / relative)

    for filename in GEOMETRY_FILES:
        solver.import_reference_asset(
            GEOMETRY / filename,
            case_path / "constant" / "geometry" / filename,
        )


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "incompressibleVoF"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    # The VoF tutorial clones the already meshed incompressibleFluid case.
    # Reproduce the OF13 Allmesh sequence through FoamPilot commands.
    solver.run_command([str(OF13_BIN / "blockMesh")], log_filename="log.blockMesh")
    solver.run_command([str(OF13_BIN / "surfaceFeatures")], log_filename="log.surfaceFeatures")
    solver.run_command([str(OF13_BIN / "decomposePar"), "-noFields"], log_filename="log.decomposePar.mesh")
    solver.run_command(
        ["/usr/bin/mpirun", "--oversubscribe", "-np", "8", str(OF13_BIN / "snappyHexMesh"), "-parallel"],
        log_filename="log.snappyHexMesh",
    )
    solver.run_command(
        ["/usr/bin/mpirun", "--oversubscribe", "-np", "8", str(OF13_BIN / "createBaffles"), "-parallel"],
        log_filename="log.createBaffles",
    )
    solver.run_command(
        ["/usr/bin/mpirun", "--oversubscribe", "-np", "8", str(OF13_BIN / "splitBaffles"), "-parallel"],
        log_filename="log.splitBaffles",
    )
    solver.run_command(
        ["/usr/bin/mpirun", "--oversubscribe", "-np", "8", str(OF13_BIN / "renumberMesh"), "-noFields", "-parallel"],
        log_filename="log.renumberMesh",
    )
    solver.run_command(
        [
            "/usr/bin/mpirun", "--oversubscribe", "-np", "8",
            str(OF13_BIN / "createNonConformalCouples"), "nonCouple1", "nonCouple2", "-parallel",
        ],
        log_filename="log.createNonConformalCouples",
    )
    solver.run_command(
        [str(OF13_BIN / "decomposePar"), "-fields", "-copyZero"],
        log_filename="log.decomposePar.fields",
    )
    solver.run_command(
        [
            "/usr/bin/mpirun", "--oversubscribe", "-np", "8", str(OF13_BIN / "foamRun"),
            "-solver", "incompressibleVoF", "-parallel",
        ],
        log_filename="log.incompressibleVoF",
    )
    solver.run_command(
        [str(OF13_BIN / "reconstructPar"), "-constant"], log_filename="log.reconstructPar"
    )


if __name__ == "__main__":
    main()

# MESH_REFERENCE is intentionally retained as the OF13 cloneMesh reference.
assert MESH_REFERENCE.exists()
