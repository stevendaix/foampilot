"""OpenFOAM 13 incompressibleFluid/simpleRushtonMRF through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleFluid/simpleRushtonMRF")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "incompressibleFluid"
    solver.transient = False
    solver.turbulence_model = "kOmegaSST"
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    for source in (REFERENCE / "system").rglob("*"):
        if source.is_file() and source.name not in {"blockMeshDict.orig", "createZonesDict.orig", "mirrorMeshDict.orig"}:
            solver.system.import_reference_file(
                source, filename=source.relative_to(REFERENCE / "system")
            )
    for source in (REFERENCE / "constant").rglob("*"):
        if source.is_file():
            solver.constant.import_reference_file(
                source, filename=source.relative_to(REFERENCE / "constant")
            )
    for source in (REFERENCE / "0").rglob("*"):
        if source.is_file():
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=source.relative_to(REFERENCE / "0")
            )

    system = solver.system
    system.import_reference_file(
        REFERENCE / "system" / "blockMeshDict.orig", filename="blockMeshDict"
    )
    system.import_reference_file(
        REFERENCE / "system" / "mirrorMeshDict.orig", filename="mirrorMeshDict"
    )
    system.import_reference_file(
        REFERENCE / "system" / "createZonesDict.orig", filename="createZonesDict"
    )

    Meshing(case_path, mesher="blockMesh").mesher.import_reference_dict(
        REFERENCE / "system" / "blockMeshDict.orig"
    )
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["mirrorMesh"], log_filename="log.mirrorMesh")
    solver.run_command(
        [
            "foamDictionary", "-set", "pointAndNormalDict/normal=(1 0 0)",
            "system/mirrorMeshDict",
        ],
        log_filename="log.foamDictionary",
    )
    solver.run_command(["mirrorMesh"], log_filename="log.mirrorMesh.second")
    solver.run_command(["createZones"], log_filename="log.createZones")
    solver.run_command(
        ["createBaffles", "-dict", "createBafflesDict.stirrer"],
        log_filename="log.createBaffles.stirrer",
    )
    solver.run_command(
        ["createBaffles", "-dict", "createBafflesDict.baffles"],
        log_filename="log.createBaffles.baffles",
    )
    solver.run_simulation(log_filename="log.incompressibleFluid")


if __name__ == "__main__":
    main()
