"""OpenFOAM 13 incompressibleFluid/movingCone through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleFluid/movingCone")


def active_name(source: Path) -> Path:
    return Path(source.name[:-5] if source.name.endswith(".orig") else source.name)


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "incompressibleFluid"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    mesh = Meshing(case_path, mesher="blockMesh")
    for source in (REFERENCE / "system").rglob("*"):
        if source.is_file():
            rel = source.relative_to(REFERENCE / "system")
            solver.system.import_reference_file(
                source, filename=rel.parent / active_name(source)
            )
    for source in (REFERENCE / "constant").rglob("*"):
        if source.is_file():
            rel = source.relative_to(REFERENCE / "constant")
            solver.constant.import_reference_file(
                source, filename=rel.parent / active_name(source)
            )
    for source in (REFERENCE / "0").rglob("*"):
        if source.is_file():
            rel = source.relative_to(REFERENCE / "0")
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=rel.parent / active_name(source)
            )

    solver.constant.remove_files(["transportProperties", "turbulenceProperties"])
    mesh.mesher.import_reference_dict(REFERENCE / "system" / "blockMeshDict")
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    for map_time in ("0.0015", "0.003"):
        source = REFERENCE / "system" / "meshes" / map_time / "blockMeshDict"
        solver.system.import_reference_file(
            source, filename=Path("meshes") / map_time / "blockMeshDict"
        )
        solver.run_command(
            ["blockMesh", "-mesh", map_time],
            log_filename=f"log.blockMesh.{map_time}"
        )
    solver.run_simulation(nb_proc=1, log_filename="log.incompressibleFluid")


if __name__ == "__main__":
    main()
