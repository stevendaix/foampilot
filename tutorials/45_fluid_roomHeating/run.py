"""OpenFOAM 13 fluid/roomHeating through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/fluid/roomHeating")


def import_reference(solver: Solver, case_path: Path) -> None:
    for source_name, target_name in (
        ("controlDict.orig", "controlDict"),
        ("fvSchemes.orig", "fvSchemes"),
        ("fvSolution.orig", "fvSolution"),
    ):
        solver.system.import_reference_file(
            REFERENCE / "system" / source_name,
            filename=target_name,
        )
    for source in (REFERENCE / "system").iterdir():
        if source.is_file() and source.name not in {"controlDict.orig", "fvSchemes.orig", "fvSolution.orig"}:
            solver.system.import_reference_file(source)
    for source in (REFERENCE / "constant").iterdir():
        if source.is_file():
            solver.constant.import_reference_file(source)
    for source in (REFERENCE / "0").iterdir():
        if source.is_file():
            solver.fields_manager.import_reference_field(source, case_path)


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "fluid"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()
    import_reference(solver, case_path)

    solver.constant.remove_files(["transportProperties", "turbulenceProperties"])
    mesh = Meshing(case_path, mesher="blockMesh")
    mesh.mesher.import_reference_dict(REFERENCE / "system" / "blockMeshDict")
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.system.run_utility("createZones", log_filename="log.createZones")
    solver.system.ensure_decomposeParDict(4)
    solver.run_parallel(4, log_filename="log.steady", force_decompose=False)

    solver.system.update_dictionary_entries(
        "system/controlDict",
        {"endTime": "6000", "deltaT": "0.01", "adjustTimeStep": "yes"},
    )
    solver.system.update_dictionary_entries(
        "system/fvSchemes",
        {"ddtSchemes/default": "Euler"},
    )
    solver.run_parallel(4, log_filename="log.transient", force_decompose=True)


if __name__ == "__main__":
    main()
