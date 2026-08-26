"""OpenFOAM 13 incompressibleFluid/pitzDailySteadyMappedToPart through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path(
    "/opt/openfoam13/tutorials/incompressibleFluid/pitzDailySteadyMappedToPart"
)
SOURCE_REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleFluid/pitzDailySteady")
MESH_REFERENCE = Path("/opt/openfoam13/tutorials/resources/blockMesh/pitzDaily")


def import_system_tree(solver: Solver, root: Path) -> None:
    for source in root.rglob("*"):
        if source.is_file():
            solver.system.import_reference_file(source, filename=source.relative_to(root))


def import_constant_tree(solver: Solver, root: Path) -> None:
    for source in root.rglob("*"):
        if source.is_file():
            solver.constant.import_reference_file(source, filename=source.relative_to(root))


def prepare_source(source_case: Path) -> None:
    source_solver = Solver(source_case)
    source_solver.solver_name = "incompressibleFluid"
    source_solver.transient = True
    source_solver.setup_case()
    source_solver.system.write()
    source_solver.constant.write()
    import_system_tree(source_solver, SOURCE_REFERENCE / "system")
    import_constant_tree(source_solver, SOURCE_REFERENCE / "constant")
    for source in (SOURCE_REFERENCE / "0").rglob("*"):
        if source.is_file():
            source_solver.fields_manager.import_reference_field(
                source, source_case, field_name=source.relative_to(SOURCE_REFERENCE / "0")
            )
    source_solver.system.import_reference_file(
        REFERENCE / "system" / "decomposeParDict", filename="decomposeParDict"
    )
    source_solver.constant.remove_files(["transportProperties", "turbulenceProperties"])
    Meshing(source_case, mesher="blockMesh").mesher.import_reference_dict(MESH_REFERENCE)
    source_solver.run_command(["blockMesh"], log_filename="log.blockMesh.source")
    source_solver.run_command(["decomposePar"], log_filename="log.decomposePar.source")
    source_solver.run_command(
        [
            "mpirun", "--oversubscribe", "-np", "4", "foamRun",
            "-solver", "incompressibleFluid", "-parallel", "-noFunctionObjects",
        ],
        log_filename="log.incompressibleFluid.source",
    )


def main() -> None:
    case_path = Path.cwd()
    source_case = case_path / "pitzDailySteady"
    prepare_source(source_case)

    solver = Solver(case_path)
    solver.solver_name = "incompressibleFluid"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    # The complete source dictionaries are the base of the variant case.
    import_system_tree(solver, SOURCE_REFERENCE / "system")
    import_constant_tree(solver, SOURCE_REFERENCE / "constant")
    # Keep only the variant dictionaries from this tutorial in addition to them.
    for filename in ("decomposeParDict", "mapFieldsDict"):
        solver.system.import_reference_file(
            REFERENCE / "system" / filename, filename=filename
        )
    # The .orig files are the target-specific partial dictionaries.
    for source in (REFERENCE / "0").rglob("*.orig"):
        solver.import_reference_asset(source, case_path / "0" / source.name)
        field = source.name.removesuffix(".orig")
        solver.fields_manager.import_reference_field(
            SOURCE_REFERENCE / "0" / field, case_path, field_name=field
        )
        solver.run_command(
            ["foamDictionary", "-dict", "-merge", f"0/{field}.orig", f"0/{field}"],
            log_filename=f"log.foamDictionary.merge.{field}",
        )

    solver.constant.remove_files(["transportProperties", "turbulenceProperties"])
    solver.system.update_dictionary_entries(
        "system/fvSolution", {"relaxationFactors/equations": '{ ".*" 0.1; }'}
    )
    Meshing(case_path, mesher="blockMesh").mesher.import_reference_dict(
        REFERENCE / "system" / "blockMeshDict.orig",
        destination="system/blockMeshDict",
    )
    solver.run_command(["blockMesh"], log_filename="log.blockMesh.target")
    solver.run_command(["decomposePar"], log_filename="log.decomposePar.target")
    solver.run_command(
        [
            "mapFieldsPar", str(source_case), "-sourceTime", "latestTime",
            "-fields", "(epsilon k nut p U)",
        ],
        log_filename="log.mapFieldsPar",
    )
    solver.run_command(
        [
            "mpirun", "--oversubscribe", "-np", "4", "foamRun",
            "-solver", "incompressibleFluid", "-parallel", "-noFunctionObjects",
        ],
        log_filename="log.incompressibleFluid.target",
    )
    solver.run_command(["reconstructPar", "-withZero"], log_filename="log.reconstructPar")


if __name__ == "__main__":
    main()
