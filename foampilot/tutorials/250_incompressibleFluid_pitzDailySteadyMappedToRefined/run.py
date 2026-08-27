"""OpenFOAM 13 incompressibleFluid/pitzDailySteadyMappedToRefined through FoamPilot only."""
from pathlib import Path
import os
import sys

OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")
OF13_LIB = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/lib")
OF13_TP_LIB = Path("/opt/openfoam13/thirdparty/platforms/linux64GccDPInt32/lib")
os.environ.update({
    "WM_PROJECT_DIR": "/opt/openfoam13",
    "MPI_BUFFER_SIZE": "20000000",
    "FOAM_MPI": "openmpi-system",
    "WM_MPLIB": "SYSTEMOPENMPI",
    "PATH": f"{OF13_BIN}:/opt/openfoam13/bin:{os.environ.get('PATH', '')}",
    "LD_LIBRARY_PATH": f"{OF13_LIB / 'openmpi-system'}:{OF13_TP_LIB / 'openmpi-system'}:{OF13_TP_LIB}:{OF13_LIB}:{OF13_LIB / 'dummy'}:{os.environ.get('LD_LIBRARY_PATH', '')}",
})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path(
    "/opt/openfoam13/tutorials/incompressibleFluid/pitzDailySteadyMappedToRefined"
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
        MESH_REFERENCE,
        destination="system/blockMeshDict",
    )
    mesh_path = case_path / "system/blockMeshDict"
    for original, refined in {
        "(18 27 1)": "(22 33 1)",
        "(18 30 1)": "(22 37 1)",
        "(180 27 1)": "(225 33 1)",
        "(180 30 1)": "(225 37 1)",
        "(25 27 1)": "(31 33 1)",
        "(25 30 1)": "(31 37 1)",
    }.items():
        solver.system.replace_file_text(mesh_path, original, refined)
    for original in ("(0 0 0)", "(0 0 0.01)", "(0 0 0.1)"):
        solver.system.replace_file_text(mesh_path, original, original.replace("(0 0", "(-1.2 0"))
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
