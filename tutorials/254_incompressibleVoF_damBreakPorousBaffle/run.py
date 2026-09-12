"""OpenFOAM 13 incompressibleVoF/damBreakPorousBaffle through FoamPilot only."""
from pathlib import Path
import os
import sys

OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")
OF13_LIB = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/lib")
OF13_TP_LIB = Path("/opt/openfoam13/thirdparty/platforms/linux64GccDPInt32/lib")
os.environ.update({
    "WM_PROJECT_DIR": "/opt/openfoam13",
    "FOAM_MPI": "openmpi-system",
    "WM_MPLIB": "SYSTEMOPENMPI",
    "FOAM_MODULES": str(OF13_LIB),
    "MPI_BUFFER_SIZE": "20000000",
    "PATH": f"{OF13_BIN}:/opt/openfoam13/bin:{os.environ.get('PATH', '')}",
    "LD_LIBRARY_PATH": f"{OF13_LIB / 'openmpi-system'}:{OF13_TP_LIB / 'openmpi-system'}:{OF13_TP_LIB}:{OF13_LIB}:{OF13_LIB / 'dummy'}:{os.environ.get('LD_LIBRARY_PATH', '')}",
})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

BASE = Path("/opt/openfoam13/tutorials/incompressibleVoF")
LAMINAR = BASE / "damBreakLaminar"
REFERENCE = BASE / "damBreakPorousBaffle"
DAM_BREAK = BASE / "damBreak"


def import_tree(solver: Solver, source_root: Path, destination: Path) -> None:
    """Import an OF13 case tree, materialising `.orig` dictionaries."""
    for source in source_root.rglob("*"):
        if not source.is_file() or source.name in {"Allrun", "Allclean"}:
            continue
        if source.name.endswith(".orig") and source.relative_to(source_root).parts[0] != "0":
            continue
        relative = source.relative_to(source_root)
        target_relative = Path(*[
            part[:-5] if part.endswith(".orig") else part
            for part in relative.parts
        ])
        if target_relative.parts and target_relative.parts[0] == "0":
            field_name = target_relative.relative_to("0")
            solver.fields_manager.import_reference_field(
                source, destination, field_name=field_name
            )
        else:
            solver.import_reference_asset(source, destination / target_relative)


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "incompressibleVoF"
    solver.setup_case()

    # Reproduce OF13 foamMergeCase damBreakLaminar -> damBreak.
    import_tree(solver, LAMINAR, case_path)
    solver.system.merge_reference_dictionary(
        "system/fvSchemes", DAM_BREAK / "system/fvSchemes.orig",
        blocks=["divSchemes"],
    )

    # damBreakPorousBaffle adds the porous baffle and limits alpha Courant.
    solver.import_reference_asset(
        REFERENCE / "system/createBafflesDict",
        case_path / "system/createBafflesDict",
    )
    solver.system.update_dictionary_entries(
        "system/controlDict", {"maxAlphaCo": "0.1"}
    )
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["setFields"], log_filename="log.setFields")
    solver.run_command(["createBaffles"], log_filename="log.createBaffles")
    solver.run_simulation(log_filename="log.incompressibleVoF")


if __name__ == "__main__":
    main()
