"""OpenFOAM 13 isothermalFluid/potentialFreeSurfaceMovingOscillatingBox through FoamPilot only."""
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

REFERENCE = Path(
    "/opt/openfoam13/tutorials/isothermalFluid/"
    "potentialFreeSurfaceMovingOscillatingBox"
)
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")


def import_reference_case(solver: Solver, case_path: Path) -> None:
    """Import the complete OF13 case tree through FoamPilot APIs."""
    for source in REFERENCE.rglob("*"):
        if not source.is_file() or source.name in {"Allrun", "Allclean"}:
            continue
        relative = source.relative_to(REFERENCE)
        if relative.parts and relative.parts[0] in {"0", "0.orig"}:
            field_name = relative.relative_to(relative.parts[0])
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=field_name
            )
        else:
            solver.import_reference_asset(source, case_path / relative)


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "isothermalFluid"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    solver.run_command([str(OF13_BIN / "blockMesh")], log_filename="log.blockMesh")
    solver.run_command(
        [str(OF13_BIN / "subsetMesh"), "-noFields"],
        log_filename="log.subsetMesh",
    )
    solver.run_command(
        [str(OF13_BIN / "foamRun")],
        log_filename="log.isothermalFluid",
    )


if __name__ == "__main__":
    main()
