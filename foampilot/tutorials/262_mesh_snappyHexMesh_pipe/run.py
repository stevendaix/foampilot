"""OpenFOAM 13 mesh/snappyHexMesh/pipe through FoamPilot only."""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/mesh/snappyHexMesh/pipe")
RESOURCE_GEOMETRY = Path("/opt/openfoam13/tutorials/resources/geometry")
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")
OF13_LIB = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/lib")
OF13_MPI_LIB = OF13_LIB / "openmpi-system"
OF13_DUMMY_LIB = OF13_LIB / "dummy"
OF13_TP_LIB = Path("/opt/openfoam13/thirdparty/platforms/linux64GccDPInt32/lib")
OF13_TP_MPI_LIB = OF13_TP_LIB / "openmpi-system"
OF13_ENV = {
    "WM_PROJECT": "OpenFOAM",
    "WM_PROJECT_VERSION": "13",
    "WM_PROJECT_DIR": "/opt/openfoam13",
    "WM_OPTIONS": "linux64GccDPInt32Opt",
    "FOAM_APPBIN": str(OF13_BIN),
    "FOAM_LIBBIN": str(OF13_LIB),
    "FOAM_MPI": "openmpi-system",
    "FOAM_MPI_LIBBIN": str(OF13_MPI_LIB),
    "FOAM_EXT_LIBBIN": str(OF13_TP_LIB),
    "WM_MPLIB": "SYSTEMOPENMPI",
    "MPI_ARCH_PATH": "/usr/lib/x86_64-linux-gnu/openmpi",
    "PATH": f"{OF13_BIN}:/opt/openfoam13/bin:/opt/openfoam13/wmake:{os.environ.get('PATH', '')}",
    "LD_LIBRARY_PATH": f"{OF13_MPI_LIB}:{OF13_TP_MPI_LIB}:{OF13_DUMMY_LIB}:{OF13_LIB}:{OF13_TP_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}",
}


def import_reference_case(solver: Solver, case_path: Path) -> None:
    for root in ("constant", "system"):
        for source in (REFERENCE / root).rglob("*"):
            if source.is_file() and source.name != "README":
                solver.import_reference_asset(
                    source, case_path / source.relative_to(REFERENCE)
                )
    for name in ("pipe.obj.gz", "pipeWall.obj.gz"):
        solver.import_reference_asset(
            RESOURCE_GEOMETRY / name, case_path / "constant/geometry" / name
        )


def run_of13(solver: Solver, executable: str, *args: str, tag: str) -> None:
    solver.run_command(
        [str(OF13_BIN / executable), *args],
        log_filename=f"log.{tag}",
        environment=OF13_ENV,
    )


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "snappyHexMesh"
    solver.setup_case()
    import_reference_case(solver, case_path)
    run_of13(solver, "surfaceFeatures", tag="surfaceFeatures")
    run_of13(solver, "blockMesh", tag="blockMesh")
    run_of13(solver, "snappyHexMesh", tag="snappyHexMesh")


if __name__ == "__main__":
    main()
