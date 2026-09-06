"""OpenFOAM 13 multiphaseEuler/wallBoilingPolydisperse via FoamPilot."""
from pathlib import Path
import os
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/multiphaseEuler/wallBoilingPolydisperse")
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")
OF13_TOOLS = Path("/opt/openfoam13/bin")
OF13_LIB = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/lib")
OF13_MPI_LIB = OF13_LIB / "openmpi-system"
OF13_DUMMY_LIB = OF13_LIB / "dummy"
OF13_TP_LIB = Path("/opt/openfoam13/thirdparty/platforms/linux64GccDPInt32/lib")
OF13_TP_MPI_LIB = OF13_TP_LIB / "openmpi-system"
OF13_PV_LIB = OF13_LIB / "paraview-5.11"
OF13_ENV = {
    "WM_PROJECT_DIR": "/opt/openfoam13",
    "FOAM_APPBIN": str(OF13_BIN),
    "FOAM_LIBBIN": str(OF13_LIB),
    "FOAM_MPI": "openmpi-system",
    "FOAM_MPI_LIBBIN": str(OF13_MPI_LIB),
    "FOAM_EXT_LIBBIN": str(OF13_TP_LIB),
    "MPI_ARCH_PATH": "/usr/lib/x86_64-linux-gnu/openmpi",
    "WM_MPLIB": "SYSTEMOPENMPI",
    "SCOTCH_TYPE": "ThirdParty",
    "PATH": f"{OF13_BIN}:{OF13_TOOLS}:{os.environ.get('PATH', '')}",
    "LD_LIBRARY_PATH": f"{OF13_MPI_LIB}:{OF13_TP_MPI_LIB}:{OF13_PV_LIB}:/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/openmpi/lib:{OF13_TP_LIB}:{OF13_DUMMY_LIB}:{OF13_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}",
}


def import_reference_case(solver: Solver, case_path: Path) -> None:
    for source in (REFERENCE / "0").rglob("*"):
        if source.is_file():
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=str(source.relative_to(REFERENCE / "0"))
            )
    for root in ("constant", "system", "validation"):
        for source in (REFERENCE / root).rglob("*"):
            if source.is_file():
                solver.import_reference_asset(source, case_path / source.relative_to(REFERENCE))
    solver.import_reference_asset(
        Path("/opt/openfoam13/tutorials/resources/thermoData/wallBoiling-liquid.gz"),
        case_path / "constant/wallBoiling-liquid.gz",
    )
    solver.import_reference_asset(
        Path("/opt/openfoam13/tutorials/resources/thermoData/wallBoiling-vapour.gz"),
        case_path / "constant/wallBoiling-vapour.gz",
    )
    solver.import_reference_asset(
        Path("/opt/openfoam13/tutorials/resources/thermoData/wallBoiling-saturation.csv"),
        case_path / "constant/wallBoiling-saturation.csv",
    )


def run_of13(solver: Solver, executable: str, *args: str, tag: str) -> None:
    command_line = shlex.join([str(OF13_BIN / executable), *args])
    solver.run_command(
        ["/bin/bash", "-lc", f"source /opt/openfoam13/etc/bashrc >/dev/null 2>&1 && exec {command_line}"],
        log_filename=f"log.{tag}", environment=OF13_ENV,
    )


def run_parallel(solver: Solver, executable: str, *args: str, tag: str) -> None:
    command_line = shlex.join(["/usr/bin/mpirun", "--oversubscribe", "-np", "4", str(OF13_BIN / executable), "-parallel", *args])
    solver.run_command(
        ["/bin/bash", "-lc", f"source /opt/openfoam13/etc/bashrc >/dev/null 2>&1 && exec {command_line}"],
        log_filename=f"log.{tag}", environment=OF13_ENV,
    )


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "multiphaseEuler"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)
    run_of13(solver, "blockMesh", tag="blockMesh")
    run_of13(solver, "extrudeMesh", tag="extrudeMesh")
    run_of13(solver, "decomposePar", tag="decomposePar")
    run_parallel(solver, "foamRun", tag="foamRun.parallel")
    run_of13(solver, "reconstructPar", "-latestTime", tag="reconstructPar.latestTime")
    run_of13(solver, "foamPostProcess", "-latestTime", "-func", "graphCell(name=graph,start=(3.4901 0 0),end=(3.4901 0.0096 0),fields=(alpha.gas T.liquid T.gas d.gas))", tag="graphCell.latestTime")
    run_of13(solver, "foamPostProcess", "-latestTime", "-func", "patchSurface(name=patchWallBoilingProperties,patch=wall,surfaceFormat=raw,interpolate=false,fields=(wallBoiling:dDeparture wallBoiling:fDeparture wallBoiling:nucleationSiteDensity wallBoiling:wetFraction))", tag="patchSurface.latestTime")


if __name__ == "__main__":
    main()
