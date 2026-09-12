"""OpenFOAM 13 multicomponentFluid/SandiaD_LTS via FoamPilot."""
from pathlib import Path
import os
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/multicomponentFluid/SandiaD_LTS")
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
            relative = source.relative_to(REFERENCE / "0")
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=str(relative)
            )
    for root in ("constant", "system", "chemkin"):
        for source in (REFERENCE / root).rglob("*"):
            if source.is_file():
                solver.import_reference_asset(
                    source, case_path / source.relative_to(REFERENCE)
                )


def run_of13(solver: Solver, executable: str, *args: str, tag: str) -> None:
    command_line = shlex.join([str(OF13_BIN / executable), *args])
    solver.run_command(
        ["/bin/bash", "-lc", f"source /opt/openfoam13/etc/bashrc >/dev/null 2>&1 && exec {command_line}"],
        log_filename=f"log.{tag}",
        environment=OF13_ENV,
    )


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "multicomponentFluid"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    run_of13(
        solver,
        "chemkinToFoam",
        "chemkin/grimech30.dat",
        "chemkin/thermo30.dat",
        "chemkin/transportProperties",
        "constant/reactionsGRI",
        "constant/thermo.compressibleGasGRI",
        tag="chemkinToFoam",
    )
    run_of13(solver, "blockMesh", tag="blockMesh")
    run_of13(solver, "setFields", tag="setFields")

    run_of13(solver, "foamDictionary", "-entry", "startTime", "-set", "0", "system/controlDict", tag="foamDictionary.startTime.0")
    run_of13(solver, "foamDictionary", "-entry", "writeInterval", "-set", "1500", "system/controlDict", tag="foamDictionary.writeInterval.1500")
    run_of13(solver, "foamDictionary", "-entry", "endTime", "-set", "1500", "system/controlDict", tag="foamDictionary.endTime.1500")
    run_of13(solver, "foamDictionary", "-entry", "chemistry", "-set", "off", "constant/chemistryProperties", tag="foamDictionary.chemistry.off")
    run_of13(solver, "foamRun", tag="foamRun.noChemistry")

    run_of13(solver, "foamDictionary", "-entry", "startTime", "-set", "1500", "system/controlDict", tag="foamDictionary.startTime.1500")
    run_of13(solver, "foamDictionary", "-entry", "writeInterval", "-set", "100", "system/controlDict", tag="foamDictionary.writeInterval.100")
    run_of13(solver, "foamDictionary", "-entry", "endTime", "-set", "5000", "system/controlDict", tag="foamDictionary.endTime.5000")
    run_of13(solver, "foamDictionary", "-entry", "chemistry", "-set", "on", "constant/chemistryProperties", tag="foamDictionary.chemistry.on")
    run_of13(solver, "foamRun", tag="foamRun.chemistry")
    run_of13(solver, "foamRun", tag="foamRun.final")


if __name__ == "__main__":
    main()
