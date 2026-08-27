"""OpenFOAM 13 mesh/refineMesh/sector through FoamPilot only."""
from pathlib import Path
import os
import sys

OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")
OF13_LIB = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/lib")
OF13_TP_LIB = Path("/opt/openfoam13/thirdparty/platforms/linux64GccDPInt32/lib")
os.environ.update({
    "WM_PROJECT": "OpenFOAM",
    "WM_PROJECT_VERSION": "13",
    "WM_PROJECT_DIR": "/opt/openfoam13",
    "FOAM_APPBIN": str(OF13_BIN),
    "FOAM_LIBBIN": str(OF13_LIB),
    "WM_OPTIONS": "linux64GccDPInt32Opt",
    "WM_ARCH": "linux64",
    "WM_ARCH_OPTION": "64",
    "WM_COMPILER": "Gcc",
    "WM_COMPILER_LIB_ARCH": "64",
    "WM_COMPILE_OPTION": "Opt",
    "WM_PRECISION_OPTION": "DP",
    "WM_LABEL_SIZE": "32",
    "WM_LINK_LANGUAGE": "c++",
    "WM_DIR": "/opt/openfoam13/wmake",
    "FOAM_MPI": "openmpi-system",
    "WM_MPLIB": "SYSTEMOPENMPI",
    "FOAM_MODULES": str(OF13_LIB),
    "MPI_BUFFER_SIZE": "20000000",
    "PATH": f"{OF13_BIN}:/opt/openfoam13/bin:/opt/openfoam13/wmake:{os.environ.get('PATH', '')}",
    "LD_LIBRARY_PATH": f"{OF13_LIB / 'openmpi-system'}:{OF13_TP_LIB / 'openmpi-system'}:{OF13_TP_LIB}:{OF13_LIB}:{OF13_LIB / 'dummy'}:{os.environ.get('LD_LIBRARY_PATH', '')}",
})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/mesh/refineMesh/sector")


def import_reference_case(solver: Solver) -> None:
    for source in REFERENCE.rglob("*"):
        if not source.is_file() or source.name in {"Allrun", "Allclean"}:
            continue
        relative = source.relative_to(REFERENCE)
        if relative.parts[0] == "system":
            solver.system.import_reference_file(
                source, filename=relative.relative_to("system")
            )


def run_utility(solver: Solver, utility: str, args: list[str], tag: str) -> None:
    solver.system.run_utility(
        utility, args=args, log_filename=f"log.{tag}"
    )


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "refineMesh"
    solver.setup_case()
    import_reference_case(solver)

    run_utility(solver, "blockMesh", [], "blockMesh")

    # Reproduce the OF13 axial refinement loop.
    minimum = 0.0
    for maximum in (0.64, 1.03, 1.94, 3.67, 7.00, 13.36):
        run_utility(solver, "foamPostProcess", ["-func", "R"], f"R_z_{minimum}_to_{maximum}")
        solver.system.import_reference_file(
            REFERENCE / "system/refineMeshDict.z", filename="refineMeshDict"
        )
        solver.system.update_dictionary_entries(
            "system/refineMeshDict", {"zone/radius": str(maximum)}
        )
        run_utility(solver, "refineMesh", [], f"z_{minimum}_to_{maximum}")
        minimum = maximum

    # Reproduce the OF13 cylindrical refinement loop.
    maximum = 18.47
    for minimum in (13.36, 7.00, 3.67):
        run_utility(solver, "foamPostProcess", ["-func", "R"], f"R_cyl_{minimum}_to_{maximum}")
        run_utility(solver, "foamPostProcess", ["-func", "eRThetaZ"], f"eRThetaZ_cyl_{minimum}_to_{maximum}")
        solver.system.import_reference_file(
            REFERENCE / "system/refineMeshDict.cyl", filename="refineMeshDict"
        )
        solver.system.update_dictionary_entries(
            "system/refineMeshDict", {"zone/innerRadius": str(minimum)}
        )
        run_utility(solver, "refineMesh", [], f"cyl_{minimum}_to_{maximum}")


if __name__ == "__main__":
    main()
