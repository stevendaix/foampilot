"""OpenFOAM 13 mesh/blockMesh/sphere7ProjectedEdges through FoamPilot only."""
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

REFERENCE = Path("/opt/openfoam13/tutorials/mesh/blockMesh/sphere7ProjectedEdges")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "blockMesh"
    solver.setup_case()
    solver.system.import_reference_file(
        REFERENCE / "system/controlDict", filename="controlDict"
    )
    solver.system.import_reference_file(
        REFERENCE / "system/blockMeshDict", filename="blockMeshDict"
    )
    solver.run_command([str(OF13_BIN / "blockMesh")], log_filename="log.blockMesh")


if __name__ == "__main__":
    main()
