"""Tobias Holzmann: Pitot Tube, generated and run with FoamPilot.

The downloaded full case provides the CAD/mesh inputs. All text case files
are generated through FoamPilot's raw dictionary writer; the original shell
workflow is not executed.
"""

from pathlib import Path
import shutil

from foampilot.solver import Solver
from foampilot.utilities import OpenFOAMDictAddFile

from templates import FILES


ROOT = Path(__file__).resolve().parent
CASE = ROOT / "case"


def write_case() -> None:
    if CASE.exists():
        shutil.rmtree(CASE)
    writer = OpenFOAMDictAddFile(object_name="tobias_pitot")
    for relative, content in FILES.items():
        relative_path = Path(relative)
        if relative_path.parts[:1] == ("system",):
            folder = "system"
        elif relative_path.parts[:1] == ("constant",):
            folder = str(Path(*relative_path.parts[:-1]))
        else:
            folder = str(Path(*relative_path.parts[:-1]))
        if relative == "system/extrudeMeshDict":
            content = content.replace("sectorCoeffs", "wedgeCoeffs", 1)
        if relative == "system/snappyHexMeshDict":
            content = content.replace('file "prandtl.stl"', 'file "../triSurface/prandtl.stl"')
            content = content.replace('file "refine.stl"', 'file "../triSurface/refine.stl"')
        if relative == "system/extrudeMeshDict":
            content = content.replace(
                "linearNormalCoeffs\n{\n    thickness    0.005;\n}",
                "linearNormalCoeffs\n{\n    nLayers      1;\n    expansionRatio 1.0;\n    thickness    0.005;\n}",
            )
        if relative == "system/controlDict":
            content = content.replace("endTime         300;", "endTime         0.0005;")
        writer.write_raw(relative_path.name, CASE, content, folder=folder)
    shutil.copytree(ROOT / "cad", CASE / "cad")
    shutil.copytree(ROOT / "triSurface", CASE / "constant" / "triSurface")


def run() -> None:
    write_case()
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    solver.run_command(["changeDictionary"], "log.changeDictionary")
    solver.run_command(["extrudeMesh"], "log.extrudeMesh")
    solver.run_command(["foamRun", "-solver", "incompressibleFluid"], "log.foamRun")
    print(f"Validated Pitot Tube generation and short calculation: {CASE}")


if __name__ == "__main__":
    run()
