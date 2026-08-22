"""Tobias Holzmann: bifluidic oscillator, generated with FoamPilot.

The UNV and STL inputs are supplied by the full Tobias archive and are not
committed because they exceed normal repository file limits. Place them next
to this script under ``cad/`` and ``triSurface/`` before running.
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
    writer = OpenFOAMDictAddFile(object_name="tobias_fluidic_oscillator")
    for relative, content in FILES.items():
        relative_path = Path(relative)
        folder = str(Path(*relative_path.parts[:-1]))
        if relative == "system/extrudeMeshDict":
            content = content.replace("sectorCoeffs", "wedgeCoeffs", 1)
            content = content.replace(
                "linearNormalCoeffs\n{\n    thickness       0.005;\n}",
                "linearNormalCoeffs\n{\n    nLayers         1;\n    expansionRatio  1.0;\n    thickness       0.005;\n}",
            )
        if relative == "system/fvSchemes":
            content = content.replace(
                "div(phi,alpha)  Gauss vanLeer;",
                "div(phi,alpha)  Gauss interfaceCompression vanLeer 1;",
            )
        if relative == "system/fvSolution":
            content = content.replace("        cAlpha          1;\n", "")
        if relative == "system/controlDict":
            content = content.replace("endTime         2;", "endTime         0.002;")
        writer.write_raw(relative_path.name, CASE, content, folder=folder)
    shutil.copytree(ROOT / "cad", CASE / "cad")
    (CASE / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    for surface in (ROOT / "triSurface").glob("*.stl"):
        shutil.copy2(surface, CASE / "constant" / "triSurface" / surface.name)
    shutil.copytree(CASE / "0.orig", CASE / "0")


def run() -> None:
    write_case()
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["transformPoints", "translate=(-0.0005 0 0)"], "log.transformPoints.pre")
    solver.run_command(["surfaceFeatures"], "log.surfaceFeatures")
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    solver.run_command(["flattenMesh"], "log.flattenMesh")
    solver.run_command(["extrudeMesh"], "log.extrudeMesh")
    solver.run_command(["transformPoints", "translate=(0 -0.005 0)"], "log.transformPoints.post")
    solver.run_command(["topoSet"], "log.topoSet")
    solver.run_command(["setFields"], "log.setFields")
    solver.run_command(["foamRun", "-solver", "incompressibleVoF"], "log.foamRun")
    print(f"Validated Fluidic Oscillator smoke run: {CASE}")


if __name__ == "__main__":
    run()
