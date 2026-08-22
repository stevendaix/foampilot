"""Tobias Holzmann: 2D rotational axis-symmetric meshing, OpenFOAM 13.

The full-case archive supplies the STL geometry. Every OpenFOAM dictionary is
written by FoamPilot's ``OpenFOAMDictAddFile.write_raw`` method; the original
shell ``run`` file is not executed.
"""

from pathlib import Path
import shutil

from foampilot.solver import Solver
from foampilot.utilities import OpenFOAMDictAddFile

from templates import DICTIONARIES


ROOT = Path(__file__).resolve().parent
CASE = ROOT / "case"


def write_case() -> None:
    if CASE.exists():
        shutil.rmtree(CASE)
    writer = OpenFOAMDictAddFile(object_name="tobias_case")
    for name, content in DICTIONARIES.items():
        if name == "extrudeMeshDict":
            content = content.replace("sectorCoeffs", "wedgeCoeffs", 1)
        writer.write_raw(name, CASE, content, folder="system")
    geometry = CASE / "constant" / "triSurface"
    geometry.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "join.stl", geometry / "join.stl")


def run() -> None:
    write_case()
    solver = Solver(CASE)
    solver.run_command(["blockMesh"], "log.blockMesh")
    solver.run_command(["surfaceFeatures"], "log.surfaceFeatures")
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    boundary = CASE / "constant" / "polyMesh" / "boundary"
    text = boundary.read_text(encoding="utf-8")
    boundary.write_text(text.replace("minZ", "front"), encoding="utf-8")
    solver.run_command(["extrudeMesh"], "log.extrudeMesh")
    solver.run_command(["createPatch", "-overwrite"], "log.createPatch")
    if not boundary.exists():
        raise RuntimeError("OpenFOAM did not produce constant/polyMesh/boundary")
    print(f"Validated meshing run: {CASE}")


if __name__ == "__main__":
    run()
