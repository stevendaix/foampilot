"""Tobias Holzmann Magnus Effect, generated and run with FoamPilot."""
from pathlib import Path
import shutil
from foampilot.solver import Solver
from foampilot.utilities import OpenFOAMDictAddFile
from templates import FILES

ROOT = Path(__file__).resolve().parent
CASE = ROOT / "case"

def write_case():
    if CASE.exists():
        shutil.rmtree(CASE)
    writer = OpenFOAMDictAddFile(object_name="tobias_magnus_effect")
    for relative, content in FILES.items():
        path = Path(relative)
        if relative == "system/snappyHexMeshDict":
            content = content.replace('file "cylinder.stl"', 'file "../triSurface/cylinder.stl"')
        if relative == "system/extrudeMeshDict":
            content = content.replace("linearNormalCoeffs\n{\n    thickness       0.05;\n}", "linearNormalCoeffs\n{\n    nLayers         1;\n    expansionRatio  1.0;\n    thickness       0.05;\n}")
        if relative == "system/controlDict":
            content = content.replace("endTime         40;", "endTime         0.2;")
        writer.write_raw(path.name, CASE, content, folder=str(Path(*path.parts[:-1])))
    shutil.copytree(ROOT / "cad", CASE / "cad")
    (CASE / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "triSurface" / "cylinder.stl", CASE / "constant" / "triSurface" / "cylinder.stl")
    shutil.copytree(CASE / "0.orig", CASE / "0")

def run():
    write_case()
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    solver.run_command(["extrudeMesh"], "log.extrudeMesh")
    solver.run_command(["changeDictionary"], "log.changeDictionary")
    solver.run_command(["transformPoints", "translate=(0 -0.15 -0.1)"], "log.transformPoints")
    solver.run_command(["foamRun", "-solver", "incompressibleFluid"], "log.foamRun")
    print(f"Validated Magnus Effect smoke run: {CASE}")

if __name__ == "__main__":
    run()
