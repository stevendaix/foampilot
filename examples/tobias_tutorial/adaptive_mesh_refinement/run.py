"""Tobias Holzmann pseudo-2D adaptive mesh refinement, FoamPilot runner."""
from pathlib import Path
import shutil
from foampilot.solver import Solver
from foampilot.utilities import OpenFOAMDictAddFile
from templates import FILES
ROOT = Path(__file__).resolve().parent
CASE = ROOT / "case"
def write_case():
    if CASE.exists(): shutil.rmtree(CASE)
    writer = OpenFOAMDictAddFile(object_name="tobias_adaptive_mesh_refinement")
    for relative, content in FILES.items():
        path = Path(relative)
        if relative in ("system/snappyHexMeshDict", "system/surfaceFeaturesDict"):
            content = content.replace('file "cylinder.stl"', 'file "cylinder.stl"').replace('"../triSurface/cylinder.stl"', '"cylinder.stl"')
        if relative == "system/controlDict":
            content = content.replace("endTime         0.5;", "endTime         0.001;").replace("endTime         1;", "endTime         0.001;").replace("endTime         40;", "endTime         0.001;").replace("endTime         0.12;", "endTime         0.001;").replace("endTime         5;", "endTime         0.001;")
        if relative == "constant/dynamicMeshDict":
            content = content.replace("maxCells        400000;", "maxCells        120000;")
        writer.write_raw(path.name, CASE, content, folder=str(Path(*path.parts[:-1])))
    shutil.copytree(ROOT / "cad", CASE / "cad")
    (CASE / "constant" / "geometry").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "triSurface" / "cylinder.stl", CASE / "constant" / "geometry" / "cylinder.stl")
    shutil.copytree(CASE / "0.orig", CASE / "0")
def run():
    write_case()
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["surfaceFeatures"], "log.surfaceFeatures")
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    solver.run_command(["foamRun"], "log.foamRun")
    print(f"Validated AMR smoke run: {CASE}")
if __name__ == "__main__": run()
