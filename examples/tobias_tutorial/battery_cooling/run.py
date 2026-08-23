"""Tobias Holzmann Tesla 4680 battery cooling tutorial, FoamPilot runner."""
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
    writer = OpenFOAMDictAddFile(object_name="tobias_battery_cooling")
    for relative, content in FILES.items():
        path = Path(relative)
        if relative == "system/snappyHexMeshDict":
            content = content.replace("level (2 2)", "level (1 1)")
            content = content.replace("nSurfaceLayers 2", "nSurfaceLayers 1")
            content = content.replace("nSolveIter 200", "nSolveIter 50")
        if relative == "system/controlDict":
            content = content.replace("endTime         100;", "endTime         0.001;")
            content = content.replace("endTime         20;", "endTime         0.001;")
            content = content.replace("endTime         5;", "endTime         0.001;")
        writer.write_raw(path.name, CASE, content, folder=str(Path(*path.parts[:-1])))
    shutil.copytree(ROOT / "cad", CASE / "cad")
    tri = CASE / "constant" / "triSurface"
    tri.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "cad" / "stl").glob("*.stl"):
        shutil.copy2(source, tri / source.name)
    shutil.copy2(ROOT / "cad" / "regionSTL.stl", tri / "regionSTL.stl")


def run() -> None:
    write_case()
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    if (CASE / "0").exists():
        shutil.rmtree(CASE / "0")
    shutil.copytree(CASE / "0.orig", CASE / "0")
    solver.run_command(["foamRun"], "log.foamRun")
    print(f"Validated battery-cooling mesh and short thermo-fluid run: {CASE}")


if __name__ == "__main__":
    run()
