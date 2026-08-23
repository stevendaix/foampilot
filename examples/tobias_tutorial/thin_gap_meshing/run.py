"""Tobias Holzmann thin-gap meshing tutorial, FoamPilot runner."""
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
    writer = OpenFOAMDictAddFile(object_name="tobias_thin_gap_meshing")
    for relative, content in FILES.items():
        path = Path(relative)
        if relative == "system/controlDict":
            content = content.replace("endTime         500;", "endTime         0.001;")
        writer.write_raw(path.name, CASE, content, folder=str(Path(*path.parts[:-1])))
    shutil.copytree(ROOT / "cad", CASE / "cad")
    shutil.copytree(ROOT / "triSurface", CASE / "constant" / "triSurface")


def run() -> None:
    write_case()
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["transformPoints", "scale=(10000 10000 10000)"], "log.transformPoints.up")
    solver.run_command(
        ["surfaceTransformPoints", "scale=(10000 10000 10000)",
         "constant/triSurface/regionSTL.orig.stl", "constant/triSurface/regionSTL.stl"],
        "log.surfaceTransformPoints.region",
    )
    solver.run_command(
        ["surfaceTransformPoints", "scale=(10000 10000 10000)",
         "constant/triSurface/specialGapRefinement.orig.stl",
         "constant/triSurface/specialGapRefinement.stl"],
        "log.surfaceTransformPoints.gap",
    )
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    solver.run_command(["transformPoints", "scale=(0.0001 0.0001 0.0001)"], "log.transformPoints.down")
    if (CASE / "0").exists():
        shutil.rmtree(CASE / "0")
    shutil.copytree(CASE / "0.orig", CASE / "0")
    solver.run_command(["foamRun"], "log.foamRun")
    print(f"Validated thin-gap meshing and short calculation: {CASE}")


if __name__ == "__main__":
    run()
