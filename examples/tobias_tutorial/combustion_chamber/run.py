"""Tobias Holzmann combustion chamber cold-flow tutorial, FoamPilot runner."""
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
    writer = OpenFOAMDictAddFile(object_name="tobias_combustion_chamber")
    for relative, content in FILES.items():
        path = Path(relative)
        if relative == "system/controlDict":
            content = content.replace("endTime         0.5;", "endTime         0.001;")
            content = content.replace("endTime         1;", "endTime         0.001;")
            content = content.replace("endTime         40;", "endTime         0.001;")
            content = content.replace("endTime         500;", "endTime         0.001;").replace("endTime         20;", "endTime         0.001;")
        writer.write_raw(path.name, CASE, content, folder=str(Path(*path.parts[:-1])))
    shutil.copytree(ROOT / "cad", CASE / "cad")
    shutil.copytree(ROOT / "triSurface", CASE / "constant" / "triSurface", dirs_exist_ok=True)


def run() -> None:
    write_case()
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    if (CASE / "0").exists():
        shutil.rmtree(CASE / "0")
    shutil.copytree(CASE / "0.orig", CASE / "0")
    solver.run_command(["foamRun"], "log.foamRun")
    print(f"Validated combustion-chamber mesh and short cold-flow run: {CASE}")


if __name__ == "__main__":
    run()
