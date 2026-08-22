"""Tobias Holzmann Falling Droplets, generated and run with FoamPilot."""
from pathlib import Path
import shutil
from foampilot.solver import Solver
from foampilot.utilities import OpenFOAMDictAddFile
from templates import FILES

ROOT = Path(__file__).resolve().parent
CASE = ROOT / "case"

def write_case():
    if CASE.exists(): shutil.rmtree(CASE)
    writer = OpenFOAMDictAddFile(object_name="tobias_falling_droplets")
    for relative, content in FILES.items():
        path = Path(relative)
        if relative == "system/fvSchemes":
            content = content.replace("div(phi,alpha)  Gauss vanLeer;", "div(phi,alpha)  Gauss interfaceCompression vanLeer 1;")
        if relative == "system/fvSolution":
            content = content.replace("        cAlpha          1;\n", "")
        if relative == "system/controlDict":
            content = content.replace("endTime         0.12;", "endTime         0.002;")
        writer.write_raw(path.name, CASE, content, folder=str(Path(*path.parts[:-1])))
    shutil.copytree(ROOT / "cad", CASE / "cad")
    shutil.copytree(CASE / "0.orig", CASE / "0")

def run():
    write_case()
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/meshSquare.unv"], "log.ideasUnvToFoam")
    solver.run_command(["changeDictionary"], "log.changeDictionary")
    solver.run_command(["setFields"], "log.setFields")
    solver.run_command(["foamRun", "-solver", "incompressibleVoF"], "log.foamRun")
    print(f"Validated Falling Droplets smoke run: {CASE}")

if __name__ == "__main__": run()
