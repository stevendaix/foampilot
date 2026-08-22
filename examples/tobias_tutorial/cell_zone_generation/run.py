"""Tobias Holzmann Cell Zone Generation, generated with FoamPilot."""
from pathlib import Path
import shutil
from foampilot.solver import Solver
from foampilot.utilities import OpenFOAMDictAddFile
from templates import FILES

ROOT = Path(__file__).resolve().parent
CASE = ROOT / "case"

def write_case():
    if CASE.exists(): shutil.rmtree(CASE)
    writer = OpenFOAMDictAddFile(object_name="tobias_cell_zone_generation")
    for relative, content in FILES.items():
        path = Path(relative)
        if relative == "system/snappyHexMeshDict":
            for name in ("regionSTL.stl", "Zone1.stl", "Zone2.stl", "Zone3.stl"):
                content = content.replace(f'file "{name}"', f'file "../triSurface/{name}"')
        writer.write_raw(path.name, CASE, content, folder=str(Path(*path.parts[:-1])))
    shutil.copytree(ROOT / "cad", CASE / "cad")
    (CASE / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    for surface in (ROOT / "triSurface").glob("*.stl"):
        shutil.copy2(surface, CASE / "constant" / "triSurface" / surface.name)

def run():
    write_case()
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["surfaceFeatures"], "log.surfaceFeatures")
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    print(f"Validated cell-zone meshing run: {CASE}")

if __name__ == "__main__": run()
