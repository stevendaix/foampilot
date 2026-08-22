"""Tobias Holzmann 2D AMI/NCC tutorial, FoamPilot runner."""
from pathlib import Path
import shutil
from foampilot.solver import Solver
from foampilot.utilities import OpenFOAMDictAddFile
from templates import FILES
ROOT = Path(__file__).resolve().parent
CASE = ROOT / "case"
def write_case():
    if CASE.exists(): shutil.rmtree(CASE)
    writer = OpenFOAMDictAddFile(object_name="tobias_2d_ami_ncc")
    for relative, content in FILES.items():
        path = Path(relative)
        if relative == "system/snappyHexMeshDict":
            for name in ("regionSTL.stl", "refine.stl", "AMI.stl"):
                content = content.replace(f'file "{name}"', f'file "../triSurface/{name}"')
        if relative == "0.orig/U":
            content = content.replace("type            zeroGradient\n", "type            zeroGradient;\n")
        if relative == "system/extrudeMeshDict":
            content = content.replace("linearNormalCoeffs\n{\n    thickness       0.005;\n}", "linearNormalCoeffs\n{\n    nLayers         1;\n    expansionRatio  1.0;\n    thickness       0.005;\n}")
        if relative == "system/controlDict":
            content = content.replace("endTime         6;", "endTime         0.001;").replace("endTime         40;", "endTime         0.001;")
        writer.write_raw(path.name, CASE, content, folder=str(Path(*path.parts[:-1])))
    shutil.copytree(ROOT / "cad", CASE / "cad")
    (CASE / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    for surface in (ROOT / "triSurface").glob("*.stl"):
        shutil.copy2(surface, CASE / "constant" / "triSurface" / surface.name)
    shutil.copy2(ROOT / ".dynamicMeshDict", CASE / "constant" / "dynamicMeshDict")
def run():
    write_case()
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    solver.run_command(["changeDictionary"], "log.changeDictionary")
    solver.run_command(["flattenMesh"], "log.flattenMesh")
    solver.run_command(["extrudeMesh"], "log.extrudeMesh")
    solver.run_command(["topoSet"], "log.topoSet")
    solver.run_command(["createBaffles", "-overwrite"], "log.createBaffles")
    solver.run_command(["splitBaffles", "-overwrite"], "log.splitBaffles")
    solver.run_command(["createNonConformalCouples", "-overwrite", "AMI1", "AMI2"], "log.createNonConformalCouples")
    shutil.copytree(CASE / "0.orig", CASE / "0", dirs_exist_ok=True)
    solver.run_command(["foamRun"], "log.foamRun")
    print(f"Validated 2D AMI NCC smoke run: {CASE}")
if __name__ == "__main__": run()
