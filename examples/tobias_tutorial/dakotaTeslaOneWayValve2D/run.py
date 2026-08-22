"""Tobias Holzmann Dakota Tesla One-Way Valve tutorial, FoamPilot runner."""
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
    writer = OpenFOAMDictAddFile(object_name="tobias_dakota_tesla")
    for relative, content in FILES.items():
        path = Path(relative)
        if relative == "system/controlDict":
            # Reduce end time for a smoke run
            content = content.replace("endTime         1000;", "endTime         5;")
        if relative in ("system/dakotaDict", "dakota.sh"):
            if relative == "system/dakotaDict":
                # Reduce Dakota iterations
                content = content.replace("initial_samples = 10", "initial_samples = 2")
            # Write directly to avoid FoamFile header
            (CASE / relative).parent.mkdir(parents=True, exist_ok=True)
            with open(CASE / relative, "w") as f:
                f.write(content)
            continue
        if relative == "system/extrudeMeshDict":
            # Add OpenFOAM 13 required coefficients for linearNormal
            content = content.replace(
                "linearNormalCoeffs\n{",
                "linearNormalCoeffs\n{\n    nLayers 1;\n    expansionRatio 1;"
            )
        writer.write_raw(path.name, CASE, content, folder=str(Path(*path.parts[:-1])))
    
    shutil.copytree(ROOT / "cad", CASE / "cad", dirs_exist_ok=True)
    shutil.copytree(ROOT / "triSurface", CASE / "constant" / "triSurface", dirs_exist_ok=True)
    
    # Make dakota.sh executable
    dakota_sh = CASE / "dakota.sh"
    if dakota_sh.exists():
        dakota_sh.chmod(0o755)

def run() -> None:
    write_case()
    solver = Solver(CASE)
    
    # Meshing
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["surfaceFeatures"], "log.surfaceFeatures")
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    solver.run_command(["extrudeMesh"], "log.extrudeMesh")
    solver.run_command(["createPatch", "-overwrite"], "log.createPatch")
    solver.run_command(["renumberMesh", "-constant", "-overwrite"], "log.renumberMesh")
    
    # Initialize optimization state
    with open(CASE / ".optimizationLoop", "w") as f:
        f.write("1\n")
    
    if (CASE / "0").exists():
        shutil.rmtree(CASE / "0")
    shutil.copytree(CASE / "0.orig", CASE / "0")
    
    # Start Dakota
    # Replace direct foamRun call with FoamPilot solve.py in dakota.sh
    dakota_sh_path = CASE / "dakota.sh"
    with open(dakota_sh_path, "r") as f:
        dakota_script = f.read()
    
    dakota_script = dakota_script.replace("foamRun > $logFolder_/solving", "python3 ../solve.py $logFolder_/solving")
    
    with open(dakota_sh_path, "w") as f:
        f.write(dakota_script)
        
    solver.run_command(["dakota", "-i", "system/dakotaDict", "-o", "dakotaLog"], "optimizationLog")
    
    print(f"Validated Dakota Tesla optimization workflow: {CASE}")

if __name__ == "__main__":
    run()
