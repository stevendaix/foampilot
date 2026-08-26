"""Tobias Holzmann Suzanne's head tutorial for OpenFOAM 13."""
from pathlib import Path
import shutil

from foampilot.solver import Solver
from foampilot.utilities import OpenFOAMDictAddFile

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"
CASE = ROOT / "case"


def write_case() -> None:
    if CASE.exists():
        shutil.rmtree(CASE)
    writer = OpenFOAMDictAddFile(object_name="tobias_suzannes_head")
    for source in sorted(TEMPLATES.rglob("*")):
        if source.is_file():
            relative = source.relative_to(TEMPLATES)
            writer.write_raw(relative.name, CASE, source.read_text(encoding="utf-8"), folder=str(relative.parent))


def run() -> None:
    write_case()
    background_mesh = CASE / "cad" / "backgroundMesh.unv"
    if not background_mesh.is_file():
        raise FileNotFoundError(
            "suzannes_head requires cad/backgroundMesh.unv; the large Tobias "
            "asset is not included in GitHub."
        )
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["decomposePar"], "log.decomposePar")
    solver.run_command(["mpirun", "--oversubscribe", "-np", "4", "snappyHexMesh", "-parallel", "-overwrite"], "log.snappyHexMesh")
    solver.run_command(["mpirun", "--oversubscribe", "-np", "4", "checkMesh", "-parallel"], "log.checkMesh")
    solver.run_command(["mpirun", "--oversubscribe", "-np", "4", "transformPoints", "scale=(0.1 0.1 0.1)", "-parallel"], "log.transformPoints")
    shutil.copytree(CASE / "0.orig", CASE / "0")
    solver.run_command(["mpirun", "--oversubscribe", "-np", "4", "renumberMesh", "-overwrite", "-parallel"], "log.renumberMesh")
    solver.run_command(["mpirun", "--oversubscribe", "-np", "4", "foamRun", "-parallel"], "log.foamRun")
    print(f"Validated Suzanne head workflow: {CASE}")


if __name__ == "__main__":
    run()
