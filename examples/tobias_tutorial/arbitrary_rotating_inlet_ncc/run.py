"""Tobias Holzmann arbitrary rotating inlet NCC tutorial for OpenFOAM 13."""
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
    writer = OpenFOAMDictAddFile(object_name="tobias_arbitrary_rotating_inlet_ncc")
    for source in sorted(TEMPLATES.rglob("*")):
        if source.is_file():
            relative = source.relative_to(TEMPLATES)
            writer.write_raw(relative.name, CASE, source.read_text(encoding="utf-8"), folder=str(relative.parent))


def run() -> None:
    write_case()
    background_mesh = CASE / "cad" / "backgroundMesh.unv"
    if not background_mesh.is_file():
        raise FileNotFoundError(
            "arbitrary_rotating_inlet_ncc requires cad/backgroundMesh.unv; "
            "the large Tobias asset is not included in GitHub."
        )
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    solver.run_command(["createPatch", "-overwrite"], "log.createPatch")
    solver.run_command(["createNonConformalCouples", "-overwrite", "interfaceOuter", "interfaceInner"], "log.createNonConformalCouples")
    solver.run_command(["topoSet"], "log.topoSet")
    solver.run_command(["renumberMesh", "-overwrite"], "log.renumberMesh")
    shutil.copytree(CASE / "0.orig", CASE / "0")
    solver.run_command(["foamRun"], "log.foamRun")
    print(f"Validated arbitrary rotating inlet NCC workflow: {CASE}")


if __name__ == "__main__":
    run()
