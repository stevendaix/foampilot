"""Tobias Holzmann vertical axial wind turbine NCC tutorial for OpenFOAM 13."""
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
    writer = OpenFOAMDictAddFile(object_name="tobias_vertical_axial_wind_turbine_ncc")
    for source in sorted(TEMPLATES.rglob("*")):
        if source.is_file():
            relative = source.relative_to(TEMPLATES)
            writer.write_raw(relative.name, CASE, source.read_text(encoding="utf-8"), folder=str(relative.parent))


def run() -> None:
    write_case()
    background_mesh = CASE / "cad" / "backgroundMesh.unv"
    if not background_mesh.is_file():
        raise FileNotFoundError(
            "vertical_axial_wind_turbine_ncc requires cad/backgroundMesh.unv; "
            "the large Tobias asset is not included in GitHub."
        )
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["surfaceFeatures"], "log.surfaceFeatures")
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    solver.run_command(["extrudeMesh"], "log.extrudeMesh")
    solver.run_command(["changeDictionary"], "log.changeDictionary")
    solver.run_command(["createNonConformalCouples", "-overwrite", "AMI", "AMI_slave"], "log.createNonConformalCouples")
    shutil.copytree(CASE / "0.orig", CASE / "0")
    solver.run_parallel(2, log_filename="log.foamRun")
    print(f"Validated vertical axial wind turbine NCC workflow: {CASE}")


if __name__ == "__main__":
    run()
