"""Tobias Holzmann solar chimney tutorial for OpenFOAM 13."""
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
    writer = OpenFOAMDictAddFile(object_name="tobias_solar_chimney")
    for source in sorted(TEMPLATES.rglob("*")):
        if source.is_file():
            relative = source.relative_to(TEMPLATES)
            writer.write_raw(
                relative.name,
                CASE,
                source.read_text(encoding="utf-8"),
                folder=str(relative.parent),
            )


def run() -> None:
    write_case()
    background_mesh = CASE / "cad" / "backgroundMesh.unv"
    if not background_mesh.is_file():
        raise FileNotFoundError(
            "solar_chimney requires cad/backgroundMesh.unv. The large Tobias "
            "asset is not stored in GitHub; download the complete case archive "
            "from Holzmann CFD before running OpenFOAM 13."
        )
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    shutil.copytree(CASE / "0.orig", CASE / "0")
    solver.run_parallel(2, log_filename="log.foamRun")
    print(f"Validated solar chimney workflow: {CASE}")


if __name__ == "__main__":
    run()
