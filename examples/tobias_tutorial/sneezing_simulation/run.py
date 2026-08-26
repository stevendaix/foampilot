"""Tobias Holzmann sneezing simulation tutorial for OpenFOAM 13."""
from pathlib import Path
import shutil

from foampilot.solver import Solver
from foampilot.utilities import OpenFOAMDictAddFile

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"
ASSETS = ROOT / "assets"
CASE = ROOT / "case"


def write_case() -> None:
    if CASE.exists():
        shutil.rmtree(CASE)
    writer = OpenFOAMDictAddFile(object_name="tobias_sneezing_simulation")
    for source in sorted(TEMPLATES.rglob("*")):
        if source.is_file():
            relative = source.relative_to(TEMPLATES)
            writer.write_raw(
                relative.name,
                CASE,
                source.read_text(encoding="utf-8"),
                folder=str(relative.parent),
            )
    shutil.copytree(ASSETS, CASE / "assets", dirs_exist_ok=True)


def run() -> None:
    write_case()
    background_mesh = CASE / "cad" / "backgroundMesh.unv"
    if not background_mesh.is_file():
        raise FileNotFoundError(
            "sneezing_simulation requires cad/backgroundMesh.unv. The Tobias "
            "GitHub repository does not include this large mesh asset."
        )
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["decomposePar"], "log.decomposePar")
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "8", "snappyHexMesh", "-overwrite", "-parallel"],
        "log.snappyHexMesh",
    )
    shutil.copytree(CASE / "0.orig", CASE / "0")
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "8", "renumberMesh", "-overwrite", "-parallel"],
        "log.renumberMesh",
    )
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "8", "foamRun", "-parallel"],
        "log.foamRun",
    )
    print(f"Validated sneezing simulation workflow: {CASE}")


if __name__ == "__main__":
    run()
