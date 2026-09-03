"""Tobias Holzmann fan rotation and NCC tutorial for OpenFOAM 13."""
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
    writer = OpenFOAMDictAddFile(object_name="tobias_fan_rotation_ncc")
    for source in sorted(TEMPLATES.rglob("*")):
        if source.is_file():
            relative = source.relative_to(TEMPLATES)
            writer.write_raw(
                relative.name,
                CASE,
                source.read_text(encoding="utf-8"),
                folder=str(relative.parent),
            )
    shutil.copytree(ROOT / "cad", CASE / "cad", dirs_exist_ok=True)


def run() -> None:
    write_case()
    background_mesh = CASE / "cad" / "backgroundMesh.unv"
    if not background_mesh.is_file():
        raise FileNotFoundError(
            "fan_rotation_ncc requires cad/backgroundMesh.unv. "
            "The Tobias GitHub repository removes this large asset; download "
            "the complete case archive from Holzmann CFD before running OpenFOAM 13."
        )

    solver = Solver(CASE)
    solver.run_command(
        ["ideasUnvToFoam", "cad/backgroundMesh.unv"],
        "log.ideasUnvToFoam",
    )
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    solver.run_command(["createBaffles", "-overwrite"], "log.createBaffles")
    solver.run_command(["splitBaffles", "-overwrite"], "log.splitBaffles")
    solver.run_command(
        ["createNonConformalCouples", "-overwrite", "AMI1", "AMI2"],
        "log.createNonConformalCouples",
    )
    solver.run_command(["renumberMesh", "-overwrite"], "log.renumberMesh")
    shutil.copytree(CASE / "0.orig", CASE / "0")
    solver.run_parallel(4, log_filename="log.foamRun")
    print(f"Validated fan rotation NCC workflow: {CASE}")


if __name__ == "__main__":
    run()
