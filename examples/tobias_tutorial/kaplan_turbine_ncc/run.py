"""Tobias Holzmann Kaplan turbine NCC tutorial for OpenFOAM 13."""
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
    writer = OpenFOAMDictAddFile(object_name="tobias_kaplan_turbine_ncc")
    for source in sorted(TEMPLATES.rglob("*")):
        if source.is_file():
            relative = source.relative_to(TEMPLATES)
            writer.write_raw(relative.name, CASE, source.read_text(encoding="utf-8"), folder=str(relative.parent))


def run() -> None:
    write_case()
    background_mesh = CASE / "cad" / "backgroundMesh.unv"
    if not background_mesh.is_file():
        raise FileNotFoundError(
            "kaplan_turbine_ncc requires cad/backgroundMesh.unv; the large "
            "Tobias asset is not included in GitHub."
        )
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["decomposePar", "-force"], "log.decomposePar.mesh")
    solver.run_command(["mpirun", "--oversubscribe", "-np", "4", "snappyHexMesh", "-overwrite", "-parallel"], "log.snappyHexMesh")
    solver.run_command(["reconstructPar", "-constant"], "log.reconstructParMesh")
    solver.run_command(["createNonConformalCouples", "-overwrite", "AMI", "AMI_slave"], "log.createNonConformalCouples")
    shutil.copytree(CASE / "0.orig", CASE / "0")
    solver.run_command(["decomposePar", "-force"], "log.decomposePar.solve")
    solver.run_command(["mpirun", "--oversubscribe", "-np", "4", "renumberMesh", "-overwrite", "-parallel"], "log.renumberMesh")
    solver.run_command(["mpirun", "--oversubscribe", "-np", "4", "foamRun", "-parallel"], "log.foamRun")
    print(f"Validated Kaplan turbine NCC workflow: {CASE}")


if __name__ == "__main__":
    run()
