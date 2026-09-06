"""Tobias Holzmann helix meshing tutorial for OpenFOAM 13."""
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
    writer = OpenFOAMDictAddFile(object_name="tobias_meshing_a_helix")
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
    required = (
        CASE / "cad" / "backgroundMesh.unv",
        CASE / "constant" / "triSurface" / "layer_orig.stl",
        CASE / "constant" / "triSurface" / "regionSTL_orig.stl",
    )
    missing = [str(path.relative_to(CASE)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "meshing_a_helix is missing required Tobias assets: " + ", ".join(missing)
        )

    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    shutil.copy2(CASE / "constant/triSurface/layer_orig.stl", CASE / "constant/triSurface/layer.stl")
    shutil.copy2(CASE / "constant/triSurface/regionSTL_orig.stl", CASE / "constant/triSurface/regionSTL.stl")
    solver.run_command(["transformPoints", "scale=(1000 1000 1000)"], "log.transformPoints.up")
    solver.run_command(["surfaceFeatures"], "log.surfaceFeatures")
    solver.run_command(
        ["decomposePar"],
        "log.decomposePar",
    )
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "4", "snappyHexMesh", "-overwrite", "-parallel"],
        "log.snappyHexMesh.layers",
    )
    shutil.copy2(CASE / "system/meshQualityDict.layer", CASE / "system/meshQualityDict")
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "4", "snappyHexMesh", "-overwrite", "-parallel"],
        "log.snappyHexMesh.layers.second",
    )
    shutil.copy2(CASE / "system/meshQualityDict.normal", CASE / "system/meshQualityDict")
    solver.run_command(["reconstructParMesh", "-constant"], "log.reconstructParMesh")
    solver.run_command(["createPatch", "-overwrite"], "log.createPatch")
    solver.run_command(["transformPoints", "scale=(0.001 0.001 0.001)"], "log.transformPoints.down")
    solver.run_command(["checkMesh", "-constant"], "log.checkMesh")
    print(f"Validated helix meshing workflow: {CASE}")


if __name__ == "__main__":
    run()
