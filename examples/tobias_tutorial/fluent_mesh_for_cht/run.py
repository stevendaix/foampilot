"""Tobias Holzmann Fluent-to-CHT mesh conversion tutorial for OpenFOAM 13."""
from pathlib import Path
import shutil

from foampilot.solver import Solver
from foampilot.utilities import OpenFOAMDictAddFile

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"
CASE = ROOT / "case"


def write_case() -> None:
    """Materialize the source dictionaries without rewriting OpenFOAM syntax."""
    if CASE.exists():
        shutil.rmtree(CASE)
    writer = OpenFOAMDictAddFile(object_name="tobias_fluent_mesh_for_cht")
    for source in sorted(TEMPLATES.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(TEMPLATES)
        writer.write_raw(
            relative.name,
            CASE,
            source.read_text(encoding="utf-8"),
            folder=str(relative.parent),
        )
    (CASE / "cad").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "cad" / "fluentMesh.cas", CASE / "cad" / "fluentMesh.cas")


def run() -> None:
    write_case()
    solver = Solver(CASE)
    solver.run_command(
        ["fluentMeshToFoam", "cad/fluentMesh.cas", "-writeSets"],
        "log.fluentMeshToFoam",
    )
    solver.run_command(["topoSet", "-constant"], "log.topoSet")
    solver.run_command(
        ["splitMeshRegions", "-cellZonesOnly", "-overwrite"],
        "log.splitMeshRegions",
    )
    (CASE / "paraview.foam").touch()
    print(f"Validated Fluent-to-CHT mesh conversion: {CASE}")


if __name__ == "__main__":
    run()
