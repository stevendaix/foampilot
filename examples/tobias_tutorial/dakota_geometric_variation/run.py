"""Tobias Holzmann Dakota geometric variation tutorial for OpenFOAM 13."""
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
    writer = OpenFOAMDictAddFile(object_name="tobias_dakota_geometric_variation")
    for source in sorted(TEMPLATES.rglob("*")):
        if source.is_file():
            relative = source.relative_to(TEMPLATES)
            writer.write_raw(
                relative.name,
                CASE,
                source.read_text(encoding="utf-8"),
                folder=str(relative.parent),
            )
    (CASE / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ASSETS / ".dakotaInput.dak", CASE / "assets" / ".dakotaInput.dak")
    shutil.copy2(ASSETS / "paraview.pvsm", CASE / "assets" / "paraview.pvsm")


def run() -> None:
    write_case()
    solver = Solver(CASE)
    solver.run_command(["blockMesh"], "log.blockMesh")
    solver.run_command(["extrudeMesh"], "log.extrudeMesh")
    solver.run_command(["createPatch", "-overwrite"], "log.createPatch")
    solver.run_command(["renumberMesh", "-constant", "-overwrite"], "log.renumberMesh")
    if shutil.which("dakota") is None:
        raise RuntimeError(
            "Dakota geometric variation mesh generated, but executable 'dakota' "
            "is not installed; install Dakota and rerun the optimization stage."
        )
    solver.run_command(
        ["dakota", "-i", "system/dakotaDict", "-o", "dakotaLog"],
        "log.dakota",
    )
    print(f"Validated Dakota geometric variation workflow: {CASE}")


if __name__ == "__main__":
    run()
