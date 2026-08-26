"""Tobias Holzmann gin-tonic conjugate heat-transfer tutorial for OpenFOAM 13."""
from pathlib import Path
import shutil

from foampilot.solver import Solver
from foampilot.utilities import OpenFOAMDictAddFile

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"
CASE = ROOT / "case"
REGIONS = ("ginTonic", "iceCube1", "iceCube2")


def write_case() -> None:
    if CASE.exists():
        shutil.rmtree(CASE)
    writer = OpenFOAMDictAddFile(object_name="tobias_gin_tonic_cht")
    for source in sorted(TEMPLATES.rglob("*")):
        if source.is_file():
            relative = source.relative_to(TEMPLATES)
            writer.write_raw(relative.name, CASE, source.read_text(encoding="utf-8"), folder=str(relative.parent))


def run() -> None:
    write_case()
    background_mesh = CASE / "cad" / "backgroundMesh.unv"
    if not background_mesh.is_file():
        raise FileNotFoundError(
            "gin_tonic_cht requires cad/backgroundMesh.unv; the large Tobias "
            "asset is not included in GitHub."
        )
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    solver.run_command(["splitMeshRegions", "-cellZones", "-defaultRegionName", "air", "-overwrite"], "log.splitMeshRegions")
    for region in REGIONS:
        solver.run_command(["createPatch", "-region", region, "-overwrite"], f"log.createPatch.{region}")
        solver.run_command(["changeDictionary", "-region", region], f"log.changeDictionary.{region}")
    for region in REGIONS:
        source = "thermoPropertiesFluid" if region == "ginTonic" else "thermoPropertiesSolid"
        target = CASE / "constant" / region
        target.mkdir(parents=True, exist_ok=True)
        for item in (CASE / "constant" / source).iterdir():
            link = target / item.name
            if not link.exists():
                link.symlink_to(Path("..") / source / item.name)
    for region in REGIONS:
        solver.run_command(["decomposePar", "-region", region], f"log.decomposePar.{region}")
    solver.run_command(
        ["mpirun", "--oversubscribe", "-np", "2", "foamMultiRun", "-parallel"],
        "log.foamMultiRun",
    )
    print(f"Validated gin-tonic CHT workflow: {CASE}")


if __name__ == "__main__":
    run()
