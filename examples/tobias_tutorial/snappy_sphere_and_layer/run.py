"""Tobias Holzmann sphere meshing with layers, FoamPilot runner."""
from pathlib import Path
import shutil
from foampilot.solver import Solver
from foampilot.utilities import OpenFOAMDictAddFile
from templates import FILES

ROOT = Path(__file__).resolve().parent
CASE = ROOT / "case"


def write_case() -> None:
    if CASE.exists():
        shutil.rmtree(CASE)
    writer = OpenFOAMDictAddFile(object_name="tobias_snappy_sphere_and_layer")
    for relative, content in FILES.items():
        path = Path(relative)
        writer.write_raw(path.name, CASE, content, folder=str(Path(*path.parts[:-1])))
    shutil.copytree(ROOT / "cad", CASE / "cad")
    tri_surface = CASE / "constant" / "triSurface"
    tri_surface.mkdir(parents=True, exist_ok=True)
    with (tri_surface / "channel.stl").open("wb") as target:
        for source in sorted((ROOT / "cad" / "stlChannel").glob("*.stl")):
            target.write(source.read_bytes())
    shutil.copy2(ROOT / "cad" / "stlSphere" / "sphere.stl", tri_surface / "sphere.stl")


def run() -> None:
    write_case()
    solver = Solver(CASE)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    # OpenFOAM 13 supersedes surfaceFeatureExtract; the source archive
    # provides the resulting channel.eMesh used by snappyHexMesh.
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    print(f"Validated sphere meshing workflow: {CASE}")


if __name__ == "__main__":
    run()
