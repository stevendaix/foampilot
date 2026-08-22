"""Tobias Holzmann feature-edge refinement tutorial, FoamPilot runner."""
from pathlib import Path
import shutil
from foampilot.solver import Solver
from foampilot.utilities import OpenFOAMDictAddFile
from templates import FILES

ROOT = Path(__file__).resolve().parent
CASE = ROOT / "case"


def write_case(edge_mesh: str, use_levels: bool = False) -> None:
    if CASE.exists():
        shutil.rmtree(CASE)
    writer = OpenFOAMDictAddFile(object_name="tobias_feature_edge_refinement")
    for relative, content in FILES.items():
        path = Path(relative)
        if relative == "system/snappyHexMeshDict":
            content = content.replace(
                'file "featureEdges.eMesh"',
                'file "featureEdges.eMesh"',
            )
            if use_levels:
                content = content.replace("        level 2", "        levels ((0.01 2))")
        writer.write_raw(path.name, CASE, content, folder=str(Path(*path.parts[:-1])))
    shutil.copytree(ROOT / "cad", CASE / "cad")
    tri_surface = CASE / "constant" / "triSurface"
    tri_surface.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "triSurface").glob("*"):
        shutil.copy2(source, tri_surface / source.name)


def run_variant(name: str, edge_mesh: str, use_levels: bool = False) -> None:
    write_case(edge_mesh, use_levels)
    solver = Solver(CASE)
    solver.run_command(
        [
            "surfaceFeatureConvert",
            f"constant/triSurface/{edge_mesh}",
            "constant/triSurface/featureEdges.eMesh",
        ],
        f"log.surfaceFeatureConvert.{name}",
    )
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], f"log.ideasUnvToFoam.{name}")
    solver.run_command(["snappyHexMesh", "-overwrite"], f"log.snappyHexMesh.{name}")
    result = ROOT / name
    if result.exists():
        shutil.rmtree(result)
    result.mkdir()
    shutil.copytree(CASE / "constant" / "polyMesh", result / "polyMesh")


def run() -> None:
    run_variant("standard_edge_mesh", "featureEdgesStandard.obj")
    run_variant("optimized_edge_mesh", "featureEdgesOptimized.obj")
    run_variant("standard_edge_mesh_levels", "featureEdgesStandard.obj", use_levels=True)
    print(f"Validated feature-edge refinement meshing variants: {CASE}")


if __name__ == "__main__":
    run()
