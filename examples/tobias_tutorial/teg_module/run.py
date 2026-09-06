"""Tobias Holzmann TEG module variants for OpenFOAM 13."""
from pathlib import Path
import shutil

from foampilot.solver import Solver
from foampilot.utilities import OpenFOAMDictAddFile

ROOT = Path(__file__).resolve().parent
VARIANTS = ("testDevice", "optimizedDevice")


def write_case(variant: str) -> Path:
    variant_root = ROOT / variant
    templates = variant_root / "templates"
    case = variant_root / "case"
    if case.exists():
        shutil.rmtree(case)
    writer = OpenFOAMDictAddFile(object_name=f"tobias_teg_{variant}")
    for source in sorted(templates.rglob("*")):
        if source.is_file():
            relative = source.relative_to(templates)
            writer.write_raw(
                relative.name,
                case,
                source.read_text(encoding="utf-8"),
                folder=str(relative.parent),
            )
    shutil.copytree(variant_root / "cad", case / "cad", dirs_exist_ok=True)
    return case


def run_variant(variant: str) -> None:
    case = write_case(variant)
    background_mesh = case / "cad" / "backgroundMesh.unv"
    if not background_mesh.is_file():
        raise FileNotFoundError(
            f"{variant} requires cad/backgroundMesh.unv. Tobias removes this "
            "large asset from GitHub; download the complete case archive from "
            "Holzmann CFD before running OpenFOAM 13."
        )
    if shutil.which("TEGFoam") is None:
        raise RuntimeError(
            "TEGFoam is not installed in the active OpenFOAM 13 environment. "
            "Compile solverTEGModule first and expose libsolverTEGModule.so."
        )
    solver = Solver(case)
    solver.run_command(["ideasUnvToFoam", "cad/backgroundMesh.unv"], "log.ideasUnvToFoam")
    solver.run_command(["snappyHexMesh", "-overwrite"], "log.snappyHexMesh")
    shutil.copytree(case / "0.orig", case / "0")
    solver.run_parallel(2, log_filename="log.TEGFoam")
    print(f"Validated TEG workflow ({variant}): {case}")


def run() -> None:
    for variant in VARIANTS:
        run_variant(variant)


if __name__ == "__main__":
    run()
