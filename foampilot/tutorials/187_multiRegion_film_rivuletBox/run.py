"""OpenFOAM 13 multiRegion/film/rivuletBox via FoamPilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/multiRegion/film/rivuletBox")
OF13_BIN = Path("/opt/openfoam13/platforms/linux64GccDPInt32Opt/bin")
OF13_TOOLS = Path("/opt/openfoam13/bin")
NPROCS = 8


def import_reference_case(solver: Solver, case_path: Path) -> None:
    for source in (REFERENCE / "0").rglob("*"):
        if source.is_file():
            relative = source.relative_to(REFERENCE / "0")
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=str(relative)
            )
    for root in ("constant", "system"):
        for source in (REFERENCE / root).rglob("*"):
            if source.is_file():
                solver.import_reference_asset(
                    source, case_path / source.relative_to(REFERENCE)
                )


def command(solver: Solver, executable: str, *args: str, tag: str) -> None:
    solver.run_command(
        [str(OF13_BIN / executable), *args], log_filename=f"log.{tag}"
    )


def parallel_command(solver: Solver, executable: str, *args: str, tag: str) -> None:
    solver.run_command(
        [
            "/usr/bin/mpirun",
            "--oversubscribe",
            "-np",
            str(NPROCS),
            str(OF13_BIN / executable),
            "-parallel",
            *args,
        ],
        log_filename=f"log.{tag}",
    )


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "foamMultiRun"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    command(solver, "blockMesh", "-region", "box", tag="blockMesh.box")
    command(
        solver,
        "extrudeToRegionMesh",
        "-dict",
        "system/extrudeToRegionMeshDict.panel",
        "-region",
        "box",
        tag="extrudeToRegionMesh.panel",
    )
    command(
        solver,
        "extrudeToRegionMesh",
        "-dict",
        "system/extrudeToRegionMeshDict.film",
        "-region",
        "panel",
        tag="extrudeToRegionMesh.film",
    )
    command(
        solver,
        "foamDictionary",
        "-set",
        "entry0/film/type=mappedExtrudedWall, entry0/film/neighbourRegion=film, entry0/film/neighbourPatch=surface, entry0/film/isExtrudedRegion=no",
        "constant/box/polyMesh/boundary",
        tag="foamDictionary.box.boundary",
    )
    command(
        solver,
        "foamDictionary",
        "-set",
        "entry0/surface/type=mappedFilmSurface, entry0/surface/neighbourRegion=box, entry0/surface/neighbourPatch=film, entry0/surface/isExtrudedRegion=yes",
        "constant/film/polyMesh/boundary",
        tag="foamDictionary.film.boundary",
    )
    solver.run_command(
        [str(OF13_TOOLS / "paraFoam"), "-touchAll"],
        log_filename="log.paraFoam.touchAll",
    )
    command(
        solver,
        "decomposePar",
        "-allRegions",
        tag="decomposePar.allRegions",
    )
    parallel_command(solver, "foamMultiRun", tag="foamMultiRun.parallel")
    command(
        solver,
        "reconstructPar",
        "-allRegions",
        tag="reconstructPar.allRegions",
    )


if __name__ == "__main__":
    main()
