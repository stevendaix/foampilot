"""OpenFOAM 13 multiRegion/CHT/shellAndTubeHeatExchanger via FoamPilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/multiRegion/CHT/shellAndTubeHeatExchanger")
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
                destination = case_path / source.relative_to(REFERENCE)
                solver.import_reference_asset(source, destination)
    solver.import_reference_asset(
        REFERENCE / "system/snappyHexMeshDict.orig",
        case_path / "system/snappyHexMeshDict",
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
            *args,
        ],
        log_filename=f"log.{tag}",
    )


def foam_dictionary(solver: Solver, entry: str, value: str, tag: str) -> None:
    command(
        solver,
        "foamDictionary",
        "system/snappyHexMeshDict",
        "-entry",
        entry,
        "-set",
        value,
        tag=tag,
    )


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "foamMultiRun"
    solver.transient = True
    solver.setup_case()
    import_reference_case(solver, case_path)

    command(solver, "blockMesh", tag="blockMesh")
    command(solver, "decomposePar", "-copyZero", tag="decomposePar.copyZero")
    foam_dictionary(solver, "castellatedMesh", "on", "foamDictionary.castellated.on")
    foam_dictionary(solver, "snap", "on", "foamDictionary.snap.on")
    foam_dictionary(solver, "addLayers", "off", "foamDictionary.layers.off")
    parallel_command(solver, "snappyHexMesh", tag="snappyHexMesh.base")
    parallel_command(solver, "createBaffles", tag="createBaffles.parallel")
    parallel_command(solver, "splitBaffles", tag="splitBaffles.parallel")
    for proc in range(NPROCS):
        solver.remove_case_asset(f"processor{proc}/constant/polyMesh/pointLevel")
    foam_dictionary(solver, "castellatedMesh", "off", "foamDictionary.castellated.off")
    foam_dictionary(solver, "snap", "off", "foamDictionary.snap.off")
    foam_dictionary(solver, "addLayers", "on", "foamDictionary.layers.on")
    parallel_command(solver, "snappyHexMesh", tag="snappyHexMesh.layers")
    parallel_command(
        solver,
        "splitMeshRegions",
        "-cellZones",
        "-defaultRegionName",
        "solid",
        tag="splitMeshRegions.cellZones",
    )
    parallel_command(solver, "foamMultiRun", tag="foamMultiRun.parallel")
    command(solver, "reconstructPar", "-allRegions", tag="reconstructPar.allRegions")
    solver.run_command(
        [str(OF13_TOOLS / "paraFoam"), "-touchAll"],
        log_filename="log.paraFoam.touchAll",
    )


if __name__ == "__main__":
    main()
