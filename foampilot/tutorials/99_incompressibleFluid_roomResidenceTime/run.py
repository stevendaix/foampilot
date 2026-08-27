"""OpenFOAM 13 incompressibleFluid/roomResidenceTime through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver
from foampilot import Meshing

REFERENCE = Path("/opt/openfoam13/tutorials/incompressibleFluid/roomResidenceTime")


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "incompressibleFluid"
    solver.transient = False
    solver.turbulence_model = "kEpsilon"
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    for source in (REFERENCE / "system").rglob("*"):
        if source.is_file():
            solver.system.import_reference_file(
                source, filename=source.relative_to(REFERENCE / "system")
            )
    for source in (REFERENCE / "constant").rglob("*"):
        if source.is_file():
            solver.constant.import_reference_file(
                source, filename=source.relative_to(REFERENCE / "constant")
            )
    for source in (REFERENCE / "0").rglob("*"):
        if source.is_file():
            solver.fields_manager.import_reference_field(
                source, case_path, field_name=source.relative_to(REFERENCE / "0")
            )

    Meshing(case_path, mesher="blockMesh").mesher.import_reference_dict(
        REFERENCE / "system" / "blockMeshDict"
    )
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_simulation(nb_proc=1, log_filename="log.incompressibleFluid")
    solver.run_command(
        [
            "foamPostProcess", "-solver", "incompressibleFluid", "-latestTime",
            "-func", "age(diffusion=true)",
        ],
        log_filename="log.age",
    )
    solver.run_command(
        ["foamPostProcess", "-func", "probes1", "-latestTime"],
        log_filename="log.probes1",
    )
    solver.run_command(
        ["foamPostProcess", "-func", "probes2", "-latestTime"],
        log_filename="log.probes2",
    )
    solver.run_command(
        [
            "foamPostProcess", "-latestTime", "-func",
            "patchFlowRate(name=inletFlowRate,patch=inlet)", "-latestTime",
        ],
        log_filename="log.patchFlowRate",
    )


if __name__ == "__main__":
    main()
