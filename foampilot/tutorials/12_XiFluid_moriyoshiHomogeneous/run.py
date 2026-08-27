"""OpenFOAM 13 XiFluid/moriyoshiHomogeneous through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot import Meshing
from foampilot.solver import Solver

REFERENCE = Path("/opt/openfoam13/tutorials/XiFluid/moriyoshiHomogeneous/moriyoshiHomogeneous")


def prepare_case(case_path: Path, hydrogen: bool = False) -> Solver:
    solver = Solver(case_path)
    solver.solver_name = "XiFluid"
    solver.transient = True
    solver.setup_case()
    solver.system.write()
    solver.constant.write()

    # Import only complete OF13 reference dictionaries through FoamPilot.
    for source in (REFERENCE / "system").iterdir():
        if source.is_file() and source.name not in {"controlDict.orig", "blockMeshDict"}:
            solver.system.import_reference_file(source)
    solver.system.import_reference_file(REFERENCE / "system" / "controlDict.orig", "controlDict")
    solver.system.import_reference_file(REFERENCE / "system" / "blockMeshDict", "blockMeshDict")

    for source in (REFERENCE / "constant").iterdir():
        if not source.is_file():
            continue
        if source.name in {"physicalProperties", "combustionPropertiesInclude"}:
            if hydrogen:
                solver.constant.import_reference_file(source.with_name(source.name + ".hydrogen"), source.name)
            else:
                solver.constant.import_reference_file(source, source.name)
        elif not hydrogen and source.name.endswith(".hydrogen"):
            continue
        elif hydrogen and source.name.endswith(".hydrogen"):
            continue
        else:
            solver.constant.import_reference_file(source)

    for source in (REFERENCE / "0").iterdir():
        if source.is_file():
            solver.fields_manager.import_reference_field(source, case_path)

    mesh = Meshing(case_path, mesher="blockMesh")
    mesh.mesher.import_reference_dict(REFERENCE / "system" / "blockMeshDict")
    mesh.mesher.run()

    if not hydrogen:
        solver.system.update_dictionary_entries(
            case_path / "system" / "controlDict",
            {"deltaT": "1e-05", "endTime": "0.015", "writeInterval": "50"},
        )
    return solver


def main() -> None:
    root = Path.cwd()
    propane = prepare_case(root / "moriyoshiHomogeneous", hydrogen=False)
    propane.run_simulation(nb_proc=1, log_filename="log.XiFluid.propane")
    hydrogen = prepare_case(root / "moriyoshiHomogeneousHydrogen", hydrogen=True)
    hydrogen.run_simulation(nb_proc=1, log_filename="log.XiFluid.hydrogen")


if __name__ == "__main__":
    main()
