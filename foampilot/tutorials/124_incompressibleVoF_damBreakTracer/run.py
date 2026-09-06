"""OpenFOAM 13 incompressibleVoF/damBreakTracer through FoamPilot only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from foampilot.solver import Solver

BASE = Path("/opt/openfoam13/tutorials/incompressibleVoF")
LAMINAR = BASE / "damBreakLaminar"
DAMBREAK = BASE / "damBreak"
REFERENCE = BASE / "damBreakTracer"


def import_tree(solver: Solver, source_root: Path, destination: Path) -> None:
    for source in source_root.rglob("*"):
        if not source.is_file() or source.name in {"Allrun", "Allclean"}:
            continue
        relative = source.relative_to(source_root)
        target_relative = Path(*[
            part[:-5] if part.endswith(".orig") else part
            for part in relative.parts
        ])
        if target_relative.parts and target_relative.parts[0] == "0":
            solver.fields_manager.import_reference_field(
                source, destination, field_name=target_relative.relative_to("0")
            )
        else:
            solver.import_reference_asset(source, destination / target_relative)


def main() -> None:
    case_path = Path.cwd()
    solver = Solver(case_path)
    solver.solver_name = "incompressibleVoF"
    solver.setup_case()
    import_tree(solver, LAMINAR, case_path)
    # Reproduce the damBreak merge (RAS turbulence and its differential fields).
    for source in (DAMBREAK / "0").glob("*"):
        solver.fields_manager.import_reference_field(source, case_path, field_name=source.name)
    solver.import_reference_asset(DAMBREAK / "constant/fvModels", case_path / "constant/fvModels")
    solver.system.update_dictionary_entries("constant/momentumTransport", {"simulationType": "RAS"})
    solver.system.merge_reference_dictionary(
        "constant/momentumTransport", DAMBREAK / "constant/momentumTransport.orig", blocks=["RAS"]
    )
    solver.system.merge_reference_dictionary(
        "system/fvSchemes", DAMBREAK / "system/fvSchemes.orig", blocks=["divSchemes"]
    )
    # Add tracer fields and the tracer-specific dictionary overlays.
    for source in (REFERENCE / "0").glob("*"):
        field_name = source.name[:-5] if source.name.endswith(".orig") else source.name
        solver.fields_manager.import_reference_field(source, case_path, field_name=field_name)
    solver.system.merge_reference_dictionary(
        "system/fvSchemes", REFERENCE / "system/fvSchemes.orig", blocks=["divSchemes"]
    )
    solver.system.merge_reference_dictionary(
        "system/fvSolution", REFERENCE / "system/fvSolution.orig", blocks=["solvers"]
    )
    solver.import_reference_asset(REFERENCE / "system/functions", case_path / "system/functions")
    solver.system.merge_reference_dictionary(
        "system/setFieldsDict", REFERENCE / "system/setFieldsDict.orig", blocks=["defaultValues", "zones"]
    )
    solver.run_command(["blockMesh"], log_filename="log.blockMesh")
    solver.run_command(["setFields"], log_filename="log.setFields")
    solver.run_simulation(log_filename="log.incompressibleVoF")


if __name__ == "__main__":
    main()
