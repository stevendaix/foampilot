from pathlib import Path

from foampilot import Solver


def test_explicit_solver_name_is_preserved(tmp_path: Path):
    solver = Solver(tmp_path, solver_name="marineFoam")
    assert solver.solver_name == "marineFoam"

    solver.is_vof = True
    solver.with_moving_mesh = True
    assert solver.solver_name == "marineFoam"


def test_propeller_solver_name_is_supported(tmp_path: Path):
    solver = Solver(tmp_path, solver_name="rhoSimpleFoam")
    assert solver.solver_name == "rhoSimpleFoam"


def test_default_vof_solver_remains_foundation_module(tmp_path: Path):
    solver = Solver(tmp_path)
    solver.is_vof = True
    assert solver.solver_name == "incompressibleVoF"
