"""Regression tests for SystemDirectory writing via core/dictionaries."""
import pytest
from pathlib import Path
from foampilot.solver.solver import Solver


def test_control_dict_write_regression(tmp_path: Path):
    """Verify that controlDict is written correctly via DictionaryWriter."""
    case_path = tmp_path / "test_case"
    solver = Solver(case_path)
    solver.transient = False
    solver.simulation_type = "incompressible"
    solver.algorithm = "SIMPLE"

    system_dir = solver.system.write()

    control_dict = system_dir / "controlDict"
    assert control_dict.exists(), "controlDict not created"

    content = control_dict.read_text()
    assert "FoamFile" in content
    assert "application" in content or "solver" in content
    assert "startFrom" in content
    assert "incompressibleFluid" in content


def test_fv_schemes_write(tmp_path: Path):
    """Verify fvSchemes is written correctly via DictionaryWriter."""
    case_path = tmp_path / "test_case"
    solver = Solver(case_path)
    solver.transient = False
    solver.simulation_type = "incompressible"

    system_dir = solver.system.write()

    fv_schemes = system_dir / "fvSchemes"
    assert fv_schemes.exists()
    content = fv_schemes.read_text()
    assert "FoamFile" in content
    assert "ddtSchemes" in content
    assert "divSchemes" in content


def test_fv_solution_write(tmp_path: Path):
    """Verify fvSolution is written correctly via DictionaryWriter."""
    case_path = tmp_path / "test_case"
    solver = Solver(case_path)
    solver.transient = False
    solver.simulation_type = "incompressible"
    solver.algorithm = "SIMPLE"

    system_dir = solver.system.write()

    fv_solution = system_dir / "fvSolution"
    assert fv_solution.exists()
    content = fv_solution.read_text()
    assert "FoamFile" in content
    assert "solvers" in content
    assert "SIMPLE" in content


def test_decompose_par_dict_write(tmp_path: Path):
    """Verify decomposeParDict is written when enabled."""
    case_path = tmp_path / "test_case"
    solver = Solver(case_path)
    solver.system.ensure_decomposeParDict(4)

    system_dir = solver.system.write()

    decompose_par_dict = system_dir / "decomposeParDict"
    assert decompose_par_dict.exists()
    content = decompose_par_dict.read_text()
    assert "FoamFile" in content
    assert "numberOfSubdomains" in content


def test_system_directory_incompressible_simple(tmp_path: Path):
    """Verify incompressible SIMPLE case writes all system files."""
    case_path = tmp_path / "test_incompressible_simple"
    solver = Solver(case_path)
    solver.transient = False
    solver.simulation_type = "incompressible"
    solver.algorithm = "SIMPLE"

    system_dir = solver.system.write()

    assert (system_dir / "controlDict").exists()
    assert (system_dir / "fvSchemes").exists()
    assert (system_dir / "fvSolution").exists()


def test_system_directory_transient_pimple(tmp_path: Path):
    """Verify transient PIMPLE case writes all system files."""
    case_path = tmp_path / "test_transient_pimple"
    solver = Solver(case_path)
    solver.transient = True
    solver.simulation_type = "incompressible"
    solver.algorithm = "PIMPLE"

    system_dir = solver.system.write()

    assert (system_dir / "controlDict").exists()
    assert (system_dir / "fvSchemes").exists()
    assert (system_dir / "fvSolution").exists()

    content = (system_dir / "fvSolution").read_text()
    assert "PIMPLE" in content


def test_system_directory_compressible_energy(tmp_path: Path):
    """Verify compressible energy case writes all system files."""
    case_path = tmp_path / "test_compressible_energy"
    solver = Solver(case_path)
    solver.compressible = True
    solver.energy_activated = True
    solver.simulation_type = "compressible"
    solver.algorithm = "SIMPLE"

    system_dir = solver.system.write()

    assert (system_dir / "controlDict").exists()
    assert (system_dir / "fvSchemes").exists()
    assert (system_dir / "fvSolution").exists()

    content = (system_dir / "fvSolution").read_text()
    assert "solvers" in content


def test_system_directory_boussinesq_transient(tmp_path: Path):
    """Verify Boussinesq transient case writes all system files."""
    case_path = tmp_path / "test_boussinesq_transient"
    solver = Solver(case_path)
    solver.transient = True
    solver.simulation_type = "boussinesq"
    solver.algorithm = "PIMPLE"
    solver.energy_activated = True

    system_dir = solver.system.write()

    assert (system_dir / "controlDict").exists()
    assert (system_dir / "fvSchemes").exists()
    assert (system_dir / "fvSolution").exists()

    content = (system_dir / "fvSchemes").read_text()
    assert "div(phi,T)" in content
