"""Regression tests for ConstantDirectory writing via core/dictionaries."""
import pytest
from pathlib import Path
from foampilot.constant.constantDirectory import ConstantDirectory

class MockFieldsManager:
    """Minimal mock fields manager for testing."""
    def __init__(self):
        self.fields = {}

    def get_field_names(self):
        return list(self.fields.keys())

class MockConfig:
    """Minimal mock solver config for testing."""
    def __init__(self, **kwargs):
        self.is_compressible = kwargs.get("is_compressible", False)
        self.use_solver_keyword = kwargs.get("use_solver_keyword", False)
        self.is_vof = kwargs.get("is_vof", False)
        self.requires_gravity = kwargs.get("requires_gravity", False)
        self.requires_energy = kwargs.get("requires_energy", False)
        self.writes_pRef = kwargs.get("writes_pRef", True)

    def get_energy_variable(self):
        return "T"

class MockSolver:
    """Minimal mock solver for testing."""
    def __init__(self, case_path):
        self.case_path = case_path
        self.is_vof = False
        self.compressible = False
        self.with_gravity = False
        self.fields_manager = MockFieldsManager()
        self.energy_activated = False
        self.config = MockConfig()

    def get_turbulence_configuration(self):
        return ("RAS", "kEpsilon")

def test_transport_properties_write_regression(tmp_path: Path):
    """Verify that transportProperties is written correctly."""
    solver = MockSolver(str(tmp_path / "test_case"))
    const_dir = ConstantDirectory(solver)

    const_dir._transportProperties.attributes["transportModel"] = "Newtonian"
    const_dir._transportProperties.attributes["nu"] = 1e-06

    const_dir.write()

    output_file = tmp_path / "test_case" / "constant" / "transportProperties"
    assert output_file.exists(), "transportProperties not created"

    content = output_file.read_text()
    assert "FoamFile" in content
    assert "transportModel" in content
    assert "Newtonian" in content
    assert "nu" in content

def test_turbulence_properties_write(tmp_path: Path):
    """Verify turbulenceProperties is written correctly."""
    solver = MockSolver(str(tmp_path / "test_case"))
    solver.compressible = False
    solver.config = MockConfig(is_compressible=False, use_solver_keyword=False)
    const_dir = ConstantDirectory(solver)

    const_dir.write()

    output_file = tmp_path / "test_case" / "constant" / "turbulenceProperties"
    assert output_file.exists()
    content = output_file.read_text()
    assert "FoamFile" in content
    assert "RAS" in content or "kEpsilon" in content

def test_physical_properties_write_compressible(tmp_path: Path):
    """Verify physicalProperties is written for compressible solvers."""
    solver = MockSolver(str(tmp_path / "test_case"))
    solver.compressible = True
    solver.config = MockConfig(is_compressible=True, use_solver_keyword=False)
    const_dir = ConstantDirectory(solver)

    const_dir.write()

    output_file = tmp_path / "test_case" / "constant" / "physicalProperties"
    assert output_file.exists()
    content = output_file.read_text()
    assert "FoamFile" in content
    assert "thermoType" in content
    assert "mixture" in content

def test_momentum_transfer_compressible(tmp_path: Path):
    """Verify momentumTransfer is written for compressible solvers."""
    solver = MockSolver(str(tmp_path / "test_case"))
    solver.compressible = True
    solver.config = MockConfig(is_compressible=True, use_solver_keyword=False)
    const_dir = ConstantDirectory(solver)

    const_dir.write()

    output_file = tmp_path / "test_case" / "constant" / "momentumTransport"
    assert output_file.exists()
    content = output_file.read_text()
    assert "FoamFile" in content
    assert "simulationType" in content

def test_gravity_write(tmp_path: Path):
    """Verify g file is written when gravity is enabled."""
    solver = MockSolver(str(tmp_path / "test_case"))
    solver.with_gravity = True
    solver.config = MockConfig(requires_gravity=True)
    const_dir = ConstantDirectory(solver)

    const_dir.write()

    output_file = tmp_path / "test_case" / "constant" / "g"
    assert output_file.exists()
    content = output_file.read_text()
    assert "FoamFile" in content
    assert "uniformDimensionedVectorField" in content

def test_prefs_write(tmp_path: Path):
    """Verify pRef file is written."""
    solver = MockSolver(str(tmp_path / "test_case"))
    solver.config = MockConfig(writes_pRef=True)
    const_dir = ConstantDirectory(solver)

    const_dir.write()

    output_file = tmp_path / "test_case" / "constant" / "pRef"
    assert output_file.exists()
    content = output_file.read_text()
    assert "FoamFile" in content
    assert "uniformDimensionedScalarField" in content

def test_radiation_write(tmp_path: Path):
    """Verify radiation files are written when enabled."""
    solver = MockSolver(str(tmp_path / "test_case"))
    solver.config = MockConfig()
    const_dir = ConstantDirectory(solver, with_radiation=True)

    const_dir.write()

    rad_file = tmp_path / "test_case" / "constant" / "radiationProperties"
    assert rad_file.exists()
    content = rad_file.read_text()
    assert "FoamFile" in content
    assert "radiationModel" in content

    fvmodels_file = tmp_path / "test_case" / "constant" / "fvModels"
    assert fvmodels_file.exists()

def test_vof_configuration(tmp_path: Path):
    """Verify VoF configuration writes phase-specific files."""
    solver = MockSolver(str(tmp_path / "test_case"))
    solver.is_vof = True
    solver.config = MockConfig(is_vof=True, requires_energy=False, writes_pRef=False)
    const_dir = ConstantDirectory(solver)
    const_dir.configure_vof(
        phases=["water", "air"],
        sigma=0.0728,
        phase_properties={
            "water": {"nu": 1e-6, "rho": 1000},
            "air": {"nu": 1e-5, "rho": 1}
        }
    )

    const_dir.write()

    phase_props = tmp_path / "test_case" / "constant" / "phaseProperties"
    assert phase_props.exists()

    pp_water = tmp_path / "test_case" / "constant" / "physicalProperties.water"
    assert pp_water.exists()

    pp_air = tmp_path / "test_case" / "constant" / "physicalProperties.air"
    assert pp_air.exists()

    mt_file = tmp_path / "test_case" / "constant" / "momentumTransport"
    assert mt_file.exists()
