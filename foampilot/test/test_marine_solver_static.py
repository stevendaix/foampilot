from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_marine_driver_uses_foundation_module_selection():
    source = (ROOT / "openfoam13/marineFoam/marineFoam.C").read_text()
    assert 'args.optionReadIfPresent("solver", solverName);' in source
    assert 'solverName = "incompressibleVoF";' in source
    assert "solver::load(solverName);" in source
    assert "pimpleSingleRegionControl" in source


def test_python_wrapper_accepts_explicit_solver():
    source = (ROOT / "foampilot/src/foampilot/solver/solver.py").read_text()
    assert "solver_name: str | None = None" in source
    assert "self._requested_solver = solver_name" in source
    assert "if self._requested_solver:" in source


def test_base_solver_runs_marine_executable_directly():
    source = (ROOT / "foampilot/src/foampilot/solver/base_solver.py").read_text()
    assert '"marineFoam"' in source
    assert "[self.solver_name]" in source


def test_marine_apis_are_publicly_exported():
    mesh = (ROOT / "foampilot/src/foampilot/mesh/__init__.py").read_text()
    solver = (ROOT / "foampilot/src/foampilot/solver/__init__.py").read_text()
    assert "write_six_dof_dynamic_mesh_dict" in mesh
    assert "MarineMRFZone" in mesh
    assert "marine_overset" in mesh
    for symbol in ("OversetZone", "build_zone_id", "write_zone_id_field", "build_donor_stencils"):
        assert symbol in mesh
    assert "MarineCaseConfig" in solver
    assert "write_actuation_disk" in solver


def test_overset_port_contract_is_explicit():
    contract = (ROOT / "openfoam13/marineFoam/OVERSET_PORT_CONTRACT.md").read_text()
    for marker in ("zoneID", "calculated", "interpolated", "hole", "solveurs asymétriques", "MPI"):
        assert marker in contract
    assert "ne supporte pas encore l’overset matriciel" in contract


def test_overset_boundary_is_documented():
    roadmap = (ROOT / "openfoam13/marineFoam/OVERSET_ROADMAP.md").read_text()
    makefiles = (ROOT / "openfoam13/marineFoam/Make/files").read_text()
    assert "dynamicOversetFvMesh" in roadmap
    assert "Portage overset fidèle" in roadmap
    assert "marineFoam.C" in makefiles
