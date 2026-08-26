from pathlib import Path
import json
import pytest

from foampilot.openfoam13 import PhysicsConfig, check_openfoam13_case


def test_default_catalog_has_all_requested_repositories():
    names = {m.name for m in PhysicsConfig().modules}
    assert names == {
        "boundaryConditions",
        "MachineLearningTurbulenceModels",
        "urbanMicroclimateFoam-tutorials",
        "adaptive-mesh-refinement",
        "PythonFOAM",
    }


def test_ml_requires_explicit_optional_module():
    cfg = PhysicsConfig(turbulence={"model": "gammaRST"})
    assert any("ML turbulence" in e for e in cfg.validate())


def test_writes_non_destructive_support_files(tmp_path: Path):
    (tmp_path / "system").mkdir()
    (tmp_path / "constant").mkdir()
    (tmp_path / "0").mkdir()
    (tmp_path / "system/controlDict").write_text("application simpleFoam;\n")
    (tmp_path / "constant/transportProperties").write_text("nu [0 2 -1 0 0 0 0] 1e-05;\n")
    cfg = PhysicsConfig(
        boundary_conditions={"inlet": {"type": "turbulentInletTable"}},
        adaptive_mesh={"sourceField": "curl(U)", "lowerRefinementLevel": 0.1},
        urban={"referenceHeight": 10.0},
        pythonfoam={"enabled": False},
    )
    written = cfg.write_support_files(tmp_path)
    assert (tmp_path / "system/dynamicMeshDict").exists()
    assert (tmp_path / "foampilotPhysics.json").exists()
    assert "dynamicMeshDict" in {p.name for p in written}
    assert (tmp_path / "constant/foampilotBoundaryConditions.json").exists()
    assert not check_openfoam13_case(tmp_path)
    assert "field refVal;" in (tmp_path / "system/dynamicMeshDict").read_text()
    assert json.loads((tmp_path / "foampilotPhysics.json").read_text())["openfoam"]["version"] == 13


def test_preflight_catches_missing_nu(tmp_path: Path):
    (tmp_path / "system").mkdir()
    (tmp_path / "constant").mkdir()
    (tmp_path / "0").mkdir()
    (tmp_path / "system/controlDict").write_text("application simpleFoam;\n")
    (tmp_path / "constant/transportProperties").write_text("FoamFile {}\n")
    assert any("explicitly define nu" in e for e in check_openfoam13_case(tmp_path))
