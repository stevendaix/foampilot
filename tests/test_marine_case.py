import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "src/foampilot/solver/marine_case.py"
SPEC = importlib.util.spec_from_file_location("marine_case_under_test", MODULE_PATH)
marine_case = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = marine_case
assert SPEC.loader is not None
SPEC.loader.exec_module(marine_case)
MarineCaseConfig = marine_case.MarineCaseConfig


def make_case(tmp_path: Path, mode: str = "dtc_moving") -> Path:
    (tmp_path / "system").mkdir()
    (tmp_path / "constant").mkdir()
    (tmp_path / "system" / "controlDict").write_text(
        "application marineFoam;\nsolver incompressibleVoF;\n", encoding="utf-8"
    )
    (tmp_path / "constant" / "marineProperties").write_text(
        f"mode {mode};\n", encoding="utf-8"
    )
    for name in ("fvSchemes", "fvSolution"):
        (tmp_path / "system" / name).touch()
    (tmp_path / "constant" / "g").touch()
    return tmp_path


def test_dtc_case_requires_dynamic_mesh(tmp_path):
    config = MarineCaseConfig.from_case(make_case(tmp_path))
    assert config.mode == "dtc_moving"
    with pytest.raises(FileNotFoundError, match="dynamicMeshDict"):
        config.validate_files()


def test_propeller_case_requires_mrf_properties(tmp_path):
    config = MarineCaseConfig.from_case(make_case(tmp_path, "propeller_mrf"))
    assert config.solver == "incompressibleVoF"
    with pytest.raises(FileNotFoundError, match="MRFProperties"):
        config.validate_files()
    (tmp_path / "constant" / "MRFProperties").touch()
    with pytest.raises(FileNotFoundError, match="fvModels"):
        config.validate_files()


def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported marine mode"):
        MarineCaseConfig.from_case(make_case(tmp_path, "overset_magic"))


def test_legacy_overset_is_rejected_for_foundation13(tmp_path):
    case = make_case(tmp_path)
    (case / "constant" / "dynamicMeshDict").write_text(
        "dynamicFvMesh dynamicOversetFvMesh;\n", encoding="utf-8"
    )
    config = MarineCaseConfig.from_case(case)
    with pytest.raises(ValueError, match="not native Foundation 13"):
        config.validate_foundation13()
