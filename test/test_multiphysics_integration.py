import json
from pathlib import Path

import pytest

import importlib.util
import sys


_MODULE_PATH = Path(__file__).parents[1] / "foampilot" / "src" / "foampilot" / "multiphysics" / "integration.py"
_SPEC = importlib.util.spec_from_file_location("foampilot_multiphysics_integration", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
MultiphysicsConfiguration = _MODULE.MultiphysicsConfiguration
MultiphysicsConfigurationError = _MODULE.MultiphysicsConfigurationError
check_openfoam13 = _MODULE.check_openfoam13


def test_sedifoam_acoustics_configuration_writes_auditable_assets(tmp_path: Path):
    config = MultiphysicsConfiguration(("sedifoam", "libacoustics"))
    manifest, dictionary = config.write_case_assets(tmp_path)

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["openfoam"] == {"distribution": "Foundation", "version": "13"}
    assert data["requiredFields"] == ["U", "p", "alpha", "phi"]
    text = dictionary.read_text(encoding="utf-8")
    assert "openfoamDistribution Foundation;" in text
    assert "openfoamVersion 13;" in text
    assert "modules (sedifoam libacoustics);" in text


def test_dem_backends_are_mutually_exclusive():
    with pytest.raises(MultiphysicsConfigurationError, match="backends DEM alternatifs"):
        MultiphysicsConfiguration(("sedifoam", "openhfdib_dem"))


def test_unknown_and_non_foundation_versions_are_rejected():
    with pytest.raises(MultiphysicsConfigurationError, match="OpenFOAM Foundation 13"):
        MultiphysicsConfiguration(("libacoustics",), openfoam_version="2412")
    with pytest.raises(MultiphysicsConfigurationError, match="Modules inconnus"):
        MultiphysicsConfiguration(("not-a-module",))


def test_openfoam13_is_installed_in_validation_environment():
    assert check_openfoam13() == "OpenFOAM-13"
