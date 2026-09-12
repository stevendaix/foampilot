import importlib.util
import sys
from pathlib import Path

import pytest


path = Path(__file__).parents[1] / "src/foampilot/solver/marine_controls.py"
spec = importlib.util.spec_from_file_location("marine_controls_under_test", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_controls_are_written_and_validated(tmp_path):
    output = module.write_marine_controls(
        tmp_path,
        propeller=module.PropellerCommand(rpm=1200, diameter=0.25),
        rudder=module.RudderCommand(angle_deg=15),
    )
    text = output.read_text()
    assert "rpm 1200;" in text
    assert "diameter 0.25;" in text
    assert "angleDeg 15;" in text


def test_invalid_propeller_is_rejected():
    with pytest.raises(ValueError, match="diameter"):
        module.PropellerCommand(rpm=100, diameter=0).validate()
    with pytest.raises(ValueError, match="axis"):
        module.PropellerCommand(rpm=100, diameter=0.2, axis=(0, 0, 0)).validate()


def test_invalid_rudder_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        module.RudderCommand(angle_deg=40, max_angle_deg=35).validate()
    with pytest.raises(ValueError, match="rate_limit"):
        module.RudderCommand(angle_deg=5, rate_limit_deg_s=0).validate()
