import importlib.util
import sys
from pathlib import Path

import pytest


path = Path(__file__).parents[1] / "src/foampilot/solver/marine_forces.py"
spec = importlib.util.spec_from_file_location("marine_forces_under_test", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_propeller_force_model_uses_open_water_scaling():
    model = module.PropellerForceModel(
        rho=1000.0, diameter=0.2, rpm=600.0, kt=0.1, kq=0.02
    )
    assert model.revolutions_per_second == 10.0
    assert model.thrust == pytest.approx(0.1 * 1000.0 * 100.0 * 0.2**4)
    assert model.torque == pytest.approx(0.02 * 1000.0 * 100.0 * 0.2**5)


def test_rudder_force_model_computes_side_force_and_yaw_moment():
    model = module.RudderForceModel(
        rho=1000.0,
        area=0.04,
        lift_coefficient=0.8,
        inflow_speed=2.0,
        angle_deg=15.0,
        moment_arm=1.5,
    )
    assert model.side_force == pytest.approx(64.0)
    assert model.yaw_moment == pytest.approx(96.0)


def test_force_models_reject_invalid_inputs():
    with pytest.raises(ValueError, match="diameter"):
        module.PropellerForceModel(1000, 0, 600, 0.1, 0.01).validate()
    with pytest.raises(ValueError, match="inflow_speed"):
        module.RudderForceModel(1000, 1, 0.8, -1, 15, 1).validate()
