import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
OPS = types.ModuleType("foampilot.mesh.ops")
MESH = types.ModuleType("foampilot.mesh")
PACKAGE = types.ModuleType("foampilot")
CALLS = []


def fake_writer(case_path, **kwargs):
    CALLS.append((case_path, kwargs))
    return Path(case_path) / "constant" / "MRFProperties"


OPS.write_rotating_zone = fake_writer
_original_modules = {name: sys.modules.get(name) for name in ("foampilot", "foampilot.mesh", "foampilot.mesh.ops")}
sys.modules.update({"foampilot": PACKAGE, "foampilot.mesh": MESH, "foampilot.mesh.ops": OPS})

path = ROOT / "foampilot/src/foampilot/mesh/marine_mrf.py"
spec = importlib.util.spec_from_file_location("marine_mrf_under_test", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
for _name, _value in _original_modules.items():
    if _value is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _value


def make_zone(**overrides):
    values = dict(
        cell_zone="rotor",
        origin=(0.0, 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),
        omega=125.0,
        non_rotating_patches=("hull", "inlet", "outlet"),
    )
    values.update(overrides)
    return module.MarineMRFZone(**values)


def test_mrf_zone_is_forwarded_to_openfoam_writer(tmp_path):
    CALLS.clear()
    module.write_marine_mrf(tmp_path, make_zone())
    assert CALLS[0][1]["cell_zone"] == "rotor"
    assert CALLS[0][1]["omega"] == 125.0


def test_mrf_zone_rejects_zero_axis_or_omega():
    with pytest.raises(ValueError, match="axis"):
        make_zone(axis=(0.0, 0.0, 0.0)).validate()
    with pytest.raises(ValueError, match="omega"):
        make_zone(omega=0.0).validate()
