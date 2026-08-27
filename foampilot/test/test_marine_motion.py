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
    return Path(case_path) / "constant" / "dynamicMeshDict"


OPS.write_dynamic_mesh_dict = fake_writer
_original_modules = {name: sys.modules.get(name) for name in ("foampilot", "foampilot.mesh", "foampilot.mesh.ops")}
sys.modules.update({"foampilot": PACKAGE, "foampilot.mesh": MESH, "foampilot.mesh.ops": OPS})

path = ROOT / "foampilot/src/foampilot/mesh/marine_motion.py"
spec = importlib.util.spec_from_file_location("marine_motion_under_test", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
for _name, _value in _original_modules.items():
    if _value is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _value


def test_six_dof_forwards_all_foundation_joints(tmp_path):
    CALLS.clear()
    module.write_six_dof_dynamic_mesh_dict(
        tmp_path,
        mass=10.0,
        centre_of_mass=(0.0, 0.0, 0.0),
        inertia=(1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        inner_distance=0.2,
        outer_distance=1.0,
    )
    assert CALLS[0][1]["joints"] == module.FOUNDATION13_JOINTS
    assert CALLS[0][1]["joints"] == ("Px", "Py", "Pz", "Rx", "Ry", "Rz")


def test_six_dof_rejects_invalid_mass_and_distances(tmp_path):
    kwargs = dict(
        centre_of_mass=(0.0, 0.0, 0.0),
        inertia=(1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        inner_distance=0.2,
        outer_distance=1.0,
    )
    with pytest.raises(ValueError, match="mass"):
        module.write_six_dof_dynamic_mesh_dict(tmp_path, mass=0.0, **kwargs)
    invalid_distance = {**kwargs, "outer_distance": 0.1}
    with pytest.raises(ValueError, match="outer_distance"):
        module.write_six_dof_dynamic_mesh_dict(tmp_path, mass=1.0, **invalid_distance)
