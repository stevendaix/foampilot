import importlib.util
import sys
from math import pi
from pathlib import Path

import pytest


path = Path(__file__).parents[1] / "src/foampilot/solver/marine_actuation_disk.py"
spec = importlib.util.spec_from_file_location("marine_actuation_disk_under_test", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_propeller_is_converted_to_native_actuation_disk(tmp_path):
    source = module.actuation_disk_from_propeller(
        cell_zone="rotor",
        diameter=0.2,
        disk_dir=(1.0, 0.0, 0.0),
        cp=0.1,
        ct=0.5,
        upstream_point=(0.0, 0.0, 0.0),
        phase_name="water",
    )
    assert source.disk_area == pytest.approx(pi * 0.2**2 / 4)
    output = module.write_actuation_disk(tmp_path, source)
    text = output.read_text()
    assert "type            actuationDisk;" in text
    assert "cellZone        rotor;" in text
    assert "phase      water;" in text


def test_actuation_disk_rejects_invalid_coefficients():
    with pytest.raises(ValueError, match="ct"):
        module.ActuationDiskSource("rotor", (1, 0, 0), 0.1, 0, 1, (0, 0, 0)).validate()
    with pytest.raises(ValueError, match="disk_area"):
        module.ActuationDiskSource("rotor", (1, 0, 0), 0.1, 0.5, 0, (0, 0, 0)).validate()
