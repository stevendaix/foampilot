from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest


_MODULE_PATH = Path(__file__).parents[1] / "src" / "foampilot" / "coupling" / "external_coupled.py"
_SPEC = spec_from_file_location("external_coupled", _MODULE_PATH)
_MODULE = module_from_spec(_SPEC)
sys.modules["external_coupled"] = _MODULE
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
ExternalCoupledTemperature = _MODULE.ExternalCoupledTemperature
ExternalCouplingTimeout = _MODULE.ExternalCouplingTimeout


def test_read_output_supports_multiple_patches(tmp_path: Path):
    output = tmp_path / "temperature.out"
    output.write_text(
        "# Patch: hot 1 300 0 0\n"
        "0.5 301 10 2\n"
        "# Patch: cold 1 290 0 0\n"
        "0.25 291 -4 1\n",
        encoding="utf-8",
    )

    coupling = ExternalCoupledTemperature(tmp_path, timeout=0.2)
    output_records = coupling._read_output(output)

    assert [(record.patch, record.area) for record in output_records] == [
        ("hot", 1.0),
        ("hot", 0.5),
        ("cold", 1.0),
        ("cold", 0.25),
    ]


def test_send_temperature_mixed_values_creates_input_and_lock(tmp_path: Path):
    coupling = ExternalCoupledTemperature(tmp_path)
    coupling.send_temperature_mixed_values([(300.0, 0.0, 1.0)])

    assert coupling.input_file.read_text(encoding="utf-8") == "300 0 1\n"
    assert coupling.lock_file.exists()


def test_wait_for_openfoam_times_out(tmp_path: Path):
    coupling = ExternalCoupledTemperature(tmp_path, wait_interval=0.01, timeout=0.03)
    with pytest.raises(ExternalCouplingTimeout):
        coupling.wait_for_openfoam()
