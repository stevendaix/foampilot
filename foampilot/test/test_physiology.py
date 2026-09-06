from pathlib import Path
import sys
import types

import numpy as np
import pytest

# Load the focused package tree without importing foampilot's eager optional
# top-level integrations (geometry, visualization, chemistry, ...).
_SRC = Path(__file__).parents[1] / "src"
foampilot_pkg = types.ModuleType("foampilot")
foampilot_pkg.__path__ = [str(_SRC / "foampilot")]
sys.modules.setdefault("foampilot", foampilot_pkg)
utilities_pkg = types.ModuleType("foampilot.utilities")
utilities_pkg.__path__ = [str(_SRC / "foampilot" / "utilities")]
sys.modules.setdefault("foampilot.utilities", utilities_pkg)
postprocess_pkg = types.ModuleType("foampilot.postprocess")
postprocess_pkg.__path__ = [str(_SRC / "foampilot" / "postprocess")]
sys.modules.setdefault("foampilot.postprocess", postprocess_pkg)

from foampilot.physiology import JOS3, JOS3NodeCoupler, SurfaceMapping
from foampilot.postprocess.openfoam_external_coupled import OpenFOAM13TemperatureProvider
from foampilot.physiology.jos3 import thermoregulation as threg
from foampilot.physiology.units import as_magnitude


def test_surface_mapping_rejects_invalid_values():
    with pytest.raises(ValueError):
        SurfaceMapping([0], [0.0])
    with pytest.raises(ValueError):
        SurfaceMapping([17], [1.0])
    with pytest.raises(ValueError):
        SurfaceMapping([0], [np.nan])


def test_surface_mapping_csv_rejects_incomplete_row(tmp_path: Path):
    path = tmp_path / "mapping.csv"
    path.write_text("zone_id,area_m2\n0,1\n1,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplète"):
        SurfaceMapping.from_csv(path)


def test_heat_resistances_contract():
    result = threg.heat_resistances()
    assert len(result) == 8
    assert all(np.asarray(value).shape == (17,) for value in result)
    assert all(np.all(np.isfinite(value)) for value in result)


def test_openfoam13_provider_converts_kelvin_and_validates_area(tmp_path: Path):
    provider = OpenFOAM13TemperatureProvider(
        tmp_path, timeout=0.2, poll_interval=0.001, temperature_unit="K"
    )
    provider.data_out_path.write_text("0.5 300 0 10\n", encoding="utf-8")
    fields = provider.read_nodal_fields()
    assert fields["surface_temperature"][0] == pytest.approx(26.85)
    with pytest.raises(ValueError):
        provider.data_out_path.write_text("0 300 0 10\n", encoding="utf-8")
        provider._last_data_mtime_ns = 0
        provider.read_nodal_fields()


def test_units_convert_temperature_and_coefficient():
    np.testing.assert_allclose(as_magnitude((300.0, "kelvin"), "degC"), 26.85)
    np.testing.assert_allclose(
        as_magnitude((1.0, "watt / meter ** 2 / kelvin"), "W/m^2/K"), 1.0
    )


def test_jos3_wet_and_posture_validation():
    model = JOS3()
    wet = model.Wet
    assert wet.shape == (17,)
    assert np.all(np.isfinite(wet))
    assert np.all((wet >= 0) & (wet <= 1))
    with pytest.raises(ValueError):
        model.posture = "unknown"


def test_jos3_rejects_wrong_vector_length():
    model = JOS3()
    with pytest.raises(ValueError):
        model.Ta = [20.0, 21.0]
    with pytest.raises(ValueError):
        model.simulate(1, dtime=0)


def test_aggregated_coupling_conserves_zone_power():
    model = JOS3()
    mapping = SurfaceMapping([0, 0, 1], [1.0, 2.0, 1.0])
    coupler = JOS3NodeCoupler(model, mapping)
    exchange = coupler.exchange(
        h=np.array([10.0, 10.0, 5.0]),
        surface_temperature=np.array([30.0, 30.0, 20.0]),
        air_temperature=np.array([20.0, 20.0, 25.0]),
    )
    expected = exchange.body_flux * mapping.areas
    assert exchange.zone_power[0] == pytest.approx(expected[:2].sum())
    assert exchange.zone_power[1] == pytest.approx(expected[2])
