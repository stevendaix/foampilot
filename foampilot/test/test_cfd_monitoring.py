"""Unit tests for engineering-oriented CFD monitors."""
from __future__ import annotations

import numpy as np

from foampilot.postprocess.monitoring import CFDMonitor, compute_y_plus, integrate_surface_forces


class FakeMesh:
    def __init__(self, point_data, cell_data=None):
        self.point_data = point_data
        self.cell_data = cell_data or {}

    def find_closest_point(self, point):
        return 0


class FakePostprocessor:
    def __init__(self):
        self.meshes = {
            0: FakeMesh({"p": np.array([100.0, 90.0]), "U": np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])}),
            1: FakeMesh({"p": np.array([95.0, 85.0]), "U": np.array([[6.0, 8.0, 0.0], [0.0, 0.0, 0.0]])}),
        }

    def get_all_time_steps(self):
        return [0, 1]

    def load_time_step(self, step):
        return {"cell": self.meshes[step], "boundaries": {}}


def test_statistics_include_engineering_percentiles_and_rms():
    stats = CFDMonitor.statistics(np.array([3.0, 4.0]))
    assert stats["mean"] == 3.5
    assert stats["p50"] == 3.5
    assert stats["rms"] == np.sqrt(12.5)


def test_track_region_supports_vector_magnitude():
    frame = CFDMonitor(FakePostprocessor()).track_region("U", magnitude=True)
    assert list(frame.index) == [0, 1]
    assert frame.loc[0, "max"] == 5.0
    assert frame.loc[1, "max"] == 10.0


def test_track_point_returns_values_over_time():
    frame = CFDMonitor(FakePostprocessor()).track_point((0.0, 0.0, 0.0), "p")
    assert list(frame["value"]) == [100.0, 95.0]


def test_compute_y_plus_uses_friction_velocity():
    result = compute_y_plus(np.array([0.001]), np.array([4.0]), rho=1.0, kinematic_viscosity=1e-3)
    assert result[0] == 2.0


def test_integrate_surface_forces_returns_lift_and_drag_coefficients():
    result = integrate_surface_forces(
        normals=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        areas=np.array([1.0, 1.0]),
        pressure=np.array([0.0, 10.0]),
        rho=1.0,
        reference_velocity=2.0,
        reference_area=1.0,
    )
    assert result["force_x"] == 10.0
    assert result["Cd"] == 5.0
    assert result["Cl"] == 0.0


def test_integrate_mass_flux_tracks_inflow_and_outflow():
    from foampilot.postprocess.monitoring import integrate_mass_flux

    result = integrate_mass_flux(
        normals=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        areas=np.array([2.0, 3.0]),
        velocity=np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        density=2.0,
    )
    assert result["outflow_mass"] == 8.0
    assert result["inflow_mass"] == 6.0
    assert result["mass_flux"] == 2.0


def test_mass_balance_aggregates_named_patches():
    from foampilot.postprocess.monitoring import mass_balance

    result = mass_balance({
        "inlet": {
            "normals": np.array([[-1.0, 0.0, 0.0]]),
            "areas": np.array([1.0]),
            "velocity": np.array([[2.0, 0.0, 0.0]]),
        },
        "outlet": {
            "normals": np.array([[1.0, 0.0, 0.0]]),
            "areas": np.array([1.0]),
            "velocity": np.array([[2.0, 0.0, 0.0]]),
        },
    })
    assert result["net_mass_flux"] == 0.0
    assert set(result["patches"]) == {"inlet", "outlet"}


def test_engineering_result_is_json_ready():
    from foampilot.postprocess.results import EngineeringResult, ResultMetadata

    result = EngineeringResult(
        metadata=ResultMetadata(field="U", units="m/s", method="volume_mean"),
        values={"mean": np.float64(3.5), "samples": np.array([1, 2])},
    )
    payload = result.to_dict()
    assert payload["metadata"]["units"] == "m/s"
    assert payload["values"] == {"mean": 3.5, "samples": [1, 2]}


def test_integrate_energy_flux_uses_outward_sign():
    from foampilot.postprocess.monitoring import integrate_energy_flux

    result = integrate_energy_flux(
        normals=np.array([[1.0, 0.0, 0.0]]),
        areas=np.array([2.0]),
        velocity=np.array([[3.0, 0.0, 0.0]]),
        temperature=np.array([300.0]),
        density=1.0,
        heat_capacity=2.0,
    )
    assert result["energy_flux"] == 3600.0


def test_integrate_momentum_flux_returns_vector_components():
    from foampilot.postprocess.monitoring import integrate_momentum_flux

    result = integrate_momentum_flux(
        normals=np.array([[1.0, 0.0, 0.0]]),
        areas=np.array([2.0]),
        velocity=np.array([[3.0, 4.0, 0.0]]),
        density=1.0,
    )
    assert result["momentum_flux_x"] == 18.0
    assert result["momentum_flux_y"] == 24.0


def test_engineering_report_exports_named_results(tmp_path):
    from foampilot.postprocess.engineering_report import EngineeringReport
    from foampilot.postprocess.results import EngineeringResult, ResultMetadata

    report = EngineeringReport(case="testCase", solver="simpleFoam")
    report.add("pressure", EngineeringResult(
        metadata=ResultMetadata(field="p", units="Pa"),
        values={"mean": 101325.0},
    ))
    json_path = report.export_json(tmp_path / "report.json")
    md_path = report.export_markdown(tmp_path / "report.md")
    assert json_path.exists() and md_path.exists()
    assert "pressure" in json_path.read_text()
    assert "CFD Engineering Report" in md_path.read_text()
