"""Lazy public namespace for OpenFOAM post-processing backends."""
from __future__ import annotations

import importlib

_LAZY_ATTRS = {
    "FoamPostProcessing": ("foampilot.postprocess.openfoam_pyvista", "FoamPostProcessing"),
    "NumpyEncoder": ("foampilot.postprocess.openfoam_pyvista", "NumpyEncoder"),
    "OpenFOAMJOS3Coupler": ("foampilot.postprocess.jos3_openfoam", "OpenFOAMJOS3Coupler"),
    "NodalThermalExchange": ("foampilot.postprocess.jos3_openfoam", "NodalThermalExchange"),
    "JOS3_SEGMENT_NAMES": ("foampilot.postprocess.jos3_openfoam", "JOS3_SEGMENT_NAMES"),
    "OpenFOAMExternalCoupledProvider": ("foampilot.postprocess.openfoam_external_coupled", "OpenFOAMExternalCoupledProvider"),
    "OpenFOAM13TemperatureProvider": ("foampilot.postprocess.openfoam_external_coupled", "OpenFOAM13TemperatureProvider"),
    "OpenFOAMDirectReader": ("foampilot.postprocess.openfoam_direct", "OpenFOAMDirectReader"),
    "CHTDirectReader": ("foampilot.postprocess.openfoam_direct", "CHTDirectReader"),
    "read_openfoam": ("foampilot.postprocess.openfoam_direct", "read_openfoam"),
    "read_cht_openfoam": ("foampilot.postprocess.openfoam_direct", "read_cht_openfoam"),
    "BoundaryViewer": ("foampilot.postprocess.boundary_viewer", "BoundaryViewer"),
    "CFDDashboard": ("foampilot.postprocess.web_presentation", "CFDDashboard"),
    "plotly_contour_from_mesh": ("foampilot.postprocess.web_presentation", "plotly_contour_from_mesh"),
    "plotly_velocity_magnitude": ("foampilot.postprocess.web_presentation", "plotly_velocity_magnitude"),
    "plotly_temperature_contour": ("foampilot.postprocess.web_presentation", "plotly_temperature_contour"),
    "plotly_pressure_contour": ("foampilot.postprocess.web_presentation", "plotly_pressure_contour"),
    "CFDMonitor": ("foampilot.postprocess.monitoring", "CFDMonitor"),
    "MonitorPoint": ("foampilot.postprocess.monitoring", "MonitorPoint"),
    "compute_y_plus": ("foampilot.postprocess.monitoring", "compute_y_plus"),
    "integrate_surface_forces": ("foampilot.postprocess.monitoring", "integrate_surface_forces"),
    "integrate_mass_flux": ("foampilot.postprocess.monitoring", "integrate_mass_flux"),
    "mass_balance": ("foampilot.postprocess.monitoring", "mass_balance"),
    "integrate_energy_flux": ("foampilot.postprocess.monitoring", "integrate_energy_flux"),
    "integrate_momentum_flux": ("foampilot.postprocess.monitoring", "integrate_momentum_flux"),
    "EngineeringResult": ("foampilot.postprocess.results", "EngineeringResult"),
    "MassBalanceResult": ("foampilot.postprocess.results", "MassBalanceResult"),
    "ResultMetadata": ("foampilot.postprocess.results", "ResultMetadata"),
    "TimeSeriesResult": ("foampilot.postprocess.results", "TimeSeriesResult"),
    "EngineeringReport": ("foampilot.postprocess.engineering_report", "EngineeringReport"),
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'foampilot.postprocess' has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value
