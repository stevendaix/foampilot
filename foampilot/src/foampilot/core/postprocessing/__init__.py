"""Lazy public namespace for OpenFOAM post-processing backends."""
from __future__ import annotations

import importlib

_LAZY_ATTRS = {
    "FoamPostProcessing": ("foampilot.core.postprocessing.openfoam_pyvista", "FoamPostProcessing"),
    "NumpyEncoder": ("foampilot.core.postprocessing.openfoam_pyvista", "NumpyEncoder"),
    "OpenFOAMJOS3Coupler": ("foampilot.core.postprocessing.jos3_openfoam", "OpenFOAMJOS3Coupler"),
    "NodalThermalExchange": ("foampilot.core.postprocessing.jos3_openfoam", "NodalThermalExchange"),
    "JOS3_SEGMENT_NAMES": ("foampilot.core.postprocessing.jos3_openfoam", "JOS3_SEGMENT_NAMES"),
    "OpenFOAMExternalCoupledProvider": ("foampilot.core.postprocessing.openfoam_external_coupled", "OpenFOAMExternalCoupledProvider"),
    "OpenFOAM13TemperatureProvider": ("foampilot.core.postprocessing.openfoam_external_coupled", "OpenFOAM13TemperatureProvider"),
    "OpenFOAMDirectReader": ("foampilot.core.postprocessing.openfoam_direct", "OpenFOAMDirectReader"),
    "CHTDirectReader": ("foampilot.core.postprocessing.openfoam_direct", "CHTDirectReader"),
    "read_openfoam": ("foampilot.core.postprocessing.openfoam_direct", "read_openfoam"),
    "read_cht_openfoam": ("foampilot.core.postprocessing.openfoam_direct", "read_cht_openfoam"),
    "BoundaryViewer": ("foampilot.core.postprocessing.boundary_viewer", "BoundaryViewer"),
    "CFDDashboard": ("foampilot.core.postprocessing.web_presentation", "CFDDashboard"),
    "plotly_contour_from_mesh": ("foampilot.core.postprocessing.web_presentation", "plotly_contour_from_mesh"),
    "plotly_velocity_magnitude": ("foampilot.core.postprocessing.web_presentation", "plotly_velocity_magnitude"),
    "plotly_temperature_contour": ("foampilot.core.postprocessing.web_presentation", "plotly_temperature_contour"),
    "plotly_pressure_contour": ("foampilot.core.postprocessing.web_presentation", "plotly_pressure_contour"),
    "CFDMonitor": ("foampilot.core.postprocessing.monitoring", "CFDMonitor"),
    "MonitorPoint": ("foampilot.core.postprocessing.monitoring", "MonitorPoint"),
    "compute_y_plus": ("foampilot.core.postprocessing.monitoring", "compute_y_plus"),
    "integrate_surface_forces": ("foampilot.core.postprocessing.monitoring", "integrate_surface_forces"),
    "integrate_mass_flux": ("foampilot.core.postprocessing.monitoring", "integrate_mass_flux"),
    "mass_balance": ("foampilot.core.postprocessing.monitoring", "mass_balance"),
    "integrate_energy_flux": ("foampilot.core.postprocessing.monitoring", "integrate_energy_flux"),
    "integrate_momentum_flux": ("foampilot.core.postprocessing.monitoring", "integrate_momentum_flux"),
    "EngineeringResult": ("foampilot.core.postprocessing.results", "EngineeringResult"),
    "MassBalanceResult": ("foampilot.core.postprocessing.results", "MassBalanceResult"),
    "ResultMetadata": ("foampilot.core.postprocessing.results", "ResultMetadata"),
    "TimeSeriesResult": ("foampilot.core.postprocessing.results", "TimeSeriesResult"),
    "EngineeringReport": ("foampilot.core.postprocessing.engineering_report", "EngineeringReport"),
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'foampilot.core.postprocessing' has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value
