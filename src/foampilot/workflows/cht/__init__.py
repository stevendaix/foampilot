"""Conjugate heat-transfer helpers with optional post-processing imports."""
from __future__ import annotations

from .regions import FluidRegion, SolidRegion
from .interfaces import CoupledInterface
from .boundary_conditions import (
    CoupledTemperatureBC,
    ExternalTemperatureBC,
    HeatFluxBC,
    FixedTemperatureBC,
    InletOutletTemperatureBC,
    SymmetryBC,
    TotalTemperatureBC,
    RadiationCoupledTemperatureBC,
    get_coupled_temperature_bc,
    get_external_temperature_bc,
    get_fixed_temperature_bc,
    get_heat_flux_bc,
    get_inlet_outlet_bc,
    get_symmetry_bc,
    get_total_temperature_bc,
    get_radiation_coupled_temperature_bc,
)

_LAZY = {
    "ChtSolver": ("foampilot.cht.solver", "ChtSolver"),
    "calc_region_heat_flux": ("foampilot.cht.postprocess", "calc_region_heat_flux"),
    "calc_interface_heat_flux": ("foampilot.cht.postprocess", "calc_interface_heat_flux"),
    "calc_nusselt_number": ("foampilot.cht.postprocess", "calc_nusselt_number"),
    "calc_thermal_boundary_layer_thickness": ("foampilot.cht.postprocess", "calc_thermal_boundary_layer_thickness"),
    "calc_heat_transfer_coefficient": ("foampilot.cht.postprocess", "calc_heat_transfer_coefficient"),
    "calc_total_heat_balance": ("foampilot.cht.postprocess", "calc_total_heat_balance"),
    "calc_temperature_contour": ("foampilot.cht.postprocess", "calc_temperature_contour"),
    "calc_thermal_resistance": ("foampilot.cht.postprocess", "calc_thermal_resistance"),
    "OpenFOAMDirectReader": ("foampilot.postprocess.openfoam_direct", "OpenFOAMDirectReader"),
    "CHTDirectReader": ("foampilot.postprocess.openfoam_direct", "CHTDirectReader"),
    "read_openfoam": ("foampilot.postprocess.openfoam_direct", "read_openfoam"),
    "read_cht_openfoam": ("foampilot.postprocess.openfoam_direct", "read_cht_openfoam"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        module_name, attribute = _LAZY[name]
        value = getattr(importlib.import_module(module_name), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChtSolver", "FluidRegion", "SolidRegion", "CoupledInterface",
    "CoupledTemperatureBC", "ExternalTemperatureBC", "HeatFluxBC",
    "FixedTemperatureBC", "InletOutletTemperatureBC", "SymmetryBC",
    "TotalTemperatureBC", "RadiationCoupledTemperatureBC",
    "get_coupled_temperature_bc", "get_external_temperature_bc",
    "get_fixed_temperature_bc", "get_heat_flux_bc", "get_inlet_outlet_bc",
    "get_symmetry_bc", "get_total_temperature_bc", "get_radiation_coupled_temperature_bc",
    "calc_region_heat_flux", "calc_interface_heat_flux", "calc_nusselt_number",
    "calc_thermal_boundary_layer_thickness", "calc_heat_transfer_coefficient",
    "calc_total_heat_balance", "calc_temperature_contour", "calc_thermal_resistance",
    "OpenFOAMDirectReader", "CHTDirectReader", "read_openfoam", "read_cht_openfoam",
]
