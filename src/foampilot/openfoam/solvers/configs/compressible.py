"""Compressible solver configuration."""

from __future__ import annotations

from foampilot.openfoam.solvers.configs.base_config import SolverConfig


class CompressibleConfig(SolverConfig):
    """Configuration for compressible solvers."""

    def __init__(self, **kwargs):
        defaults = {
            "turbulence_file": "momentumTransport",
            "transport_file": "physicalProperties",
            "writes_pRef": True,
            "energy_variable": "h",
            "pressure_dimensions": "[1 -1 -2 0 0 0 0]",
            "t_default": "uniform 293",
            "adds_rho_solver": True,
            "adds_energy_solver": True,
            "use_solver_keyword": False,
            "default_turbulence_model": "kEpsilon",
        }
        kwargs.setdefault("is_compressible", True)
        kwargs.setdefault("requires_energy", True)
        kwargs.setdefault("is_transient", False)
        super().__init__(**{**defaults, **kwargs})
