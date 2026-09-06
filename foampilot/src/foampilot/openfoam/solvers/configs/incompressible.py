"""Incompressible solver configuration."""

from __future__ import annotations

from foampilot.openfoam.solvers.configs.base_config import SolverConfig


class IncompressibleConfig(SolverConfig):
    """Configuration for incompressible solvers."""

    def __init__(self, **kwargs):
        defaults = {
            "turbulence_file": "turbulenceProperties",
            "transport_file": "transportProperties",
            "writes_pRef": True,
            "default_ddt_steady": "steadyState",
            "default_ddt_transient": "Euler",
            "default_div_phi_U_steady": "Gauss upwind",
            "default_div_phi_U_transient": "bounded Gauss linearUpwind grad(U)",
            "default_algorithm_steady": "SIMPLE",
            "default_algorithm_transient": "PIMPLE",
            "residual_control_defaults": {
                "p": "1e-4",
                "U": "1e-4",
            },
            "relaxation_defaults": {
                "fields": {"p": "0.3"},
                "equations": {"U": "0.7"},
            },
            "default_turbulence_model": "kEpsilon",
        }
        kwargs.setdefault("is_compressible", False)
        kwargs.setdefault("requires_energy", False)
        kwargs.setdefault("is_transient", False)
        super().__init__(**{**defaults, **kwargs})
