"""Multiphase / VoF solver configuration."""

from __future__ import annotations

from foampilot.openfoam.solvers.configs.base_config import SolverConfig


class VoFConfig(SolverConfig):
    """Configuration for multiphase volume-of-fluid solvers."""

    def __init__(self, **kwargs):
        defaults = {
            "turbulence_file": "momentumTransport",
            "transport_file": None,
            "writes_phase_properties": True,
            "writes_physical_properties_per_phase": True,
            "removes_transport_properties": True,
            "removes_turbulence_properties": True,
            "removes_pRef": True,
            "default_sigma": 0.0728,
            "default_phases": ["water", "air"],
            "default_wall_dist": {"method": "meshWave"},
            "default_div_alpha": "Gauss MPLIC",
            "default_div_rhoPhi_U": "Gauss linearUpwind grad(U)",
            "adds_interface_compression": True,
            "default_turbulence_model": "kEpsilon",
        }
        kwargs.setdefault("is_vof", True)
        kwargs.setdefault("is_compressible", False)
        kwargs.setdefault("requires_energy", False)
        kwargs.setdefault("is_transient", True)
        super().__init__(**{**defaults, **kwargs})
