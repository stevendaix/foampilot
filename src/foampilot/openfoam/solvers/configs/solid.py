"""Solid solver configuration."""

from __future__ import annotations

from foampilot.openfoam.solvers.configs.base_config import SolverConfig


class SolidConfig(SolverConfig):
    """Configuration for solid mechanics solvers."""

    def __init__(self, **kwargs):
        defaults = {
            "turbulence_file": None,
            "transport_file": None,
            "writes_pRef": False,
            "requires_energy": False,
            "requires_gravity": False,
            "is_compressible": False,
            "is_transient": False,
            "default_turbulence_model": None,
        }
        kwargs.setdefault("is_solid", True)
        super().__init__(**{**defaults, **kwargs})
