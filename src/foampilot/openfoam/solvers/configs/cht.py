"""CHT solver configuration."""

from __future__ import annotations

from foampilot.openfoam.solvers.configs.base_config import SolverConfig


class CHTConfig(SolverConfig):
    """Configuration for conjugate heat transfer solvers."""

    def __init__(self, **kwargs):
        defaults = {
            "turbulence_file": "momentumTransport",
            "transport_file": None,
            "writes_pRef": False,
            "requires_energy": True,
            "requires_gravity": False,
            "is_transient": False,
            "default_turbulence_model": "kOmegaSST",
            "writes_region_solvers": True,
            "per_region_system_files": True,
            "solid_properties_file": "physicalProperties",
            "fluid_properties_file": "physicalProperties",
            "executable": "foamMultiRun",
        }
        kwargs.setdefault("is_solid", False)
        kwargs.setdefault("is_compressible", False)
        kwargs.setdefault("is_vof", False)
        super().__init__(**{**defaults, **kwargs})
