"""Base solver configuration classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SolverConfig:
    """Configuration metadata for an OpenFOAM solver."""

    name: str
    foamrun_module: str
    is_legacy: bool = False
    executable: Optional[str] = None
    requires_energy: bool = False
    requires_gravity: bool = False
    is_vof: bool = False
    is_solid: bool = False
    is_transient: bool = False
    is_compressible: bool = False
    turbulence_model: Optional[str] = None
    sub_solver: Optional[str] = None
    turbulence_file: Optional[str] = None
    transport_file: Optional[str] = None
    writes_pRef: bool = False
    writes_phase_properties: bool = False
    writes_physical_properties_per_phase: bool = False
    removes_transport_properties: bool = False
    removes_turbulence_properties: bool = False
    removes_pRef: bool = False
    default_sigma: float = 0.0
    default_phases: Optional[list] = None
    default_wall_dist: Optional[dict] = None
    default_div_alpha: Optional[str] = None
    default_div_rhoPhi_U: Optional[str] = None
    adds_interface_compression: bool = False
    default_turbulence_model: Optional[str] = None
    writes_region_solvers: bool = False
    per_region_system_files: bool = False
    solid_properties_file: Optional[str] = None
    fluid_properties_file: Optional[str] = None
    energy_variable: Optional[str] = None
    pressure_dimensions: Optional[str] = None
    t_default: Optional[str] = None
    adds_rho_solver: bool = False
    adds_energy_solver: bool = False
    use_solver_keyword: bool = False
    default_ddt_steady: Optional[str] = None
    default_ddt_transient: Optional[str] = None
    default_div_phi_U_steady: Optional[str] = None
    default_div_phi_U_transient: Optional[str] = None
    default_algorithm_steady: Optional[str] = None
    default_algorithm_transient: Optional[str] = None
    residual_control_defaults: Optional[dict] = None
    relaxation_defaults: Optional[dict] = None

    def get_simulation_type(self) -> str:
        if self.is_solid:
            return "solid"
        if self.is_compressible:
            return "compressible"
        if self.is_vof:
            return "vof"
        return "incompressible"

    def get_energy_variable(self) -> str:
        if self.is_compressible and not self.is_vof:
            return "h"
        return "T"

    @property
    def algorithm(self) -> str:
        if self.is_transient:
            return self.default_algorithm_transient or "PIMPLE"
        return self.default_algorithm_steady or "SIMPLE"


def config_from_flags(
    solver_name: str,
    *,
    compressible: bool = False,
    with_gravity: bool = False,
    is_vof: bool = False,
    is_solid: bool = False,
    energy_activated: bool = False,
    transient: bool = False,
    turbulence_model: Optional[str] = None,
    with_moving_mesh: bool = False,
) -> SolverConfig:
    from foampilot.openfoam.solvers.registry import SolverRegistry

    foamrun_module = SolverRegistry.get_module(solver_name)
    is_legacy = SolverRegistry.is_legacy(solver_name)

    if is_solid:
        effective_energy = False
        effective_compressible = False
        effective_gravity = False
    else:
        effective_energy = energy_activated
        effective_compressible = compressible
        effective_gravity = with_gravity

    if is_vof and with_gravity:
        effective_energy = False

    base_kwargs = dict(
        name=solver_name,
        foamrun_module=foamrun_module,
        is_legacy=is_legacy,
        executable=solver_name if is_legacy else None,
        requires_energy=effective_energy,
        requires_gravity=effective_gravity,
        is_vof=is_vof,
        is_solid=is_solid,
        is_transient=transient,
        is_compressible=effective_compressible,
        turbulence_model=turbulence_model,
        sub_solver=None,
    )

    if is_solid:
        from foampilot.openfoam.solvers.configs.solid import SolidConfig
        return SolidConfig(**base_kwargs)
    if is_vof:
        from foampilot.openfoam.solvers.configs.multiphase import VoFConfig
        return VoFConfig(**base_kwargs)
    if solver_name == "chtMultiRegionFoam" or solver_name == "chtMultiRegionSimpleFoam":
        from foampilot.openfoam.solvers.configs.cht import CHTConfig
        return CHTConfig(**base_kwargs)
    if effective_compressible or solver_name in {"fluid", "rhoCentralFoam", "sonicFoam", "reactingFoam"}:
        from foampilot.openfoam.solvers.configs.compressible import CompressibleConfig
        return CompressibleConfig(**base_kwargs)
    from foampilot.openfoam.solvers.configs.incompressible import IncompressibleConfig
    return IncompressibleConfig(**base_kwargs)
