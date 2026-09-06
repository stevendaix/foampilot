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

    return SolverConfig(
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
