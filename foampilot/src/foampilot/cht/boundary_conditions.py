from typing import List, Dict, Any, Optional


class CoupledTemperatureBC:
    """Boundary condition for coupled temperature at a fluid-solid interface.

    Used at a CoupledInterface where the fluid and solid sides exchange
    heat and the interface must maintain temperature continuity.

    Parameters
    ----------
    patch_name : str
        Name of the interface patch.
    T_init : float
        Initial temperature for the `value` entry (Kelvin).
    T_neighbor : float, optional
        Temperature of the neighbouring region (Kelvin), written to
        ``Tnbr``.  Defaults to ``T_init``.
    kappa_method : str, optional
        Method to obtain the thermal conductivity.  Common values are
        ``"none"``, ``"solidThermo"``, ``"lookup"``.

    Attributes
    ----------
    type : str
        OpenFOAM boundary condition type (``coupledTemperature``).
    """

    def __init__(
        self,
        patch_name: str,
        T_init: float = 300.0,
        T_neighbor: Optional[float] = None,
        kappa_method: str = "none",
    ):
        self.patch_name = patch_name
        self.T_init = T_init
        self.T_neighbor = T_neighbor if T_neighbor is not None else T_init
        self.kappa_method = kappa_method
        self.type = "coupledTemperature"

    def to_of(self) -> Dict[str, Any]:
        """Return the boundary-condition parameters as an OpenFOAM-style dict."""
        return {
            "type": self.type,
            "value": f"uniform {self.T_init}",
            "Tnbr": f"uniform {self.T_neighbor}",
            "kappaMethod": self.kappa_method,
        }

    def __str__(self) -> str:
        return f"CoupledTemperatureBC({self.patch_name!r}, T={self.T_init})"


class ExternalTemperatureBC:
    """Fixed-temperature wall with external convection boundary condition.

    OpenFOAM type ``externalTemperature``.  The wall temperature evolves
    according to the external heat transfer coefficient ``h`` and the
    ambient temperature ``Ta``.

    Parameters
    ----------
    patch_name : str
        Name of the wall patch.
    ambient_temperature : float
        Ambient (free-stream) temperature ``Ta`` in Kelvin.
    heat_transfer_coefficient : float
        Convective coefficient ``h`` in W/(m²·K).
    """

    def __init__(
        self,
        patch_name: str,
        ambient_temperature: float = 300.0,
        heat_transfer_coefficient: float = 10.0,
    ):
        self.patch_name = patch_name
        self.ambient_temperature = ambient_temperature
        self.heat_transfer_coefficient = heat_transfer_coefficient
        self.type = "externalTemperature"

    def to_of(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "Ta": f"uniform {self.ambient_temperature}",
            "h": f"uniform {self.heat_transfer_coefficient}",
            "value": f"uniform {self.ambient_temperature}",
        }

    def __str__(self) -> str:
        return (
            f"ExternalTemperatureBC({self.patch_name!r}, Ta={self.ambient_temperature}, "
            f"h={self.heat_transfer_coefficient})"
        )


class HeatFluxBC:
    """Boundary condition for a prescribed wall heat flux.

    Uses the OpenFOAM ``externalWallHeatFluxTemperature`` condition with
    a fixed flux ``q``.

    Parameters
    ----------
    patch_name : str
        Name of the wall patch.
    heat_flux : float
        Prescribed heat flux ``q`` in W/m² (positive = into the domain).
    T_init : float
        Initial / back-reference temperature (Kelvin).
    kappa_method : str, optional
        How the material thermal conductivity is obtained.
        Common values: ``"solidThermo"``, ``"lookup"``.
    """

    def __init__(
        self,
        patch_name: str,
        heat_flux: float = 0.0,
        T_init: float = 300.0,
        kappa_method: str = "solidThermo",
    ):
        self.patch_name = patch_name
        self.heat_flux = heat_flux
        self.T_init = T_init
        self.kappa_method = kappa_method
        self.type = "externalWallHeatFluxTemperature"

    def to_of(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "flux": "q",
            "q": f"uniform {self.heat_flux}",
            "kappaMethod": self.kappa_method,
            "value": f"uniform {self.T_init}",
        }

    def __str__(self) -> str:
        return f"HeatFluxBC({self.patch_name!r}, q={self.heat_flux})"


class FixedTemperatureBC:
    """Fixed-value (Dirichlet) temperature boundary condition.

    Parameters
    ----------
    patch_name : str
        Name of the patch.
    temperature : float
        Fixed temperature in Kelvin.
    """

    def __init__(self, patch_name: str, temperature: float = 300.0):
        self.patch_name = patch_name
        self.temperature = temperature
        self.type = "fixedValue"

    def to_of(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "value": f"uniform {self.temperature}",
        }

    def __str__(self) -> str:
        return f"FixedTemperatureBC({self.patch_name!r}, T={self.temperature})"


class InletOutletTemperatureBC:
    """Inlet-outlet temperature boundary condition.

    Applies ``fixedValue`` at inlets and ``inletOutlet`` at outlets.

    Parameters
    ----------
    inlet_name : str
        Name of the inlet patch.
    outlet_name : str
        Name of the outlet patch.
    inlet_temperature : float
        Temperature at the inlet (Kelvin).
    outlet_temperature : float, optional
        Back-flow temperature at the outlet (Kelvin).  Defaults to
        ``inlet_temperature``.
    """

    def __init__(
        self,
        inlet_name: str,
        outlet_name: str,
        inlet_temperature: float = 300.0,
        outlet_temperature: Optional[float] = None,
    ):
        self.inlet_name = inlet_name
        self.outlet_name = outlet_name
        self.inlet_temperature = inlet_temperature
        self.outlet_temperature = (
            outlet_temperature if outlet_temperature is not None
            else inlet_temperature
        )

    def to_of(self) -> Dict[str, Dict[str, Any]]:
        return {
            self.inlet_name: {
                "type": "fixedValue",
                "value": f"uniform {self.inlet_temperature}",
            },
            self.outlet_name: {
                "type": "inletOutlet",
                "value": f"uniform {self.outlet_temperature}",
                "inletValue": f"uniform {self.outlet_temperature}",
            },
        }

    def __str__(self) -> str:
        return (
            f"InletOutletTemperatureBC(inlet={self.inlet_name!r}, "
            f"outlet={self.outlet_name!r})"
        )


class SymmetryBC:
    """Symmetry (or slip) boundary condition for temperature.

    Parameters
    ----------
    patch_name : str
        Name of the symmetry patch.
    """

    def __init__(self, patch_name: str):
        self.patch_name = patch_name
        self.type = "symmetry"

    def to_of(self) -> Dict[str, Any]:
        return {"type": self.type}

    def __str__(self) -> str:
        return f"SymmetryBC({self.patch_name!r})"


class TotalTemperatureBC:
    """Total (stagnation) temperature boundary condition.

    Used at inlets of compressible flows where the total temperature
    is prescribed.

    Parameters
    ----------
    patch_name : str
        Name of the inlet patch.
    total_temperature : float
        Total temperature in Kelvin.
    """

    def __init__(self, patch_name: str, total_temperature: float = 300.0):
        self.patch_name = patch_name
        self.total_temperature = total_temperature
        self.type = "totalTemperature"

    def to_of(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "T0": f"uniform {self.total_temperature}",
            "value": f"uniform {self.total_temperature}",
        }

    def __str__(self) -> str:
        return f"TotalTemperatureBC({self.patch_name!r}, T0={self.total_temperature})"


class RadiationCoupledTemperatureBC:
    """Radiation-coupled temperature boundary condition at a fluid-solid
    interface where thermal radiation is active.

    Parameters
    ----------
    patch_name : str
        Name of the interface patch.
    T_init : float
        Initial temperature (Kelvin).
    T_neighbor : float, optional
        Neighbour-side temperature (Kelvin).
    """

    def __init__(
        self,
        patch_name: str,
        T_init: float = 300.0,
        T_neighbor: Optional[float] = None,
    ):
        self.patch_name = patch_name
        self.T_init = T_init
        self.T_neighbor = T_neighbor if T_neighbor is not None else T_init
        self.type = "radiationCoupledTemperature"

    def to_of(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "value": f"uniform {self.T_init}",
            "Tnbr": f"uniform {self.T_neighbor}",
        }

    def __str__(self) -> str:
        return f"RadiationCoupledTemperatureBC({self.patch_name!r})"


# ---------------------------------------------------------------------------
# Functional helpers (kept for backward compatibility / procedural use)
# ---------------------------------------------------------------------------

def get_coupled_temperature_bc(
    interface_name: str,
    neighbor_temperature: float = 300.0,
) -> str:
    """Generate a coupledTemperature boundary condition entry (legacy function)."""
    return (
        f'    {interface_name}\n'
        f'    {{\n'
        f'        type            coupledTemperature;\n'
        f'        value           uniform 300;\n'
        f'        Tnbr            uniform {neighbor_temperature};\n'
        f'        kappaMethod     none;\n'
        f'    }}\n'
    )


def get_external_temperature_bc(
    patch_name: str,
    ambient_temperature: float = 300.0,
    heat_transfer_coefficient: float = 10.0,
) -> str:
    """Generate an externalTemperature boundary condition entry (legacy function)."""
    return (
        f'    {patch_name}\n'
        f'    {{\n'
        f'        type            externalTemperature;\n'
        f'        Ta              uniform {ambient_temperature};\n'
        f'        h               uniform {heat_transfer_coefficient};\n'
        f'        value           uniform {ambient_temperature};\n'
        f'    }}\n'
    )


def get_fixed_temperature_bc(
    patch_name: str,
    temperature: float = 300.0,
) -> str:
    """Generate a fixedValue boundary condition entry for temperature."""
    return (
        f'    {patch_name}\n'
        f'    {{\n'
        f'        type            fixedValue;\n'
        f'        value           uniform {temperature};\n'
        f'    }}\n'
    )


def get_heat_flux_bc(
    patch_name: str,
    heat_flux: float = 0.0,
) -> str:
    """Generate a fixedFluxTemperature boundary condition entry."""
    return (
        f'    {patch_name}\n'
        f'    {{\n'
        f'        type            externalWallHeatFluxTemperature;\n'
        f'        flux            q;\n'
        f'        q               uniform {heat_flux};\n'
        f'        kappaMethod     solidThermo;\n'
        f'        value           uniform 300;\n'
        f'    }}\n'
    )


def get_inlet_outlet_bc(
    inlet_name: str,
    outlet_name: str,
    inlet_temperature: float = 300.0,
    outlet_temperature: float = 300.0,
) -> str:
    """Generate inlet (fixedValue) and outlet (inletOutlet) BC entries."""
    return (
        f'    {inlet_name}\n'
        f'    {{\n'
        f'        type            fixedValue;\n'
        f'        value           uniform {inlet_temperature};\n'
        f'    }}\n'
        f'    {outlet_name}\n'
        f'    {{\n'
        f'        type            inletOutlet;\n'
        f'        value           uniform {outlet_temperature};\n'
        f'        inletValue      uniform {outlet_temperature};\n'
        f'    }}\n'
    )


def get_symmetry_bc(patch_name: str) -> str:
    """Generate a symmetry boundary condition entry."""
    return (
        f'    {patch_name}\n'
        f'    {{\n'
        f'        type            symmetry;\n'
        f'    }}\n'
    )


def get_total_temperature_bc(
    patch_name: str,
    total_temperature: float = 300.0,
) -> str:
    """Generate a totalTemperature boundary condition entry."""
    return (
        f'    {patch_name}\n'
        f'    {{\n'
        f'        type            totalTemperature;\n'
        f'        T0              uniform {total_temperature};\n'
        f'        value           uniform {total_temperature};\n'
        f'    }}\n'
    )


def get_radiation_coupled_temperature_bc(
    patch_name: str,
    T_init: float = 300.0,
    T_neighbor: float = 300.0,
) -> str:
    """Generate a radiationCoupledTemperature boundary condition entry."""
    return (
        f'    {patch_name}\n'
        f'    {{\n'
        f'        type            radiationCoupledTemperature;\n'
        f'        value           uniform {T_init};\n'
        f'        Tnbr            uniform {T_neighbor};\n'
        f'    }}\n'
    )
