from typing import List


def get_coupled_temperature_bc(
    interface_name: str,
    neighbor_temperature: float = 300.0,
) -> str:
    """Generate a coupledTemperature boundary condition entry.

    Parameters
    ----------
    interface_name : str
        Name of the interface patch.
    neighbor_temperature : float
        Temperature of the neighboring region in Kelvin.

    Returns
    -------
    str
        OpenFOAM dictionary entry for the boundary condition.
    """
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
    """Generate an externalTemperature boundary condition entry.

    Parameters
    ----------
    patch_name : str
        Name of the wall patch.
    ambient_temperature : float
        Ambient temperature Ta in Kelvin.
    heat_transfer_coefficient : float
        Convective heat transfer coefficient h in W/(m²·K).

    Returns
    -------
    str
        OpenFOAM dictionary entry for the boundary condition.
    """
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
    """Generate a fixedValue boundary condition entry for temperature.

    Parameters
    ----------
    patch_name : str
        Name of the patch.
    temperature : float
        Fixed temperature value in Kelvin.

    Returns
    -------
    str
        OpenFOAM dictionary entry for the boundary condition.
    """
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
    """Generate a fixedFluxTemperature boundary condition entry.

    Parameters
    ----------
    patch_name : str
        Name of the wall patch.
    heat_flux : float
        Prescribed heat flux q in W/m².

    Returns
    -------
    str
        OpenFOAM dictionary entry for the boundary condition.
    """
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
    """Generate inlet (fixedValue) and outlet (inletOutlet) BC entries.

    Parameters
    ----------
    inlet_name : str
        Name of the inlet patch.
    outlet_name : str
        Name of the outlet patch.
    inlet_temperature : float
        Inlet temperature in Kelvin.
    outlet_temperature : float
        Outlet temperature used for inletOutlet BC.

    Returns
    -------
    str
        OpenFOAM dictionary entries for both patches.
    """
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
    """Generate a symmetry boundary condition entry.

    Parameters
    ----------
    patch_name : str
        Name of the symmetry patch.

    Returns
    -------
    str
        OpenFOAM dictionary entry for the boundary condition.
    """
    return (
        f'    {patch_name}\n'
        f'    {{\n'
        f'        type            symmetry;\n'
        f'    }}\n'
    )