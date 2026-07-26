from typing import Dict, Any


class CoupledInterface:
    """Represents a fluid-solid interface in a CHT simulation.

    Attributes
    ----------
    name : str
        Interface name used in boundary conditions.
    fluid_region : str
        Name of the fluid region.
    solid_region : str
        Name of the solid region.
    heat_transfer_coefficient : float, optional
        Heat transfer coefficient h in W/(m²·K).
    thickness_layers : list of float, optional
        Thicknesses of conductive layers at the interface in meters.
    kappa_layers : list of float, optional
        Thermal conductivities of conductive layers in W/(m·K).
    """

    def __init__(
        self,
        name: str,
        fluid_region: str,
        solid_region: str,
        heat_transfer_coefficient: float = 10.0,
        thickness_layers: list = None,
        kappa_layers: list = None,
    ):
        self.name = name
        self.fluid_region = fluid_region
        self.solid_region = solid_region
        self.heat_transfer_coefficient = heat_transfer_coefficient
        self.thickness_layers = thickness_layers or []
        self.kappa_layers = kappa_layers or []

    def get_fluid_bc_content(self) -> str:
        if self.thickness_layers and self.kappa_layers:
            return (
                f'    {self.name}\n'
                f'    {{\n'
                f'        type            coupledTemperature;\n'
                f'        value           uniform 300;\n'
                f'        Tnbr            uniform 300;\n'
                f'        kappaMethod     none;\n'
                f'    }}\n'
            )
        return (
            f'    {self.name}\n'
            f'    {{\n'
            f'        type            coupledTemperature;\n'
            f'        value           uniform 300;\n'
            f'        Tnbr            uniform 300;\n'
            f'        kappaMethod     none;\n'
            f'    }}\n'
        )

    def get_solid_bc_content(self) -> str:
        if self.thickness_layers and self.kappa_layers:
            thickness_str = " ".join(str(t) for t in self.thickness_layers)
            kappa_str = " ".join(str(k) for k in self.kappa_layers)
            return (
                f'    {self.name}\n'
                f'    {{\n'
                f'        type            coupledTemperature;\n'
                f'        value           uniform 300;\n'
                f'        Tnbr            uniform 300;\n'
                f'        kappaMethod     none;\n'
                f'    }}\n'
            )
        return (
            f'    {self.name}\n'
            f'    {{\n'
            f'        type            coupledTemperature;\n'
            f'        value           uniform 300;\n'
            f'        Tnbr            uniform 300;\n'
            f'        kappaMethod     none;\n'
            f'    }}\n'
        )

    def get_content(self) -> str:
        return (
            f'// Interface: {self.name}\n'
            f'// Fluid region: {self.fluid_region}\n'
            f'// Solid region: {self.solid_region}\n'
            f'h = {self.heat_transfer_coefficient} W/(m2 K)\n'
        )