from typing import Dict, Any, Optional


class FluidRegion:
    """Represents a fluid domain in a conjugate heat transfer case.

    Attributes
    ----------
    name : str
        Region name as it appears in the OpenFOAM case directory.
    temperature : float
        Initial temperature in Kelvin.
    velocity : tuple
        Initial velocity vector (ux, uy, uz) in m/s.
    turbulence_model : str
        Turbulence model name (e.g., ``"kOmegaSST"``).
    """

    def __init__(
        self,
        name: str,
        temperature: float = 300.0,
        velocity: tuple = (0.0, 0.0, 0.0),
        turbulence_model: str = "kOmegaSST",
    ):
        self.name = name
        self.temperature = temperature
        self.velocity = velocity
        self.turbulence_model = turbulence_model

    def get_T_field_content(self) -> str:
        return (
            f'FoamFile\n'
            f'{{\n'
            f'    format      ascii;\n'
            f'    class       volScalarField;\n'
            f'    location    "0/{self.name}";\n'
            f'    object      T;\n'
            f'}}\n'
            f'// * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
            f'\n'
            f'dimensions      [ 0 0 0 1 0 0 0 ];\n'
            f'\n'
            f'internalField   uniform {self.temperature};\n'
            f'\n'
            f'boundaryField\n'
            f'{{\n'
            f'    #includeEtc "caseDicts/setConstraintTypes"\n'
            f'\n'
            f'}}\n'
            f'\n'
            f'// ************************************************************************* //\n'
        )

    def get_U_field_content(self) -> str:
        ux, uy, uz = self.velocity
        return (
            f'FoamFile\n'
            f'{{\n'
            f'    format      ascii;\n'
            f'    class       volVectorField;\n'
            f'    location    "0/{self.name}";\n'
            f'    object      U;\n'
            f'}}\n'
            f'// * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
            f'\n'
            f'dimensions      [ 0 1 -1 0 0 0 0 ];\n'
            f'\n'
            f'internalField   uniform ({ux} {uy} {uz});\n'
            f'\n'
            f'boundaryField\n'
            f'{{\n'
            f'    #includeEtc "caseDicts/setConstraintTypes"\n'
            f'\n'
            f'}}\n'
            f'\n'
            f'// ************************************************************************* //\n'
        )

    def get_thermophysical_properties(self) -> str:
        return ""

    def get_transport_properties(self) -> str:
        return (
            "transportModel  Newtonian;\n"
            "nu              nu [0 2 -1 0 0 0 0] 1e-06;\n"
        )


class SolidRegion:
    """Represents a solid domain in a conjugate heat transfer case.

    Attributes
    ----------
    name : str
        Region name as it appears in the OpenFOAM case directory.
    temperature : float
        Initial temperature in Kelvin.
    thermal_conductivity : float
        Thermal conductivity k in W/(m·K).
    density : float
        Density rho in kg/m³.
    specific_heat : float
        Specific heat cp in J/(kg·K).
    """

    def __init__(
        self,
        name: str,
        temperature: float = 300.0,
        thermal_conductivity: float = 50.0,
        density: float = 7800.0,
        specific_heat: float = 500.0,
    ):
        self.name = name
        self.temperature = temperature
        self.thermal_conductivity = thermal_conductivity
        self.density = density
        self.specific_heat = specific_heat

    def get_T_field_content(self) -> str:
        return (
            f'FoamFile\n'
            f'{{\n'
            f'    format      ascii;\n'
            f'    class       volScalarField;\n'
            f'    location    "0/{self.name}";\n'
            f'    object      T;\n'
            f'}}\n'
            f'// * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
            f'\n'
            f'dimensions      [ 0 0 0 1 0 0 0 ];\n'
            f'\n'
            f'internalField   uniform {self.temperature};\n'
            f'\n'
            f'boundaryField\n'
            f'{{\n'
            f'    #includeEtc "caseDicts/setConstraintTypes"\n'
            f'\n'
            f'}}\n'
            f'\n'
            f'// ************************************************************************* //\n'
        )

    def get_U_field_content(self) -> str:
        return (
            f'FoamFile\n'
            f'{{\n'
            f'    format      ascii;\n'
            f'    class       volVectorField;\n'
            f'    location    "0/{self.name}";\n'
            f'    object      U;\n'
            f'}}\n'
            f'// * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
            f'\n'
            f'dimensions      [ 0 1 -1 0 0 0 0 ];\n'
            f'\n'
            f'internalField   uniform (0 0 0);\n'
            f'\n'
            f'boundaryField\n'
            f'{{\n'
            f'    #includeEtc "caseDicts/setConstraintTypes"\n'
            f'\n'
            f'}}\n'
            f'\n'
            f'// ************************************************************************* //\n'
        )

    def get_thermophysical_properties(self) -> str:
        kappa = self.thermal_conductivity / (self.density * self.specific_heat)
        return (
            "thermoType\n"
            "{\n"
            "    type            heSolidThermo;\n"
            "    mixture         pureMixture;\n"
            "    transport       thermoConstant;\n"
            "    thermo          hConst;\n"
            "    equationOfState perfectGas;\n"
            "    specie          specie;\n"
            "    energy          sensibleEnthalpy;\n"
            "}\n"
            f"\n"
            f"specificHeat    {self.specific_heat} [0 2 -2 0 0 0 0];\n"
            f"thermalConductivity {self.thermal_conductivity} [1 1 -3 0 0 0 0];\n"
            f"rho             {self.density} [1 -3 0 0 0 0 0];\n"
            f"kappa           {kappa} [0 0 0 0 0 0 0];\n"
        )

    def get_transport_properties(self) -> str:
        return (
            f"thermoType\n"
            f"{{\n"
            f"    type            thermoTransport;\n"
            f"    mixture         pureMixture;\n"
            f"    transport       thermoConstant;\n"
            f"}}\n"
            f"\n"
            f"mu              0 [1 -1 -1 0 0 0 0];\n"
            f"nu              0 [0 2 -1 0 0 0 0];\n"
        )