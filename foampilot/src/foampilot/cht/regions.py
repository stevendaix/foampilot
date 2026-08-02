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
        thermophysical_model: str = "heRhoThermo",
        mixture_type: str = "pureMixture",
        transport_model: str = "const",
        thermo_model: str = "hConst",
        equation_of_state: str = "perfectGas",
        specie: str = "specie",
        energy: str = "sensibleEnthalpy",
    ):
        self.name = name
        self.temperature = temperature
        self.velocity = velocity
        self.turbulence_model = turbulence_model
        self.thermophysical_model = thermophysical_model
        self.mixture_type = mixture_type
        self.transport_model = transport_model
        self.thermo_model = thermo_model
        self.equation_of_state = equation_of_state
        self.specie = specie
        self.energy = energy

    def get_T_field_content(self) -> str:
        """Generate the OpenFOAM content for the initial temperature field ``T``."""
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
        """Generate the OpenFOAM content for the initial velocity field ``U``."""
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
        """Generate the ``thermophysicalProperties`` dictionary for the fluid region.

        Uses the selected thermo type (e.g. ``heRhoThermo``) with a
        pure-mixture, constant-transport, constant-heat thermo model.
        """
        return (
            f"thermoType\n"
            f"{{\n"
            f"    type            {self.thermophysical_model};\n"
            f"    mixture         {self.mixture_type};\n"
            f"    transport       {self.transport_model};\n"
            f"    thermo          {self.thermo_model};\n"
            f"    equationOfState {self.equation_of_state};\n"
            f"    specie          {self.specie};\n"
            f"    energy          {self.energy};\n"
            f"}}\n"
        )

    def get_transport_properties(self) -> str:
        """Return transport properties (nu) for an incompressible fluid."""
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
        thermophysical_model: str = "heSolidThermo",
        mixture_type: str = "pureMixture",
        transport_model: str = "const",
        thermo_model: str = "hConst",
        equation_of_state: str = "perfectGas",
        specie: str = "specie",
        energy: str = "sensibleEnthalpy",
    ):
        self.name = name
        self.temperature = temperature
        self.thermal_conductivity = thermal_conductivity
        self.density = density
        self.specific_heat = specific_heat
        self.thermophysical_model = thermophysical_model
        self.mixture_type = mixture_type
        self.transport_model = transport_model
        self.thermo_model = thermo_model
        self.equation_of_state = equation_of_state
        self.specie = specie
        self.energy = energy

    def get_T_field_content(self) -> str:
        """Generate the OpenFOAM content for the initial temperature field ``T``."""
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
        """Return placeholder ``U`` content for solid regions.

        Solid regions in CHT do not solve for velocity; this method
        returns the FoamFile header and internalField of zero for
        completeness, but it is generally not written for solid regions.
        """
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
        """Generate the ``thermophysicalProperties`` dictionary for the solid region.

        Uses ``heSolidThermo`` which reads thermal conductivity, density,
        and specific heat from the dictionary entries below.
        """
        return (
            f"thermoType\n"
            f"{{\n"
            f"    type            {self.thermophysical_model};\n"
            f"    mixture         {self.mixture_type};\n"
            f"    transport       {self.transport_model};\n"
            f"    thermo          {self.thermo_model};\n"
            f"    equationOfState {self.equation_of_state};\n"
            f"    specie          {self.specie};\n"
            f"    energy          {self.energy};\n"
            f"}}\n"
            f"\n"
            f"specificHeat        {self.specific_heat} [0 2 -2 0 0 0 0];\n"
            f"thermalConductivity {self.thermal_conductivity} [1 1 -3 0 0 0 0];\n"
            f"rho                 {self.density} [1 -3 0 0 0 0 0];\n"
        )

    def get_transport_properties(self) -> str:
        """Return ``transportProperties`` for the solid region.

        Solids use ``const`` transport with zero dynamic viscosity and
        kinematic viscosity (they do not flow).
        """
        return (
            f"transportModel  const;\n"
            f"mu              0 [1 -1 -1 0 0 0 0];\n"
            f"nu              0 [0 2 -1 0 0 0 0];\n"
        )
