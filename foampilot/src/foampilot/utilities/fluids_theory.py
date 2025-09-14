from pyfluids import Fluid, FluidsList, Input
from foampilot.utilities.manageunits import Quantity

class FluidMechanics:
    """
    A comprehensive fluid mechanics calculator for CFD applications.
    
    This class provides methods to calculate key fluid mechanics parameters and boundary layer
    characteristics essential for CFD mesh sizing and simulation setup. It supports various
    fluids through the pyfluids library and handles unit conversions automatically.

    Attributes:
        fluid_name (FluidsList): The fluid being analyzed (from FluidsList enum)
        temperature (Quantity): Fluid temperature with units
        pressure (Quantity): Fluid pressure with units
        velocity (Quantity): Characteristic flow velocity with units
        characteristic_length (Quantity): Relevant length scale for dimensionless numbers
        fluid: PyFluids Fluid object initialized with current state
    """

    @staticmethod
    def get_available_fluids() -> dict:
        """
        Get a dictionary of all available fluids in the pyfluids library.

        Returns:
            dict: Dictionary where keys are fluid names and values are FluidsList enum members.
                Example: {
                    'Water': FluidsList.Water,
                    'Air': FluidsList.Air,
                    'Nitrogen': FluidsList.Nitrogen,
                    ...
                }

        Example:
            >>> fluids = FluidMechanics.get_available_fluids()
            >>> for name, fluid in fluids.items():
            ...     print(f"Fluid: {name}")
            >>> 
            >>> # Create instance with a specific fluid
            >>> fm = FluidMechanics(fluids['Water'], ...)
        """
        available_fluids = {}
        for fluid in FluidsList:
            # Get the name without the enum class prefix
            name = fluid.name
            available_fluids[name] = fluid
        return available_fluids

    def __init__(self, fluid_name: FluidsList, temperature: Quantity, pressure: Quantity, 
                 velocity: Quantity | None = None, characteristic_length: Quantity | None = None):
        """
        Initialize fluid mechanics calculator with fluid properties and optional flow conditions.

        Args:
            fluid_name: Fluid type from FluidsList enum (e.g., FluidsList.Water)
            temperature: Fluid temperature (e.g., Quantity(300, 'K'))
            pressure: Fluid pressure (e.g., Quantity(101325, 'Pa'))
            velocity: Optional characteristic flow velocity (e.g., Quantity(2, 'm/s'))
            characteristic_length: Optional relevant length scale (e.g., pipe diameter)

        Notes:
            velocity and characteristic_length are only required for dimensionless number
            calculations (Reynolds, y+, etc.) and boundary layer analysis. They can be
            omitted if you only need fluid properties.
        """
        self.fluid_name = fluid_name
        self.temperature = temperature
        self.pressure = pressure
        self.velocity = velocity
        self.characteristic_length = characteristic_length
        self.fluid = Fluid(fluid_name).with_state(
            Input.pressure(pressure.get_in("Pa")), 
            Input.temperature(temperature.get_in("K")))

    def get_fluid_properties(self) -> dict:
        """
        Retrieve available fluid properties at specified temperature.

        Args:
            temperature: Temperature at which to evaluate properties

        Returns:
            dict: Dictionary containing available fluid properties. Properties that are not
            available for the current fluid or state are omitted. Most properties are
            returned as Quantity objects with appropriate units:
                - density [kg/m³]
                - dynamic_viscosity [Pa·s]
                - kinematic_viscosity [m²/s]
                - thermal_conductivity [W/(m·K)]
                - specific_heat [J/(kg·K)]
                - enthalpy [J/kg]
                - entropy [J/(kg·K)]
                - internal_energy [J/kg]
                - quality [-] (dimensionless)
                - speed_of_sound [m/s]
                - surface_tension [N/m]
                - phase [-] (string)
                - pressure [Pa]
                - temperature [K]
                - critical_pressure [Pa]
                - critical_temperature [K]

        Note:
            The actual properties available depend on the fluid type and its current state.
            Some properties might not be available for certain fluids or conditions.
            Properties without units (like 'quality' and 'phase') are returned as raw values.
        """
        fluid_state = Fluid(self.fluid_name).with_state(
            Input.pressure(self.pressure.get_in("Pa")), 
            Input.temperature(self.temperature.get_in("K")))
        
        # Create dictionary for available properties
        properties = {}

        # Dictionary of property definitions with their units
        property_definitions = {
            'density': ('density', 'kg/m^3'),
            'dynamic_viscosity': ('dynamic_viscosity', 'Pa.s'),
            'kinematic_viscosity': ('kinematic_viscosity', 'm^2/s'),
            'thermal_conductivity': ('conductivity', 'W/(m.K)'),
            'specific_heat': ('specific_heat', 'J/(kg.K)'),
            'enthalpy': ('enthalpy', 'J/kg'),
            'entropy': ('entropy', 'J/(kg.K)'),
            'internal_energy': ('internal_energy', 'J/kg'),
            'quality': ('quality', None),  # No units for quality
            'speed_of_sound': ('speed_of_sound', 'm/s'),
            'surface_tension': ('surface_tension', 'N/m'),
            'phase': ('phase', None),  # No units for phase
            'pressure': ('pressure', 'Pa'),
            'temperature': ('temperature', 'K'),
            'critical_pressure': ('critical_pressure', 'Pa'),
            'critical_temperature': ('critical_temperature', 'K')
        }

        # Add properties that are available
        for prop_name, (attr_name, unit) in property_definitions.items():
            try:
                value = getattr(fluid_state, attr_name)
                if value is not None:  # Skip None values
                    if unit is not None:
                        properties[prop_name] = Quantity(value, unit)
                    else:
                        properties[prop_name] = value
            except (AttributeError, ValueError) as e:
                # Skip properties that don't exist or can't be converted
                continue
        
        return properties

    def calculate_reynolds(self) -> float:
        """
        Calculate Reynolds number for the current flow conditions.

        Returns:
            float: Reynolds number (Re = ρvL/μ)
        
        Raises:
            ValueError: If velocity or characteristic_length were not provided during initialization
        """
        if self.velocity is None or self.characteristic_length is None:
            raise ValueError("Reynolds number calculation requires both velocity and characteristic_length")
        
        props = self.get_fluid_properties(self.temperature)
        return (props['density'].get_in('kg/m^3') * self.velocity.get_in('m/s') * 
                self.characteristic_length.get_in('m')) / props['dynamic_viscosity'].get_in('Pa.s')

    def calculate_y_plus(self, wall_shear_stress: Quantity) -> float:
        """
        Calculate y+ value (dimensionless wall distance) for turbulence modeling.

        Args:
            wall_shear_stress: Wall shear stress with units

        Returns:
            float: y+ value (y+ = τ_w·L/μ)
            
        Raises:
            ValueError: If characteristic_length was not provided during initialization
        """
        if self.characteristic_length is None:
            raise ValueError("y+ calculation requires characteristic_length")
            
        viscosity = self.fluid.dynamic_viscosity  
        return (wall_shear_stress.get_in('Pa') * 
                self.characteristic_length.get_in('m')) / viscosity

    def calculate_prandtl(self) -> float:
        """
        Calculate Prandtl number for the current fluid state.

        Returns:
            float: Prandtl number (Pr = μ·c_p/k)
        """
        props = self.get_fluid_properties(self.temperature)
        return ((props['dynamic_viscosity'].get_in('Pa.s') * 
                props['specific_heat'].get_in('J/(kg.K)')) / 
                props['thermal_conductivity'].get_in('W/(m.K)'))

    def calculate_thermal_expansion_coefficient(self, temperature1: Quantity, 
                                              temperature2: Quantity,
                                              density1: float, density2: float,
                                              density_ave: float) -> float:
        """
        Calculate thermal expansion coefficient (β) using finite differences.

        Args:
            temperature1: First temperature point
            temperature2: Second temperature point
            density1: Density at temperature1
            density2: Density at temperature2
            density_ave: Average density between the two points

        Returns:
            float: Thermal expansion coefficient [1/K]
        """
        d_density = density1 - density2
        d_temp = temperature1.get_in("K") - temperature2.get_in("K")  
        return -(1/density_ave) * (d_density/d_temp)

    def calculate_rayleigh(self, temperature1: Quantity, temperature2: Quantity) -> float:
        """
        Calculate Rayleigh number for natural convection analysis.

        Args:
            temperature1: Cold wall temperature
            temperature2: Hot wall temperature

        Returns:
            float: Rayleigh number (Ra = g·β·ΔT·L³/(ν·α))
        """
        temp_ave = (temperature1.get_in("K") + temperature2.get_in("K"))/2
        props1 = self.get_fluid_properties(temperature1)
        props2 = self.get_fluid_properties(temperature2)
        props_ave = self.get_fluid_properties(Quantity(temp_ave, "K"))

        density = (props1['density'].get_in('kg/m^3') + props2['density'].get_in('kg/m^3'))/2
        viscosity = (props1['dynamic_viscosity'].get_in('Pa.s') + props2['dynamic_viscosity'].get_in('Pa.s'))/2
        conductivity = (props1['thermal_conductivity'].get_in('W/(m.K)') + props2['thermal_conductivity'].get_in('W/(m.K)'))/2

        g = Quantity(9.81, 'm/s^2').get_in('m/s^2')
        delta_T = abs(temperature1.get_in('K') - temperature2.get_in('K'))
        beta = self.calculate_thermal_expansion_coefficient(
            temperature1, temperature2,
            props1['density'].get_in('kg/m^3'),
            props2['density'].get_in('kg/m^3'),
            props_ave['density'].get_in('kg/m^3'))

        return (density * g * beta * delta_T * 
                self.characteristic_length.get_in('m')**3 / 
                (viscosity * conductivity))

    def characteristic_thickness_turbulent(self) -> Quantity:
        """
        Estimate turbulent boundary layer thickness using empirical correlation.

        Returns:
            Quantity: Boundary layer thickness with units (δ = 0.37L/Re^(1/5))
        """
        re = self.calculate_reynolds()
        return Quantity(0.37 * self.characteristic_length.get_in('m') / 
                      (re ** 0.2), 'm')

    def calculate_layers_for_cell_size(self, target_cell_size: Quantity, 
                                     expansion_ratio: float = 1.2) -> int:
        """
        Calculate number of boundary layer cells needed to reach target size.

        Args:
            target_cell_size: Desired cell size at edge of boundary layer
            expansion_ratio: Growth ratio between adjacent layers (default: 1.2)

        Returns:
            int: Number of layers required

        Raises:
            ValueError: If boundary layer thickness is non-positive
        """
        delta = self.characteristic_thickness_turbulent()
        if delta.get_in('m') <= 0:
            raise ValueError("Boundary layer thickness must be positive.")

        num_layers = 0
        current_size = delta.get_in('m')

        while current_size < target_cell_size.get_in('m'):
            current_size *= expansion_ratio
            num_layers += 1

        return num_layers

# Example usage
if __name__ == "__main__":
    # Initialize with water at 300K, 1atm, 2m/s flow in 0.1m channel
    fm = FluidMechanics(FluidsList.Water, 
                        Quantity(300, 'K'), 
                        Quantity(101325, 'Pa'),
                        Quantity(2, 'm/s'), 
                        Quantity(0.1, 'm'))

    print(f"Reynolds: {fm.calculate_reynolds():.1f}")
    print(f"y+: {fm.calculate_y_plus(Quantity(0.1, 'Pa')):.2f}")
    print(f"Prandtl: {fm.calculate_prandtl():.2f}")
    
    # Natural convection between 10°C and 30°C surfaces
    print(f"Rayleigh: {fm.calculate_rayleigh(Quantity(10, 'degC'),Quantity(30, 'degC')):.2e}")
    
    # Mesh recommendations
    print(f"BL thickness: {fm.characteristic_thickness_turbulent():.4f}")
    print(f"Layers needed: {fm.calculate_layers_for_cell_size(Quantity(0.02, 'm'))}")