from pint import UnitRegistry

# Initialize pint's unit registry
ureg = UnitRegistry()
Q_ = ureg.Quantity  # Shortcut to define quantities with units


class Quantity:
    """
    Wrapper for Pint quantities to manage values with physical units.

    This class provides a convenient interface to store values
    with units, convert them to other units, and display them.

    Attributes
    ----------
    quantity : pint.Quantity
        The quantity with its numerical value and associated unit.
    """

    def __init__(self, value: float, unit: str):
        """
        Initialize a quantity with a value and unit.

        Parameters
        ----------
        value : float
            The numerical value of the quantity.
        unit : str
            The unit as a string (e.g., ``"m/s"``, ``"Pa"``, ``"kg"``).
        """
        self.quantity = Q_(value, unit)

    def set_quantity(self, value: float, unit: str):
        """
        Update the quantity with a new value and unit.

        Parameters
        ----------
        value : float
            The numerical value.
        unit : str
            The associated unit (e.g., ``"m"``, ``"kg"``, ``"m/s"``).
        """
        self.quantity = Q_(value, unit)

    def get_in(self, target_unit: str) -> float:
        """
        Convert the quantity to a specified unit and return its value.

        Parameters
        ----------
        target_unit : str
            The target unit for conversion (e.g., ``"ft/s"``, ``"inch"``, ``"psi"``).

        Returns
        -------
        float
            The numerical value converted to the target unit.
        """
        return self.quantity.to(target_unit).magnitude

    def __repr__(self) -> str:
        """
        Return a string representation of the quantity.

        Returns
        -------
        str
            The numerical value with its unit (e.g., ``"10.0 m/s"``).
        """
        return f"{self.quantity.magnitude} {self.quantity.units}"


# Example usage
if __name__ == "__main__":
    # Initialize with speed in meters per second
    speed = Quantity(10, "m/s")
    print("Speed:", speed)  # Output: "10.0 m / s"

    # Convert speed to kilometers per hour
    speed_kmh = speed.get_in("km/h")
    print("Speed in km/h:", speed_kmh)  # Output: 36.0

    # Initialize with pressure in pascals
    pressure = Quantity(101325, "Pa")
    print("Pressure:", pressure)  # Output: "101325.0 Pa"

    # Convert pressure to atmospheres
    pressure_atm = pressure.get_in("atm")
    print("Pressure in atm:", pressure_atm)  # Output: 1.0