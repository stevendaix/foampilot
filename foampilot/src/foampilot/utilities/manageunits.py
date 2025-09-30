from __future__ import annotations
from pint import UnitRegistry
from typing import Union, Any

# Initialize pint's unit registry
ureg = UnitRegistry()
Q_ = ureg.Quantity  # Shortcut to define quantities with units


class Quantity:
    """
    Wrapper for Pint quantities to manage values with physical units.

    Provides an interface to store values with units,
    convert them, display them, serialize/deserialize them,
    and perform arithmetic operations.
    """

    def __init__(self, value: Union[float, int], unit: str):
        """Initialize a quantity with a value and unit."""
        self.quantity = Q_(value, unit)

    def set_quantity(self, value: Union[float, int], unit: str) -> None:
        """Update the quantity with a new value and unit."""
        self.quantity = Q_(value, unit)

    def get_in(self, target_unit: str) -> float:
        """
        Convert the quantity to a specified unit and return its magnitude.
        """
        try:
            return self.quantity.to(target_unit).magnitude
        except Exception as e:
            raise ValueError(f"Cannot convert {self.quantity} to {target_unit}: {e}")

    def to(self, target_unit: str) -> Quantity:
        """
        Convert the quantity to a specified unit and return a new Quantity object.
        """
        try:
            q_new = self.quantity.to(target_unit)
            return Quantity(q_new.magnitude, str(q_new.units))
        except Exception as e:
            raise ValueError(f"Cannot convert {self.quantity} to {target_unit}: {e}")

    # ---- Serialization ----
    def as_dict(self) -> dict[str, Any]:
        """
        Return a dictionary representation of the quantity.
        Example: {"value": 10.0, "unit": "meter / second"}
        """
        return {
            "value": float(self.quantity.magnitude),
            "unit": str(self.quantity.units)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Quantity:
        """Create a Quantity object from a dictionary."""
        return cls(data["value"], data["unit"])

    @classmethod
    def from_pint(cls, pint_quantity: Q_) -> Quantity:
        """Create a Quantity from a Pint Quantity."""
        return cls(pint_quantity.magnitude, str(pint_quantity.units))

    # ---- Representation ----
    def __repr__(self) -> str:
        return f"Quantity({self.quantity.magnitude}, '{self.quantity.units}')"

    def __str__(self) -> str:
        return f"{self.quantity.magnitude} {self.quantity.units}"

    # ---- Arithmetic operations ----
    def __add__(self, other: Union[Quantity, float, int]) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity.from_pint(self.quantity + other.quantity)
        return Quantity.from_pint(self.quantity + other)

    def __sub__(self, other: Union[Quantity, float, int]) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity.from_pint(self.quantity - other.quantity)
        return Quantity.from_pint(self.quantity - other)

    def __mul__(self, other: Union[Quantity, float, int]) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity.from_pint(self.quantity * other.quantity)
        return Quantity.from_pint(self.quantity * other)

    def __truediv__(self, other: Union[Quantity, float, int]) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity.from_pint(self.quantity / other.quantity)
        return Quantity.from_pint(self.quantity / other)


# ==== Example usage ====
if __name__ == "__main__":
    import json

    speed = Quantity(10, "m/s")
    print("Speed:", speed)
    print("Speed in km/h:", speed.get_in("km/h"))

    pressure = Quantity(101325, "Pa")
    print("Pressure:", pressure)
    print("Pressure in atm:", pressure.get_in("atm"))

    # Arithmetic
    d1 = Quantity(5, "m")
    d2 = Quantity(3, "m")
    print("Sum of distances:", d1 + d2)

    force = Quantity(10, "N")
    area = Quantity(2, "m^2")
    pressure_calc = force / area
    print("Calculated pressure:", pressure_calc)

    # Serialization
    data = {"speed": speed.as_dict(), "pressure": pressure.as_dict()}
    json_str = json.dumps(data, indent=2)
    print("\nJSON dump:\n", json_str)

    # Deserialization
    loaded_data = json.loads(json_str)
    speed_loaded = Quantity.from_dict(loaded_data["speed"])
    pressure_loaded = Quantity.from_dict(loaded_data["pressure"])
    print("\nReloaded objects:")
    print("Speed:", speed_loaded)
    print("Pressure:", pressure_loaded)