from __future__ import annotations
from pint import UnitRegistry, Quantity as PintQuantity
from typing import Union, Any

# Initialize pint's unit registry
ureg = UnitRegistry()
Q_ = ureg.Quantity  # Shortcut to define quantities with units


class Quantity:
    """
    Wrapper for Pint quantities to manage values with physical units.

    Features:
    - Store values with units
    - Convert between units (get_in)
    - Convert to SI base units
    - Export OpenFOAM dimensions
    - Serialize/deserialize
    - Perform arithmetic operations
    - Provide Pythonic representations
    """

    def __init__(self, value: Union[float, int], unit: str):
        """Initialize a quantity with a value and unit."""
        self.quantity: PintQuantity = Q_(value, unit)

    # ---- Accessors ----
    @property
    def magnitude(self) -> float:
        """Return the numeric value of the quantity."""
        return float(self.quantity.magnitude)

    @property
    def units(self) -> str:
        """Return the unit as a string."""
        return str(self.quantity.units)

    def copy(self) -> Quantity:
        """Return a copy of this quantity."""
        return Quantity.from_pint(self.quantity)

    # ---- Conversion ----
    def set_quantity(self, value: Union[float, int], unit: str) -> None:
        """Update the quantity with a new value and unit."""
        self.quantity = Q_(value, unit)

    def get_in(self, target_unit: str) -> float:
        """Convert the quantity to a specified unit and return its magnitude."""
        try:
            return self.quantity.to(target_unit).magnitude
        except Exception as e:
            raise ValueError(f"Cannot convert {self.quantity} to {target_unit}: {e}")

    def to_base_units(self) -> Quantity:
        """
        Convert the quantity to SI base units and return a new Quantity object.
        Example: 1 bar → 100000 Pa
        """
        try:
            q_base = self.quantity.to_base_units()
            return Quantity.from_pint(q_base)
        except Exception as e:
            raise ValueError(f"Cannot convert {self.quantity} to base units: {e}")

    def to_openfoam_dimensions(self) -> str:
        """
        Return the OpenFOAM-style dimensions string for the quantity's units.

        Format:
            dimensions      [M L T Θ N J A];

        Example:
            Quantity(10, "m/s").to_openfoam_dimensions()
            -> "dimensions      [0 1 -1 0 0 0 0];"
        """
        mapping = {
            "[mass]": 0,                # M
            "[length]": 1,              # L
            "[time]": 2,                # T
            "[temperature]": 3,         # Θ
            "[substance]": 4,           # N
            "[luminous_intensity]": 5,  # J
            "[current]": 6              # A
        }

        exponents = [0] * 7  # Default exponents

        for dim, power in self.quantity.dimensionality.items():
            if dim in mapping:
                exponents[mapping[dim]] = int(power)

        return f"dimensions      [{' '.join(str(e) for e in exponents)}];"

    # ---- Serialization ----
    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the quantity."""
        return {"value": self.magnitude, "unit": self.units}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Quantity:
        """Create a Quantity object from a dictionary."""
        return cls(data["value"], data["unit"])

    @classmethod
    def from_pint(cls, pint_quantity: PintQuantity) -> Quantity:
        """Create a Quantity from a Pint Quantity."""
        return cls(pint_quantity.magnitude, str(pint_quantity.units))

    # ---- Representation ----
    def __repr__(self) -> str:
        return f"Quantity({self.magnitude}, '{self.units}')"

    def __str__(self) -> str:
        return f"{self.magnitude} {self.units}"

    # ---- Equality & Hashing ----
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return False
        try:
            return self.quantity == other.quantity
        except Exception:
            return False

    def __hash__(self) -> int:
        return hash((round(self.magnitude, 12), self.units))

    # ---- Arithmetic operations ----
    def _wrap(self, result: PintQuantity) -> Quantity:
        return Quantity.from_pint(result)

    def __add__(self, other: Union[Quantity, float, int]) -> Quantity:
        return self._wrap(self.quantity + (other.quantity if isinstance(other, Quantity) else other))

    def __radd__(self, other: Union[float, int]) -> Quantity:
        return self._wrap(other + self.quantity)

    def __sub__(self, other: Union[Quantity, float, int]) -> Quantity:
        return self._wrap(self.quantity - (other.quantity if isinstance(other, Quantity) else other))

    def __rsub__(self, other: Union[float, int]) -> Quantity:
        return self._wrap(other - self.quantity)

    def __mul__(self, other: Union[Quantity, float, int]) -> Quantity:
        return self._wrap(self.quantity * (other.quantity if isinstance(other, Quantity) else other))

    def __rmul__(self, other: Union[float, int]) -> Quantity:
        return self._wrap(other * self.quantity)

    def __truediv__(self, other: Union[Quantity, float, int]) -> Quantity:
        return self._wrap(self.quantity / (other.quantity if isinstance(other, Quantity) else other))

    def __rtruediv__(self, other: Union[float, int]) -> Quantity:
        return self._wrap(other / self.quantity)


# ==== Example usage ====
if __name__ == "__main__":
    v = Quantity(10, "m/s")
    print(v, "→", v.to_openfoam_dimensions())

    p = Quantity(1, "Pa")
    print(p, "→", p.to_openfoam_dimensions())

    rho = Quantity(1, "kg/m^3")
    print(rho, "→", rho.to_openfoam_dimensions())