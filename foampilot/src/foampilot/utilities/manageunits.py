from __future__ import annotations
from pint import UnitRegistry, ValueWithUnit as PintValueWithUnit
from typing import Union, Any

# Initialize pint's unit registry
ureg = UnitRegistry()
Q_ = ureg.ValueWithUnit  # Shortcut to define quantities with units


class ValueWithUnit:
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
        """Initialize a ValueWithUnit with a value and unit."""
        self.ValueWithUnit: PintValueWithUnit = Q_(value, unit)

    # ---- Accessors ----
    @property
    def magnitude(self) -> float:
        """Return the numeric value of the ValueWithUnit."""
        return float(self.ValueWithUnit.magnitude)

    @property
    def units(self) -> str:
        """Return the unit as a string."""
        return str(self.ValueWithUnit.units)

    def copy(self) -> ValueWithUnit:
        """Return a copy of this ValueWithUnit."""
        return ValueWithUnit.from_pint(self.ValueWithUnit)

    # ---- Conversion ----
    def set_ValueWithUnit(self, value: Union[float, int], unit: str) -> None:
        """Update the ValueWithUnit with a new value and unit."""
        self.ValueWithUnit = Q_(value, unit)

    def get_in(self, target_unit: str) -> float:
        """Convert the ValueWithUnit to a specified unit and return its magnitude."""
        try:
            return self.ValueWithUnit.to(target_unit).magnitude
        except Exception as e:
            raise ValueError(f"Cannot convert {self.ValueWithUnit} to {target_unit}: {e}")

    def to_base_units(self) -> ValueWithUnit:
        """
        Convert the ValueWithUnit to SI base units and return a new ValueWithUnit object.
        Example: 1 bar → 100000 Pa
        """
        try:
            q_base = self.ValueWithUnit.to_base_units()
            return ValueWithUnit.from_pint(q_base)
        except Exception as e:
            raise ValueError(f"Cannot convert {self.ValueWithUnit} to base units: {e}")

    def to_openfoam_dimensions(self) -> str:
        """
        Return the OpenFOAM-style dimensions string for the ValueWithUnit's units.

        Format:
            dimensions      [M L T Θ N J A];

        Example:
            ValueWithUnit(10, "m/s").to_openfoam_dimensions()
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

        for dim, power in self.ValueWithUnit.dimensionality.items():
            if dim in mapping:
                exponents[mapping[dim]] = int(power)

        return f"dimensions      [{' '.join(str(e) for e in exponents)}];"

    # ---- Serialization ----
    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the ValueWithUnit."""
        return {"value": self.magnitude, "unit": self.units}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValueWithUnit:
        """Create a ValueWithUnit object from a dictionary."""
        return cls(data["value"], data["unit"])

    @classmethod
    def from_pint(cls, pint_ValueWithUnit: PintValueWithUnit) -> ValueWithUnit:
        """Create a ValueWithUnit from a Pint ValueWithUnit."""
        return cls(pint_ValueWithUnit.magnitude, str(pint_ValueWithUnit.units))

    # ---- Representation ----
    def __repr__(self) -> str:
        return f"ValueWithUnit({self.magnitude}, '{self.units}')"

    def __str__(self) -> str:
        return f"{self.magnitude} {self.units}"

    # ---- Equality & Hashing ----
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ValueWithUnit):
            return False
        try:
            return self.ValueWithUnit == other.ValueWithUnit
        except Exception:
            return False

    def __hash__(self) -> int:
        return hash((round(self.magnitude, 12), self.units))

    # ---- Arithmetic operations ----
    def _wrap(self, result: PintValueWithUnit) -> ValueWithUnit:
        return ValueWithUnit.from_pint(result)

    def __add__(self, other: Union[ValueWithUnit, float, int]) -> ValueWithUnit:
        return self._wrap(self.ValueWithUnit + (other.ValueWithUnit if isinstance(other, ValueWithUnit) else other))

    def __radd__(self, other: Union[float, int]) -> ValueWithUnit:
        return self._wrap(other + self.ValueWithUnit)

    def __sub__(self, other: Union[ValueWithUnit, float, int]) -> ValueWithUnit:
        return self._wrap(self.ValueWithUnit - (other.ValueWithUnit if isinstance(other, ValueWithUnit) else other))

    def __rsub__(self, other: Union[float, int]) -> ValueWithUnit:
        return self._wrap(other - self.ValueWithUnit)

    def __mul__(self, other: Union[ValueWithUnit, float, int]) -> ValueWithUnit:
        return self._wrap(self.ValueWithUnit * (other.ValueWithUnit if isinstance(other, ValueWithUnit) else other))

    def __rmul__(self, other: Union[float, int]) -> ValueWithUnit:
        return self._wrap(other * self.ValueWithUnit)

    def __truediv__(self, other: Union[ValueWithUnit, float, int]) -> ValueWithUnit:
        return self._wrap(self.ValueWithUnit / (other.ValueWithUnit if isinstance(other, ValueWithUnit) else other))

    def __rtruediv__(self, other: Union[float, int]) -> ValueWithUnit:
        return self._wrap(other / self.ValueWithUnit)


# ==== Example usage ====
if __name__ == "__main__":
    v = ValueWithUnit(10, "m/s")
    print(v, "→", v.to_openfoam_dimensions())

    p = ValueWithUnit(1, "Pa")
    print(p, "→", p.to_openfoam_dimensions())

    rho = ValueWithUnit(1, "kg/m^3")
    print(rho, "→", rho.to_openfoam_dimensions())