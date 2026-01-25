from __future__ import annotations
from pint import UnitRegistry
from typing import Union, Any

# ===== Initialisation Pint =====
ureg = UnitRegistry()
Q_ = ureg.Quantity

class ValueWithUnit:
    """
    Wrapper pour gérer des valeurs physiques avec unités via Pint.

    Fonctionnalités :
    - Stockage de valeurs avec unités
    - Conversion entre unités (get_in)
    - Conversion vers unités SI de base
    - Génération des dimensions OpenFOAM
    - Sérialisation / désérialisation
    - Opérations arithmétiques
    - Représentation Pythonique
    """

    def __init__(self, value: Union[float, int], unit: str):
        self.quantity: Q_ = Q_(value, unit)

    # ---- Accesseurs ----
    @property
    def magnitude(self) -> float:
        return float(self.quantity.magnitude)

    @property
    def units(self) -> str:
        return str(self.quantity.units)

    def copy(self) -> ValueWithUnit:
        return ValueWithUnit(self.magnitude, self.units)

    # ---- Conversion ----
    def set_value(self, value: Union[float, int], unit: str) -> None:
        self.quantity = Q_(value, unit)

    def get_in(self, target_unit: str) -> float:
        try:
            return self.quantity.to(target_unit).magnitude
        except Exception as e:
            raise ValueError(f"Impossible de convertir {self.quantity} en {target_unit}: {e}")

    def to_base_units(self) -> ValueWithUnit:
        try:
            q_base = self.quantity.to_base_units()
            return ValueWithUnit(q_base.magnitude, str(q_base.units))
        except Exception as e:
            raise ValueError(f"Impossible de convertir {self.quantity} en unités de base: {e}")

    # ---- OpenFOAM dimensions ----
    def to_openfoam_dimensions(self) -> str:
        """
        Retourne les dimensions pour OpenFOAM [M L T Θ N J A]
        """
        # Mapping indices OpenFOAM
        mapping = {
            '[mass]': 0,                # M
            '[length]': 1,              # L
            '[time]': 2,                # T
            '[temperature]': 3,         # Θ
            '[substance]': 4,           # N
            '[luminous_intensity]': 5,  # J
            '[current]': 6              # A
        }

        exponents = [0]*7
        for dim, power in self.quantity.dimensionality.items():
            key = str(dim)  # Pint retourne Dimension objects
            if key in mapping:
                exponents[mapping[key]] = int(power)

        return f"dimensions      [{' '.join(str(e) for e in exponents)}];"

    # ---- Sérialisation ----
    def as_dict(self) -> dict[str, Any]:
        return {"value": self.magnitude, "unit": self.units}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValueWithUnit:
        return cls(data["value"], data["unit"])

    # ---- Représentation ----
    def __repr__(self) -> str:
        return f"ValueWithUnit({self.magnitude}, '{self.units}')"

    def __str__(self) -> str:
        return f"{self.magnitude} {self.units}"

    # ---- Comparaison ----
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ValueWithUnit):
            return False
        try:
            return self.quantity == other.quantity
        except Exception:
            return False

    def __hash__(self) -> int:
        return hash((round(self.magnitude, 12), self.units))

    # ---- Opérations arithmétiques ----
    def _wrap(self, result: Q_) -> ValueWithUnit:
        return ValueWithUnit(result.magnitude, str(result.units))

    def __add__(self, other: Union[ValueWithUnit, float, int]) -> ValueWithUnit:
        return self._wrap(self.quantity + (other.quantity if isinstance(other, ValueWithUnit) else other))

    def __radd__(self, other: Union[float, int]) -> ValueWithUnit:
        return self._wrap(other + self.quantity)

    def __sub__(self, other: Union[ValueWithUnit, float, int]) -> ValueWithUnit:
        return self._wrap(self.quantity - (other.quantity if isinstance(other, ValueWithUnit) else other))

    def __rsub__(self, other: Union[float, int]) -> ValueWithUnit:
        return self._wrap(other - self.quantity)

    def __mul__(self, other: Union[ValueWithUnit, float, int]) -> ValueWithUnit:
        return self._wrap(self.quantity * (other.quantity if isinstance(other, ValueWithUnit) else other))

    def __rmul__(self, other: Union[float, int]) -> ValueWithUnit:
        return self._wrap(other * self.quantity)

    def __truediv__(self, other: Union[ValueWithUnit, float, int]) -> ValueWithUnit:
        return self._wrap(self.quantity / (other.quantity if isinstance(other, ValueWithUnit) else other))

    def __rtruediv__(self, other: Union[float, int]) -> ValueWithUnit:
        return self._wrap(other / self.quantity)

    def __neg__(self) -> ValueWithUnit:
        return self._wrap(-self.quantity)


# ==== Exemple d'utilisation ====
if __name__ == "__main__":
    v = ValueWithUnit(10, "m/s")
    print(v, "→", v.to_openfoam_dimensions())

    p = ValueWithUnit(1, "Pa")
    print(p, "→", p.to_openfoam_dimensions())

    rho = ValueWithUnit(1, "kg/m^3")
    print(rho, "→", rho.to_openfoam_dimensions())

    # Exemple d'arithmétique
    a = ValueWithUnit(2, "m")
    b = ValueWithUnit(500, "cm")
    print(a + b)  # 7 m
