from dataclasses import dataclass, field
from typing import Dict, Optional, List, Union
from foampilot.utilities.manageunits import ValueWithUnit
from abc import ABC
import warnings


# ============================================================
#  SUPPORT : Modèles de turbulence
# ============================================================

TURBULENCE_MODELS = {
    "kEpsilon": ["k", "epsilon", "nut"],
    "RNGkEpsilon": ["k", "epsilon", "nut"],
    "RealizableKE": ["k", "epsilon", "nut"],
    "kOmega": ["k", "omega", "nut"],
    "SST": ["k", "omega", "nut"],
    "SpalartAllmaras": ["nut"],
}


# ============================================================
#  VARIABLE OPENFOAM
# ============================================================

@dataclass
class OpenFOAMVariable:
    """Représente une variable OpenFOAM avec valeur, unités et BC."""
    active: bool = False
    default_value: Optional[ValueWithUnit] = None
    expected_units: Optional[str] = None
    foam_name: Optional[str] = None  # nom OpenFOAM (alpha.phase1 par ex.)
    boundary_conditions: Dict[str, dict] = field(default_factory=dict)

    # --------------------------------------------------------

    def activate(self):
        """Active ce champ."""
        self.active = True

    # --------------------------------------------------------

    def set_default_value(self, value: Union[float, ValueWithUnit]):
        """Définit et valide la valeur par défaut avec gestion des unités."""
        if self.expected_units is None and not isinstance(value, ValueWithUnit):
            raise ValueError(
                f"Cannot assign float to a dimensionless variable without ValueWithUnit: {value}"
            )

        # Conversion float → ValueWithUnit
        if isinstance(value, (int, float)):
            if self.expected_units is None:
                raise ValueError("This variable has no expected units. Provide a ValueWithUnit.")
            value = ValueWithUnit(float(value), self.expected_units)

        if not isinstance(value, ValueWithUnit):
            raise ValueError(f"Value must be float or ValueWithUnit. Got {type(value)}.")

        # Vérif unités
        if self.expected_units is not None:
            try:
                value = value.convert_to(self.expected_units)
            except Exception:
                warnings.warn(
                    f"Unit mismatch for default value: expected [{self.expected_units}], got {value.ValueWithUnit.units}. "
                    "Attempting conversion."
                )
                value = value.convert_to(self.expected_units)

        self.default_value = value

    # --------------------------------------------------------

    def get_default_value(self, unit: Optional[str] = None) -> Optional[ValueWithUnit]:
        if self.default_value is None:
            return None
        return self.default_value.convert_to(unit) if unit else self.default_value


# ============================================================
# CONTAINER : Toutes les variables OpenFOAM
# ============================================================

@dataclass
class OpenFOAMVariables:

    # Champs toujours présents
    U: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=True,
        default_value=ValueWithUnit([0, 0, 0], "m/s"),
        expected_units="m/s",
        foam_name="U"
    ))

    p: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=True,
        default_value=ValueWithUnit(0, "Pa"),
        expected_units="Pa",
        foam_name="p"
    ))

    # Thermique
    T: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=False,
        default_value=ValueWithUnit(300, "K"),
        expected_units="K",
        foam_name="T"
    ))

    alpha: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=False,
        default_value=ValueWithUnit(0, "m^2/s"),
        expected_units="m^2/s",
        foam_name="alpha"
    ))

    # Turbulence
    k: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=False,
        default_value=ValueWithUnit(0.375, "m^2/s^2"),
        expected_units="m^2/s^2",
        foam_name="k"
    ))

    epsilon: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=False,
        default_value=ValueWithUnit(0.125, "m^2/s^3"),
        expected_units="m^2/s^3",
        foam_name="epsilon"
    ))

    omega: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=False,
        default_value=ValueWithUnit(1, "1/s"),
        expected_units="1/s",
        foam_name="omega"
    ))

    nut: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=False,
        default_value=ValueWithUnit(0, "m^2/s"),
        expected_units="m^2/s",
        foam_name="nut"
    ))

    # Multiphase / VOF
    alpha_phase1: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=False,
        default_value=ValueWithUnit(0, "-"),
        expected_units=None,
        foam_name="alpha.phase1"
    ))

    # --------------------------------------------------------

    def activate_field(self, field_name: str, value=None):
        if not hasattr(self, field_name):
            raise ValueError(f"Field '{field_name}' not supported.")
        field = getattr(self, field_name)
        field.activate()
        if value is not None:
            field.set_default_value(value)

    # --------------------------------------------------------

    def deactivate_field(self, field_name: str):
        if not hasattr(self, field_name):
            raise ValueError(f"Field '{field_name}' not supported.")
        getattr(self, field_name).active = False

    # --------------------------------------------------------

    def active_fields(self) -> Dict[str, OpenFOAMVariable]:
        """Retourne un dict {nom_py: variable} pour tous les champs actifs."""
        return {
            name: v for name, v in self.__dict__.items()
            if isinstance(v, OpenFOAMVariable) and v.active
        }

    # --------------------------------------------------------

    def set_boundary_condition(
        self,
        field_name: str,
        patch_name: str,
        bc_type: str,
        value: Optional[Union[float, ValueWithUnit, str]] = None,
        unit: Optional[str] = None
    ):
        """Définit une BC avec support vectoriel."""
        if not hasattr(self, field_name):
            raise ValueError(f"Field '{field_name}' not supported.")

        field = getattr(self, field_name)
        bc_dict = {"type": bc_type}

        if value is not None:

            # Cas string → direct
            if isinstance(value, str):
                bc_dict["value"] = value

            else:
                # Float → ValueWithUnit
                if isinstance(value, (int, float)):
                    if unit is None:
                        raise ValueError(f"Float BC must specify unit for {field_name}.")
                    value = ValueWithUnit(float(value), unit)

                # ValueWithUnit → compatibilité
                if field.expected_units:
                    value = value.convert_to(field.expected_units)

                # BC vectorielle
                mag = value.ValueWithUnit.magnitude
                if isinstance(mag, (list, tuple)):
                    components = " ".join(f"{c:.12g}" for c in mag)
                    bc_dict["value"] = f"uniform ({components})"
                else:
                    bc_dict["value"] = f"uniform {mag:.12g}"

        field.boundary_conditions[patch_name] = bc_dict


# ============================================================
#  BASE SOLVER
# ============================================================

class BaseSolver(ABC):
    def __init__(self, *, energy_activated=False, turbulence_model="kEpsilon"):
        self.variables = OpenFOAMVariables()
        self._update_active_variables(energy_activated, turbulence_model)

    # --------------------------------------------------------

    def _update_active_variables(self, energy_activated: bool, turbulence_model: str):

        # Thermique
        if energy_activated:
            self.variables.activate_field("T")
            self.variables.activate_field("alpha")

        # Turbulence
        for field_name in TURBULENCE_MODELS.get(turbulence_model, []):
            self.variables.activate_field(field_name)

    # --------------------------------------------------------

    def set_default_value(self, field_name: str, value: Union[float, ValueWithUnit]):
        if not hasattr(self.variables, field_name):
            raise ValueError(f"Field '{field_name}' not supported.")
        getattr(self.variables, field_name).set_default_value(value)

    # --------------------------------------------------------

    def set_boundary_condition(self, *args, **kwargs):
        self.variables.set_boundary_condition(*args, **kwargs)
