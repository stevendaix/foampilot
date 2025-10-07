from dataclasses import dataclass, field
from typing import Dict, Optional, List, Union, Type
from foampilot.utilities.manageunits import Quantity
import warnings

@dataclass
class OpenFOAMVariable:
    """Représente une variable OpenFOAM avec sa valeur par défaut et ses unités."""
    active: bool = False
    default_value: Optional[Quantity] = None
    expected_units: Optional[str] = None  # Unités attendues (ex: "m/s" pour U)
    boundary_conditions: Dict[str, dict] = field(default_factory=dict)  # {patch_name: bc_dict}

    def set_default_value(self, value: Union[float, Quantity]):
        """Définit la valeur par défaut avec validation des unités."""
        if self.expected_units is None:
            raise ValueError(f"No expected units defined for this variable.")

        if isinstance(value, (int, float)):
            value = Quantity(float(value), self.expected_units)
        elif not isinstance(value, Quantity):
            raise ValueError(f"Value must be a float or Quantity. Got {type(value)}.")

        if not value.quantity.check(f"[{self.expected_units}]"):
            warnings.warn(
                f"Unit mismatch for variable: expected [{self.expected_units}], got {value.quantity.units}. "
                f"Converting to {self.expected_units}."
            )
            value = value.convert_to(self.expected_units)

        self.default_value = value

    def get_default_value(self, unit: Optional[str] = None) -> Optional[Quantity]:
        """Récupère la valeur par défaut, avec conversion d'unité si nécessaire."""
        if self.default_value is None:
            return None
        if unit is not None:
            return self.default_value.convert_to(unit)
        return self.default_value

@dataclass
class OpenFOAMVariables:
    """Conteneur centralisé pour toutes les variables OpenFOAM."""
    # Champs de base (toujours actifs)
    U: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=True,
        default_value=Quantity([0, 0, 0], "m/s"),
        expected_units="m/s"
    ))
    p: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=True,
        default_value=Quantity(0, "Pa"),
        expected_units="Pa"
    ))

    # Champs thermiques (activés si energy_activated=True)
    T: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=False,
        default_value=Quantity(300, "K"),
        expected_units="K"
    ))
    alpha: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=False,
        default_value=Quantity(0, "m^2/s"),
        expected_units="m^2/s"
    ))

    # Champs de turbulence (activés selon le modèle)
    k: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=False,
        default_value=Quantity(0.375, "m^2/s^2"),
        expected_units="m^2/s^2"
    ))
    epsilon: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=False,
        default_value=Quantity(0.125, "m^2/s^3"),
        expected_units="m^2/s^3"
    ))
    omega: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=False,
        default_value=Quantity(1, "1/s"),
        expected_units="1/s"
    ))
    nut: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=False,
        default_value=Quantity(0, "m^2/s"),
        expected_units="m^2/s"
    ))

    # Champs multiphasiques (ex: VoF)
    alpha_phase1: OpenFOAMVariable = field(default_factory=lambda: OpenFOAMVariable(
        active=False,
        default_value=Quantity(0, "-"),
        expected_units=None  # Sans unité (fraction volumique)
    ))

    def __post_init__(self):
        """Initialise les valeurs par défaut pour les champs actifs."""
        pass  # Les valeurs sont déjà définies dans les factory functions

    def activate_field(self, field_name: str, value: Optional[Union[float, Quantity]] = None):
        """Active un champ et définit sa valeur par défaut si fournie."""
        if not hasattr(self, field_name):
            raise ValueError(f"Field '{field_name}' not supported.")
        field = getattr(self, field_name)
        field.active = True
        if value is not None:
            field.set_default_value(value)

    def deactivate_field(self, field_name: str):
        """Désactive un champ."""
        if not hasattr(self, field_name):
            raise ValueError(f"Field '{field_name}' not supported.")
        getattr(self, field_name).active = False

    def set_boundary_condition(
        self,
        field_name: str,
        patch_name: str,
        bc_type: str,
        value: Optional[Union[float, Quantity, str]] = None,
        unit: Optional[str] = None
    ):
        """
        Définit une condition aux limites pour un champ et une frontière.

        Args:
            field_name: Nom du champ (ex: "U", "T").
            patch_name: Nom de la frontière (ex: "inlet").
            bc_type: Type de condition (ex: "fixedValue", "zeroGradient").
            value: Valeur de la condition (ex: Quantity, float, ou string comme "uniform 300").
            unit: Unité cible si value est un float.
        """
        if not hasattr(self, field_name):
            raise ValueError(f"Field '{field_name}' not supported.")
        field = getattr(self, field_name)

        bc_dict = {"type": bc_type}
        if value is not None:
            if isinstance(value, str):
                bc_dict["value"] = value  # Ex: "uniform 300"
            else:
                # Convertir en Quantity si nécessaire
                if isinstance(value, (int, float)) and unit is not None:
                    value = Quantity(float(value), unit)
                elif isinstance(value, Quantity):
                    if unit is not None:
                        value = value.convert_to(unit)
                else:
                    raise ValueError(f"Value must be a float, Quantity, or string. Got {type(value)}.")

                if field.expected_units is not None and hasattr(value, "quantity"):
                    # Vérifier la compatibilité des unités
                    if not value.quantity.check(f"[{field.expected_units}]"):
                        warnings.warn(
                            f"Unit mismatch for {field_name}: expected [{field.expected_units}], "
                            f"got {value.quantity.units}. Converting."
                        )
                        value = value.convert_to(field.expected_units)
                    bc_dict["value"] = f"uniform {value.quantity.magnitude:.15g}"

        field.boundary_conditions[patch_name] = bc_dict


class BaseSolver(ABC):
    def __init__(self, ..., energy_activated: bool = False, turbulence_model: str = "kEpsilon"):
        # ...
        self.variables = OpenFOAMVariables()
        self._update_active_variables(energy_activated, turbulence_model)

    def _update_active_variables(self, energy_activated: bool, turbulence_model: str):
        """Met à jour les flags d'activation des variables."""
        self.variables.T.active = energy_activated
        self.variables.alpha.active = energy_activated

        if turbulence_model == "kEpsilon":
            self.variables.k.activate_field()
            self.variables.epsilon.activate_field()
            self.variables.nut.activate_field()
        elif turbulence_model == "kOmega":
            self.variables.k.activate_field()
            self.variables.omega.activate_field()
            self.variables.nut.activate_field()

    def set_default_value(self, field_name: str, value: Union[float, Quantity]):
        """Définit la valeur par défaut pour un champ."""
        if not hasattr(self.variables, field_name):
            raise ValueError(f"Field '{field_name}' not supported.")
        getattr(self.variables, field_name).set_default_value(value)

    def set_boundary_condition(
        self,
        field_name: str,
        patch_name: str,
        bc_type: str,
        value: Optional[Union[float, Quantity, str]] = None,
        unit: Optional[str] = None
    ):
        """Délègue la définition des conditions aux limites à la dataclass."""
        self.variables.set_boundary_condition(field_name, patch_name, bc_type, value, unit)
