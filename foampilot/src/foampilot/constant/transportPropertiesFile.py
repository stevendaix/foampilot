from foampilot.base.openFOAMFile import OpenFOAMFile
from foampilot.utilities.manageunits import Quantity
from typing import Optional, Any, Dict, Union

class TransportPropertiesFile(OpenFOAMFile):
    """
    Represents the OpenFOAM `transportProperties` configuration file.

    Supports dynamic configuration based on:
      - CaseFieldsManager (for field detection)
      - Transport model (Newtonian, nonNewtonian, etc.)
      - Kinematic viscosity (nu) with unit validation

    Examples
    --------
    >>> # Default Newtonian fluid with nu = 1e-05 m²/s
    >>> transport = TransportPropertiesFile()

    >>> # Custom viscosity using Quantity
    >>> from foampilot.utilities.manageunits import Quantity
    >>> transport = TransportPropertiesFile(nu=Quantity(5e-6, "m^2/s"))

    >>> # Non-Newtonian model (requires additional parameters)
    >>> transport = TransportPropertiesFile(
    >>>     transportModel="nonNewtonian",
    >>>     nu="1e-05",
    >>>     crossPowerLawCoeffs={"nu0": 1e-6, "nuInf": 1e-3, "m": 0.5, "n": 1.0}
    >>> )
    """

    # Unités par défaut pour les propriétés de transport
    DEFAULT_UNITS = {
        "nu": "m^2/s",          # Viscosité cinématique
        "rho": "kg/m^3",        # Masse volumique (optionnelle)
        "mu": "kg/m/s",         # Viscosité dynamique (optionnelle)
        "crossPowerLawCoeffs": {
            "nu0": "m^2/s",     # Viscosité à cisaillement nul
            "nuInf": "m^2/s",    # Viscosité à cisaillement infini
            "m": None,           # Indice de consistance (sans unité)
            "n": None,           # Indice de comportement (sans unité)
        }
    }

    def __init__(
        self,
        parent: Optional[Any] = None,
        transportModel: str = "Newtonian",
        nu: Union[str, Quantity, float] = "1e-05",
        rho: Optional[Union[str, Quantity, float]] = None,
        mu: Optional[Union[str, Quantity, float]] = None,
        crossPowerLawCoeffs: Optional[Dict[str, Union[str, Quantity, float]]] = None,
    ):
        """
        Initialize a `transportProperties` file.

        Parameters
        ----------
        parent : Any, optional
            Parent object with `fields_manager` (for dynamic field detection).
        transportModel : str, optional
            Transport model: "Newtonian" (default), "nonNewtonian", etc.
        nu : Union[str, Quantity, float], optional
            Kinematic viscosity (default: "1e-05" m²/s).
        rho : Union[str, Quantity, float], optional
            Density (required for non-Newtonian models).
        mu : Union[str, Quantity, float], optional
            Dynamic viscosity (alternative to nu).
        crossPowerLawCoeffs : Dict[str, Union[str, Quantity, float]], optional
            Coefficients for CrossPowerLaw model (if transportModel="nonNewtonian").

        Raises
        ------
        ValueError
            If units are incompatible or required parameters are missing.
        """
        self.parent = parent
        self.transportModel = transportModel
        self._nu = self._to_quantity(nu, "nu") # Stockage interne
        self.rho = self._to_quantity(rho, "rho") if rho is not None else None
        self.mu = self._to_quantity(mu, "mu") if mu is not None else None
        self.crossPowerLawCoeffs = self._process_coeffs(crossPowerLawCoeffs) if crossPowerLawCoeffs else None

        # Validate transport model and parameters
        self._validate_parameters()

        # Configure attributes based on transport model
        self._configure_attributes()

        # Override with dynamic fields if parent has CaseFieldsManager
        if self.parent and hasattr(self.parent, "fields_manager"):
            self._configure_from_fields()

        super().__init__(object_name="transportProperties", **self.attributes)

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, value):
        self._nu = self._to_quantity(value, "nu")
        self._configure_attributes()
        # Mise à jour de self.attributes pour l'écriture
        self.attributes = self.to_dict()

    def _to_quantity(self, value: Union[str, Quantity, float], name: str) -> Quantity:
        """
        Convert a value to Quantity with unit validation.
        """
        if isinstance(value, Quantity):
            expected_unit = self.DEFAULT_UNITS.get(name)
            if expected_unit and not value.quantity.check(expected_unit):
                raise ValueError(f"{name} must have units compatible with {expected_unit}")
            return value
        else:
            expected_unit = self.DEFAULT_UNITS.get(name)
            if expected_unit:
                return Quantity(float(value), expected_unit)
            else:
                return float(value)

    def _process_coeffs(self, coeffs: Dict[str, Union[str, Quantity, float]]) -> Dict[str, float]:
        """
        Process CrossPowerLaw coefficients with unit validation.
        """
        processed = {}
        for key, value in coeffs.items():
            if key in self.DEFAULT_UNITS["crossPowerLawCoeffs"]:
                processed[key] = self._to_quantity(value, f"crossPowerLawCoeffs.{key}")
            else:
                processed[key] = float(value)  # No unit check for custom keys
        return processed

    def _validate_parameters(self):
        """
        Validate transport model and required parameters.
        """
        if self.transportModel == "nonNewtonian":
            if self.rho is None:
                raise ValueError("Density (rho) is required for non-Newtonian models")
            if self.crossPowerLawCoeffs is None:
                raise ValueError("crossPowerLawCoeffs are required for non-Newtonian models")

        if self.mu is not None and self.nu is not None:
            raise ValueError("Provide either nu (kinematic viscosity) or mu (dynamic viscosity), not both")

        if self.mu is not None:
            # Convert dynamic viscosity (mu) to kinematic viscosity (nu)
            if self.rho is None:
                raise ValueError("Density (rho) is required to convert dynamic viscosity (mu) to kinematic viscosity (nu)")
            self._nu = self.mu / self.rho
            self.mu = None  # Clear to avoid confusion

    def _configure_attributes(self):
        """
        Configure OpenFOAM attributes based on transport model.
        """
        self.attributes = {
            "transportModel": self.transportModel,
        }
        
        # Ajouter le champ nu si le modèle est Newtonian
        if self.transportModel == "Newtonian":
            self.attributes["nu"] = f"[0 2 -1 0 0 0 0] {self.nu.magnitude if isinstance(self.nu, Quantity) else self.nu}"

        elif self.transportModel == "nonNewtonian":
            self.attributes.update({
                "transportModel": "nonNewtonian",
                "rho": f"[1 -3 0 0 0 0 0] {self.rho.magnitude if isinstance(self.rho, Quantity) else self.rho}",
                "crossPowerLawCoeffs": {
                    k: v.magnitude if isinstance(v, Quantity) else v
                    for k, v in self.crossPowerLawCoeffs.items()
                }
            })
        else:
            raise ValueError(f"Unsupported transport model: {self.transportModel}")

    def _configure_from_fields(self):
        """
        Override configuration based on fields available in CaseFieldsManager.
        """
        if not hasattr(self.parent, "fields_manager"):
            return

        fields_manager = self.parent.fields_manager
        field_names = fields_manager.get_field_names()

        # Example: Detect if energy is activated (T or h field present)
        if "T" in field_names or "h" in field_names:
            # Could adjust transport properties based on energy activation
            pass  # Placeholder for future extensions

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary for OpenFOAM file writing.
        """
        return self.attributes
