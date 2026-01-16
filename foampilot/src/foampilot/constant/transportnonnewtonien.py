from foampilot.base.openFOAMFile import OpenFOAMFile
from foampilot.utilities.manageunits import Quantity
from typing import Optional, Any, Dict, Union
from pathlib import Path


class NonNewtonianModels:
    """Modèles non-newtoniens OpenFOAM (section 7.3)."""

    NEWTONIAN = "Newtonian"
    BIRD_CARREAU = "BirdCarreau"
    CROSS_POWER_LAW = "CrossPowerLaw"
    POWER_LAW = "powerLaw"
    HERSCHEL_BULKLEY = "HerschelBulkley"
    CASSON = "Casson"
    STRAIN_RATE = "strainRateFunction"

    @classmethod
    def list_models(cls):
        return [
            v for k, v in cls.__dict__.items()
            if not k.startswith("_") and isinstance(v, str)
        ]


class TransportPropertiesFile(OpenFOAMFile):
    """
    Gère le fichier OpenFOAM `transportProperties`.

    Supporte :
    - Newtonian
    - Tous les modèles non-newtoniens OpenFOAM
    - nu / mu avec conversion automatique
    - Validation dimensionnelle
    - Intégration CaseFieldsManager
    """

    # --- UNITÉS DE RÉFÉRENCE ---
    DEFAULT_UNITS = {
        "nu": "m^2/s",
        "mu": "kg/m/s",
        "rho": "kg/m^3",
    }

    # --- UNITÉS DES COEFFICIENTS PAR MODÈLE ---
    MODEL_COEFF_UNITS = {
        NonNewtonianModels.BIRD_CARREAU: {
            "nu0": "m^2/s",
            "nuInf": "m^2/s",
            "k": "s",
            "n": None,
        },
        NonNewtonianModels.CROSS_POWER_LAW: {
            "nu0": "m^2/s",
            "nuInf": "m^2/s",
            "m": None,
            "n": None,
        },
        NonNewtonianModels.POWER_LAW: {
            "k": "m^2/s",
            "n": None,
        },
        NonNewtonianModels.HERSCHEL_BULKLEY: {
            "k": "m^2/s",
            "n": None,
            "tau0": "kg/m/s^2",
        },
        NonNewtonianModels.CASSON: {
            "mu": "kg/m/s",
            "tau0": "kg/m/s^2",
        },
    }

    def __init__(
        self,
        parent: Optional[Any] = None,
        transportModel: str = NonNewtonianModels.NEWTONIAN,
        nu: Optional[Union[str, Quantity, float]] = "1e-05",
        mu: Optional[Union[str, Quantity, float]] = None,
        rho: Optional[Union[str, Quantity, float]] = None,
    ):
        self.parent = parent
        self._transportModel = transportModel

        self._nu = self._to_quantity(nu, "nu") if nu is not None else None
        self._mu = self._to_quantity(mu, "mu") if mu is not None else None
        self._rho = self._to_quantity(rho, "rho") if rho is not None else None

        self._modelCoeffs: Dict[str, Any] = {}

        # Conversion mu -> nu si nécessaire
        self._resolve_viscosity_definition()

        # Validation globale
        self._validate_base_parameters()

        # Construction initiale
        self._configure_attributes()

        # Intégration CaseFieldsManager
        if self.parent and hasattr(self.parent, "fields_manager"):
            self._configure_from_fields()

        # CONTRAT OpenFOAMFile : on passe attributes au super
        super().__init__(object_name="transportProperties", **self.attributes)

    # ------------------------------------------------------------------
    # Conversion & validation
    # ------------------------------------------------------------------

    def _to_quantity(self, value: Any, name: str) -> Quantity:
        if value is None:
            return None

        if isinstance(value, Quantity):
            expected = self.DEFAULT_UNITS.get(name)
            if expected and not value.quantity.check(expected):
                raise ValueError(f"{name} must have units compatible with {expected}")
            return value

        unit = self.DEFAULT_UNITS.get(name)
        return Quantity(float(value), unit) if unit else float(value)

    def _resolve_viscosity_definition(self):
        """Gère nu / mu / rho."""
        if self._nu and self._mu:
            raise ValueError("Specify either nu or mu, not both")

        if self._mu:
            if not self._rho:
                raise ValueError("rho is required to convert mu to nu")
            self._nu = self._mu / self._rho
            self._mu = None

    def _validate_base_parameters(self):
        if self._transportModel not in NonNewtonianModels.list_models():
            raise ValueError(
                f"Unknown transport model: {self._transportModel}"
            )

        if self._transportModel != NonNewtonianModels.NEWTONIAN:
            if not self._rho:
                raise ValueError("rho is required for non-Newtonian models")

    # ------------------------------------------------------------------
    # API publique non-newtonienne
    # ------------------------------------------------------------------

    def set_non_newtonian(self, model: str, **coeffs):
        if model == NonNewtonianModels.NEWTONIAN:
            raise ValueError("Use nu directly for Newtonian model")

        self._transportModel = model
        self._validate_model_coeffs(model, coeffs)
        self._modelCoeffs = self._process_model_coeffs(model, coeffs)

        self._configure_attributes()

    def _validate_model_coeffs(self, model: str, coeffs: Dict[str, Any]):
        if model not in self.MODEL_COEFF_UNITS:
            raise ValueError(f"No coefficient definition for model {model}")

        required = self.MODEL_COEFF_UNITS[model]
        missing = [k for k in required if k not in coeffs]
        if missing:
            raise ValueError(f"Missing coefficients for {model}: {missing}")

    def _process_model_coeffs(self, model: str, coeffs: Dict[str, Any]) -> Dict[str, float]:
        processed = {}

        for name, value in coeffs.items():
            expected_unit = self.MODEL_COEFF_UNITS[model].get(name)

            if isinstance(value, Quantity):
                if expected_unit and not value.quantity.check(expected_unit):
                    raise ValueError(
                        f"{model} coefficient '{name}' must have units {expected_unit}"
                    )
                processed[name] = value.magnitude
            else:
                processed[name] = float(value)

        return processed

    # ------------------------------------------------------------------
    # OpenFOAM attribute assembly
    # ------------------------------------------------------------------

    def _configure_attributes(self):
        attrs = {
            "transportModel": self._transportModel
        }

        if self._transportModel == NonNewtonianModels.NEWTONIAN:
            if not self._nu:
                raise ValueError("nu must be defined for Newtonian model")

            attrs["nu"] = f"[0 2 -1 0 0 0 0] {self._nu.magnitude}"

        else:
            attrs["rho"] = f"[1 -3 0 0 0 0 0] {self._rho.magnitude}"

            coeffs_key = f"{self._transportModel}Coeffs"
            attrs[coeffs_key] = {
                "nu0": self._nu.magnitude,
                **self._modelCoeffs,
            }

        self.attributes = attrs

    # ------------------------------------------------------------------
    # Intégration CaseFieldsManager
    # ------------------------------------------------------------------

    def _configure_from_fields(self):
        fm = self.parent.fields_manager
        fields = fm.get_field_names()

        # Exemple : ajustements futurs
        if "T" in fields or "h" in fields:
            pass

    # ------------------------------------------------------------------
    # Écriture
    # ------------------------------------------------------------------

    def write(self, filepath: Path):
        self.write_file(filepath)