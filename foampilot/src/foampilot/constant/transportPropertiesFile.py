from foampilot.base.openFOAMFile import OpenFOAMFile
from foampilot.utilities.manageunits import ValueWithUnit
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
        return [v for k, v in cls.__dict__.items() if not k.startswith("_") and isinstance(v, str)]


class TransportPropertiesFile(OpenFOAMFile):
    """
    OpenFOAM `transportProperties` file.
    Support Newtonian and non-Newtonian models with dynamic configuration.
    """
    DEFAULT_UNITS = {
        "nu": "m^2/s",
        "rho": "kg/m^3",
        "nu0": "m^2/s",
        "nuInf": "m^2/s",
        "k": "s",
        "m": None,
        "n": None,
        "tau0": None,
        "nuMin": "m^2/s",
        "nuMax": "m^2/s",
    }

    def __init__(
        self,
        parent: Optional[Any] = None,
        transportModel: str = NonNewtonianModels.NEWTONIAN,
        nu: Union[str, ValueWithUnit, float] = "1e-05",
        rho: Optional[Union[str, ValueWithUnit, float]] = None,
        crossPowerLawCoeffs: Optional[Dict[str, Union[str, ValueWithUnit, float]]] = None,
    ):
        self.parent = parent
        self.transportModel = transportModel
        self._nu = self._to_ValueWithUnit(nu, "nu")
        self._rho = self._to_ValueWithUnit(rho, "rho") if rho is not None else None
        self.crossPowerLawCoeffs = self._process_coeffs(crossPowerLawCoeffs) if crossPowerLawCoeffs else None

        # Configure les attributes pour l’écriture
        self._configure_attributes()

        # Appel du parent sans passer attributes pour éviter l’effet de copie initiale
        super().__init__(object_name="transportProperties")

    # ------------------ Properties ------------------

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, value):
        self._nu = self._to_ValueWithUnit(value, "nu")
        if self.transportModel == NonNewtonianModels.NEWTONIAN:
            self.attributes["nu"] = self._nu.magnitude if isinstance(self._nu, ValueWithUnit) else float(self._nu)

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        self._rho = self._to_ValueWithUnit(value, "rho")
        if self.transportModel == NonNewtonianModels.NEWTONIAN or self.transportModel != NonNewtonianModels.NEWTONIAN:
            self.attributes["rho"] = self._rho.magnitude if isinstance(self._rho, ValueWithUnit) else float(self._rho)

    # ------------------ Public Methods ------------------

    def set_non_newtonian(
        self,
        model: str,
        rho: Union[str, float, ValueWithUnit],
        **coeffs: Union[str, float, ValueWithUnit]
    ):
        """Configure le fluide comme non-Newtonien."""
        if model not in NonNewtonianModels.list_models():
            raise ValueError(f"Unsupported non-Newtonian model: {model}")

        self.transportModel = model
        self.rho = self._to_ValueWithUnit(rho, "rho")
        self.crossPowerLawCoeffs = {k: self._to_ValueWithUnit(v, k) for k, v in coeffs.items()}

        # Mettre à jour les attributes pour OpenFOAM
        self._configure_attributes()

    # ------------------ Internal Methods ------------------

    def _to_ValueWithUnit(self, value: Union[str, ValueWithUnit, float], name: str) -> ValueWithUnit:
        if isinstance(value, ValueWithUnit):
            expected_unit = self.DEFAULT_UNITS.get(name)
            if expected_unit and not value.ValueWithUnit.check(expected_unit):
                raise ValueError(f"{name} must have units compatible with {expected_unit}")
            return value
        else:
            expected_unit = self.DEFAULT_UNITS.get(name)
            if expected_unit:
                return ValueWithUnit(float(value), expected_unit)
            else:
                return float(value)

    def _process_coeffs(self, coeffs: Dict[str, Union[str, ValueWithUnit, float]]) -> Dict[str, float]:
        processed = {}
        for key, value in coeffs.items():
            if key in self.DEFAULT_UNITS["crossPowerLawCoeffs"]:
                processed[key] = self._to_ValueWithUnit(value, f"crossPowerLawCoeffs.{key}")
            else:
                processed[key] = float(value)
        return processed

    def _configure_attributes(self):
        """Met à jour self.attributes selon le modèle et les coefficients."""
        self.attributes = {"transportModel": self.transportModel}

        if self.transportModel == NonNewtonianModels.NEWTONIAN:
            # Newtonian
            self.attributes["nu"] = self.nu.magnitude
            if self.rho is not None:
                self.attributes["rho"] = self.rho.magnitude
        else:
            # Non-Newtonien
            if self.rho is None:
                raise ValueError(f"Density (rho) must be provided for non-Newtonian model {self.transportModel}")
            self.attributes["rho"] = self.rho.magnitude

            if self.crossPowerLawCoeffs:
                # Nom exact du dictionnaire attendu par OpenFOAM
                coeffs_dict_name = f"{self.transportModel}Coeffs"
                self.attributes[coeffs_dict_name] = {
                    k: v.magnitude if isinstance(v, ValueWithUnit) else float(v)
                    for k, v in self.crossPowerLawCoeffs.items()
                }
                
    def _configure_from_fields(self):
        if not hasattr(self.parent, "fields_manager"):
            return
        fields_manager = self.parent.fields_manager
        field_names = fields_manager.get_field_names()
        if "T" in field_names or "h" in field_names:
            pass  # future extensions

    def to_dict(self) -> Dict[str, Any]:
        return self.attributes

    def write(self, filepath: Path):
        self.write_file(filepath)
