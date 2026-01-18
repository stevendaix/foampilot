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
        return [v for k, v in cls.__dict__.items() if not k.startswith("_") and isinstance(v, str)]

class TransportPropertiesFile(OpenFOAMFile):
    """
    OpenFOAM `transportProperties` file.
    Support dynamic configuration based on:
      - transport model (Newtonian, nonNewtonian, etc.)
      - kinematic viscosity (nu)
      - optional density (rho)
    """

    DEFAULT_UNITS = {
        "nu": "m^2/s",
        "rho": "kg/m^3",
        "crossPowerLawCoeffs": {
            "nu0": "m^2/s",
            "nuInf": "m^2/s",
            "m": None,
            "n": None,
        }
    }

    def __init__(
        self,
        parent: Optional[Any] = None,
        transportModel: str = NonNewtonianModels.NEWTONIAN,
        nu: Union[str, Quantity, float] = "1e-05",
        rho: Optional[Union[str, Quantity, float]] = None,
        crossPowerLawCoeffs: Optional[Dict[str, Union[str, Quantity, float]]] = None,
    ):
        self.parent = parent
        self.transportModel = transportModel
        self._nu = self._to_quantity(nu, "nu")
        self._rho = self._to_quantity(rho, "rho") if rho is not None else None
        self.crossPowerLawCoeffs = self._process_coeffs(crossPowerLawCoeffs) if crossPowerLawCoeffs else None

        self._validate_parameters()
        self._configure_attributes()

        if self.parent and hasattr(self.parent, "fields_manager"):
            self._configure_from_fields()

        super().__init__(object_name="transportProperties", **self.attributes)

    # ------------------ Properties ------------------

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, value):
        self._nu = self._to_quantity(value, "nu")
        self.attributes["nu"] = self._nu.magnitude if isinstance(self._nu, Quantity) else float(self._nu)

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        self._rho = self._to_quantity(value, "rho")
        self.attributes["rho"] = self._rho.magnitude if isinstance(self._rho, Quantity) else float(self._rho)

    # ------------------ Internal Methods ------------------

    def _to_quantity(self, value: Union[str, Quantity, float], name: str) -> Quantity:
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
        processed = {}
        for key, value in coeffs.items():
            if key in self.DEFAULT_UNITS["crossPowerLawCoeffs"]:
                processed[key] = self._to_quantity(value, f"crossPowerLawCoeffs.{key}")
            else:
                processed[key] = float(value)
        return processed

    def _validate_parameters(self):
        if self.transportModel == NonNewtonianModels.NEWTONIAN:
            return  # pas de validation spéciale
        # Pour tous les non-newtoniens
        if self._rho is None:
            raise ValueError(f"Density (rho) is required for {self.transportModel} model")
        if self.crossPowerLawCoeffs is None and self.transportModel in [
            NonNewtonianModels.CROSS_POWER_LAW,
            NonNewtonianModels.BIRD_CARREAU,
            NonNewtonianModels.POWER_LAW,
            NonNewtonianModels.HERSCHEL_BULKLEY,
            NonNewtonianModels.CASSON,
        ]:
            raise ValueError(f"Coefficients are required for {self.transportModel} model")

    def _configure_attributes(self):
        self.attributes = {"transportModel": self.transportModel}

        if self.transportModel == NonNewtonianModels.NEWTONIAN:
            self.attributes["nu"] = self.nu.magnitude if isinstance(self.nu, Quantity) else float(self.nu)
            if self.rho is not None:
                self.attributes["rho"] = self.rho.magnitude if isinstance(self.rho, Quantity) else float(self.rho)
        else:
            # Non-Newtonien
            rho_value = self.rho.magnitude if isinstance(self.rho, Quantity) else float(self.rho)
            self.attributes.update({
                "transportModel": self.transportModel,
                "rho": rho_value,
            })
            if self.crossPowerLawCoeffs:
                self.attributes["crossPowerLawCoeffs"] = {
                    k: v.magnitude if isinstance(v, Quantity) else float(v)
                    for k, v in self.crossPowerLawCoeffs.items()
                }


    def _configure_from_fields(self):
        if not hasattr(self.parent, "fields_manager"):
            return
        fields_manager = self.parent.fields_manager
        field_names = fields_manager.get_field_names()
        if "T" in field_names or "h" in field_names:
            pass  # future extensions

    # ------------------ Public Methods ------------------

    def to_dict(self) -> Dict[str, Any]:
        return self.attributes

    def write(self, filepath: Path):
        self.write_file(filepath)
