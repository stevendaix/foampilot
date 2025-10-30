from foampilot.base.openFOAMFile import OpenFOAMFile
from foampilot.utilities.manageunits import Quantity
from typing import Optional, Dict, Any

class PhysicalPropertiesFile(OpenFOAMFile):
    """
    Represents the OpenFOAM `physicalProperties` configuration file.
    Supports dynamic configuration based on CaseFieldsManager and simulation flags.
    Handles both Boussinesq and non-Boussinesq (perfectGas) cases.
    """

    DEFAULT_UNITS = {
        "mu": "kg/m/s",  # Correction: viscosité dynamique en kg/m/s (ou m²/s pour cinématique)
        "Pr": None,      # Sans unité
        "Cv": "J/kg/K",
        "Cp": "J/kg/K",
        "hf": "J/kg",
        "rho0": "kg/m^3",
        "T0": "K",
        "beta": "1/K",
        "pRef": "Pa"
    }

    def __init__(
        self,
        parent: Optional[Any] = None,
        energy: bool = False,
        boussinesq: bool = False,
        mu: str | Quantity = "1e-05",
        Pr: float = 0.7,
        Cv: str | Quantity = "712",
        Cp: str | Quantity = "1000",
        hf: str | Quantity = "0",
        rho0: str | Quantity = "1",
        T0: str | Quantity = "300",
        beta: str | Quantity = "3e-03",
        pRef: str | Quantity = "100000"
    ):
        super().__init__(object_name="physicalProperties")
        self.parent = parent
        self._energy = energy
        self._boussinesq = boussinesq

        # Store properties with units (using internal names)
        self._mu = self._to_quantity(mu, "mu")
        self._Pr = Pr  # Sans unité
        self._Cv = self._to_quantity(Cv, "Cv")
        self._Cp = self._to_quantity(Cp, "Cp")
        self._hf = self._to_quantity(hf, "hf")
        self._rho0 = self._to_quantity(rho0, "rho0")
        self._T0 = self._to_quantity(T0, "T0")
        self._beta = self._to_quantity(beta, "beta")
        self._pRef = self._to_quantity(pRef, "pRef")

        # Configure attributes based on fields and flags
        self._configure_attributes()

        # Override with dynamic fields if parent has CaseFieldsManager
        if self.parent and hasattr(self.parent, "fields_manager"):
            self._configure_from_fields()

    def _to_quantity(self, value: str | Quantity, name: str) -> Quantity:
        """
        Convert string or Quantity to Quantity with correct units.
        """
        if isinstance(value, Quantity):
            unit = self.DEFAULT_UNITS.get(name)
            if unit and not value.quantity.check(unit):
                raise ValueError(f"{name} must have units compatible with {unit}")
            return value
        else:
            unit = self.DEFAULT_UNITS.get(name)
            return Quantity(float(value), unit) if unit else float(value)

    def _configure_attributes(self):
        """
        Configure attributes based on energy and Boussinesq flags.
        """
        # thermoType block
        thermoType = {
            "type": "heRhoThermo",
            "mixture": "pureMixture",
            "transport": "const",
            "specie": "specie"
        }

        if self._boussinesq:
            thermoType["thermo"] = "eConst"
            thermoType["equationOfState"] = "Boussinesq"
            if self._energy:
                thermoType["energy"] = "sensibleInternalEnergy"
        else:
            thermoType["thermo"] = "hConst"
            thermoType["equationOfState"] = "perfectGas"
            if self._energy:
                thermoType["energy"] = "sensibleEnthalpy"

        # mixture block
        mixture = {
            "specie": {"molWeight": 28.9},
            "transport": {
                "mu": self._mu.magnitude if isinstance(self._mu, Quantity) else self._mu,
                "Pr": self._Pr
            }
        }

        if self._boussinesq:
            mixture["equationOfState"] = {
                "rho0": self._rho0.magnitude if isinstance(self._rho0, Quantity) else self._rho0,
                "T0": self._T0.magnitude if isinstance(self._T0, Quantity) else self._T0,
                "beta": self._beta.magnitude if isinstance(self._beta, Quantity) else self._beta
            }
            if self._energy:
                mixture["thermodynamics"] = {
                    "Cv": self._Cv.magnitude if isinstance(self._Cv, Quantity) else self._Cv,
                    "hf": self._hf.magnitude if isinstance(self._hf, Quantity) else self._hf
                }
        else:
            if self._energy:
                mixture["thermodynamics"] = {
                    "Cp": self._Cp.magnitude if isinstance(self._Cp, Quantity) else self._Cp,
                    "hf": self._hf.magnitude if isinstance(self._hf, Quantity) else self._hf
                }

        # pRef only for non-Boussinesq
        if not self._boussinesq:
            self.attributes["pRef"] = self._pRef.magnitude if isinstance(self._pRef, Quantity) else self._pRef

        # Assign attributes
        self.attributes["thermoType"] = thermoType
        self.attributes["mixture"] = mixture

    def _configure_from_fields(self):
        """
        Override configuration based on fields available in CaseFieldsManager.
        """
        if not hasattr(self.parent, "fields_manager"):
            return

        fields_manager = self.parent.fields_manager
        field_names = fields_manager.get_field_names()

        # Update energy flag if T or h is present
        if "T" in field_names or "h" in field_names or "e" in field_names:
            self.energy = True

        # Update Boussinesq flag if needed (example logic)
        # Note: Boussinesq is typically set manually, but could be inferred from other flags
        # if hasattr(self.parent, "simulation_type"):
        #     self.boussinesq = self.parent.simulation_type == "boussinesq"

        # Re-configure attributes with updated flags
        self._configure_attributes()

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, value: bool):
        self._energy = value
        self._configure_attributes()

    @property
    def boussinesq(self):
        return self._boussinesq

    @boussinesq.setter
    def boussinesq(self, value: bool):
        self._boussinesq = value
        self._configure_attributes()

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value: str | Quantity):
        self._mu = self._to_quantity(value, "mu")
        self._configure_attributes()

    @property
    def Pr(self):
        return self._Pr

    @Pr.setter
    def Pr(self, value: float):
        self._Pr = value
        self._configure_attributes()

    @property
    def Cv(self):
        return self._Cv

    @Cv.setter
    def Cv(self, value: str | Quantity):
        self._Cv = self._to_quantity(value, "Cv")
        self._configure_attributes()

    @property
    def Cp(self):
        return self._Cp

    @Cp.setter
    def Cp(self, value: str | Quantity):
        self._Cp = self._to_quantity(value, "Cp")
        self._configure_attributes()

    @property
    def hf(self):
        return self._hf

    @hf.setter
    def hf(self, value: str | Quantity):
        self._hf = self._to_quantity(value, "hf")
        self._configure_attributes()

    @property
    def rho0(self):
        return self._rho0

    @rho0.setter
    def rho0(self, value: str | Quantity):
        self._rho0 = self._to_quantity(value, "rho0")
        self._configure_attributes()

    @property
    def T0(self):
        return self._T0

    @T0.setter
    def T0(self, value: str | Quantity):
        self._T0 = self._to_quantity(value, "T0")
        self._configure_attributes()

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value: str | Quantity):
        self._beta = self._to_quantity(value, "beta")
        self._configure_attributes()

    @property
    def pRef(self):
        return self._pRef

    @pRef.setter
    def pRef(self, value: str | Quantity):
        self._pRef = self._to_quantity(value, "pRef")
        self._configure_attributes()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary for OpenFOAM file writing.
        """
        return self.attributes