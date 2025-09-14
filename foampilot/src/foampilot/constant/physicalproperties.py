# constant/PhysicalPropertiesFile.py

from foampilot.base.openFOAMFile import OpenFOAMFile
from foampilot.utilities.manageunits import Quantity

class PhysicalPropertiesFile(OpenFOAMFile):
    """
    Represents the OpenFOAM `physicalProperties` configuration file.

    Handles both Boussinesq and non-Boussinesq (perfectGas) cases.
    Supports Quantity objects for physical values.
    """

    DEFAULT_UNITS = {
        "mu": "m^2/s",
        "Pr": None,
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
        self.energy = energy
        self.boussinesq = boussinesq

        # Store properties
        self.mu = self._to_float(mu, "mu")
        self.Pr = Pr
        self.Cv = self._to_float(Cv, "Cv")
        self.Cp = self._to_float(Cp, "Cp")
        self.hf = self._to_float(hf, "hf")
        self.rho0 = self._to_float(rho0, "rho0")
        self.T0 = self._to_float(T0, "T0")
        self.beta = self._to_float(beta, "beta")
        self.pRef = self._to_float(pRef, "pRef")

        # Configure attributes
        self._configure_attributes()

    def _to_float(self, value: str | Quantity, name: str) -> float:
        """
        Convert string or Quantity to float in SI units.
        """
        if isinstance(value, Quantity):
            unit = self.DEFAULT_UNITS.get(name)
            if unit is not None and not value.quantity.check(unit):
                raise ValueError(f"{name} must have units compatible with {unit}")
            return float(value.get_in(unit)) if unit else float(value.magnitude)
        return float(value)

    def _configure_attributes(self):
        """
        Set up the attributes for OpenFOAM file based on energy and boussinesq flags.
        """
        # thermoType block
        thermoType = {
            "type": "heRhoThermo",
            "mixture": "pureMixture",
            "transport": "const",
            "specie": "specie"
        }

        if self.boussinesq:
            thermoType["thermo"] = "eConst"
            thermoType["equationOfState"] = "Boussinesq"
            if self.energy:
                thermoType["energy"] = "sensibleInternalEnergy"
        else:
            thermoType["thermo"] = "hConst"
            thermoType["equationOfState"] = "perfectGas"
            if self.energy:
                thermoType["energy"] = "sensibleEnthalpy"

        # mixture block
        mixture = {
            "specie": {"molWeight": 28.9},
            "transport": {"mu": self.mu, "Pr": self.Pr}
        }

        if self.boussinesq:
            if self.energy:
                mixture["thermodynamics"] = {"Cv": self.Cv, "hf": self.hf}
            mixture["equationOfState"] = {"rho0": self.rho0, "T0": self.T0, "beta": self.beta}
        else:
            if self.energy:
                mixture["thermodynamics"] = {"Cp": self.Cp, "hf": self.hf}

        # pRef only for non-Boussinesq
        if not self.boussinesq:
            self.attributes["pRef"] = self.pRef

        # assign attributes
        self.attributes["thermoType"] = thermoType
        self.attributes["mixture"] = mixture