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
        self.energy = energy
        self.boussinesq = boussinesq

        # Store properties with units
        self.mu = self._to_quantity(mu, "mu")
        self.Pr = Pr  # Sans unité
        self.Cv = self._to_quantity(Cv, "Cv")
        self.Cp = self._to_quantity(Cp, "Cp")
        self.hf = self._to_quantity(hf, "hf")
        self.rho0 = self._to_quantity(rho0, "rho0")
        self.T0 = self._to_quantity(T0, "T0")
        self.beta = self._to_quantity(beta, "beta")
        self.pRef = self._to_quantity(pRef, "pRef")

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
            "transport": {
                "mu": self.mu.magnitude if isinstance(self.mu, Quantity) else self.mu,
                "Pr": self.Pr
            }
        }

        if self.boussinesq:
            mixture["equationOfState"] = {
                "rho0": self.rho0.magnitude if isinstance(self.rho0, Quantity) else self.rho0,
                "T0": self.T0.magnitude if isinstance(self.T0, Quantity) else self.T0,
                "beta": self.beta.magnitude if isinstance(self.beta, Quantity) else self.beta
            }
            if self.energy:
                mixture["thermodynamics"] = {
                    "Cv": self.Cv.magnitude if isinstance(self.Cv, Quantity) else self.Cv,
                    "hf": self.hf.magnitude if isinstance(self.hf, Quantity) else self.hf
                }
        else:
            if self.energy:
                mixture["thermodynamics"] = {
                    "Cp": self.Cp.magnitude if isinstance(self.Cp, Quantity) else self.Cp,
                    "hf": self.hf.magnitude if isinstance(self.hf, Quantity) else self.hf
                }

        # pRef only for non-Boussinesq
        if not self.boussinesq:
            self.attributes["pRef"] = self.pRef.magnitude if isinstance(self.pRef, Quantity) else self.pRef

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary for OpenFOAM file writing.
        """
        return self.attributes
