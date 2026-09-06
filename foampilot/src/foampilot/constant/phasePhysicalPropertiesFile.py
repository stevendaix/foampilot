from pathlib import Path
from typing import Optional, Any, Union
from foampilot.base.openFOAMFile import OpenFOAMFile
from foampilot.utilities.manageunits import ValueWithUnit


class PhasePhysicalPropertiesFile(OpenFOAMFile):
    """
    OpenFOAM ``constant/physicalProperties.<phase>`` file for VoF solvers.

    Each phase writes its own Newtonian-transport file (``nu`` and ``rho``).
    This replaces the single ``transportProperties`` used by non-VoF solvers.
    """

    DEFAULT_UNITS = {
        "nu": "m^2/s",
        "rho": "kg/m^3",
    }

    def __init__(
        self,
        parent: Optional[Any] = None,
        phase: str = "water",
        nu: Union[str, ValueWithUnit, float] = 1e-6,
        rho: Union[str, ValueWithUnit, float] = 1000,
        thermo_type: Optional[dict] = None,
        mixture: Optional[dict] = None,
    ):
        self.parent = parent
        self.phase = phase
        self._nu = self._to_vwu(nu, "nu")
        self._rho = self._to_vwu(rho, "rho")
        self.thermo_type = thermo_type
        self.mixture = mixture

        obj_name = f"physicalProperties.{phase}"
        super().__init__(object_name=obj_name)
        if thermo_type is not None:
            self.attributes = {
                "thermoType": dict(thermo_type),
                "mixture": dict(mixture or {}),
            }
        else:
            self.attributes = {
                "viscosityModel": "constant",
                "nu": self._nu.magnitude,
                "rho": self._rho.magnitude,
            }

    def _to_vwu(self, value, name):
        expected_unit = self.DEFAULT_UNITS.get(name)
        if isinstance(value, ValueWithUnit):
            return value
        if expected_unit:
            return ValueWithUnit(float(value), expected_unit)
        return ValueWithUnit(float(value), "dimensionless")

    def write(self, filepath: Path):
        self.write_file(filepath)
