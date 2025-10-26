from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Union, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from foampilot.solver import Solver

# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Import fichiers constants
from foampilot.constant.transportPropertiesFile import TransportPropertiesFile
from foampilot.constant.turbulencePropertiesFile import TurbulencePropertiesFile
from foampilot.constant.physicalProperties import PhysicalPropertiesFile
from foampilot.constant.gravityFile import GravityFile
from foampilot.constant.pRefFile import PRefFile
from foampilot.constant.radiationProperties import RadiationPropertiesFile, FvModelsFile

class ConstantDirectory:
    def __init__(self, solver: Solver, *, with_radiation: bool = False):
        """
        Initialize constant directory manager.

        Args:
            solver: Base solver instance
            with_radiation: Enable radiation by default
        """
        self.solver = solver
        self.with_radiation = with_radiation

        # Initialisation des fichiers constants
        self._transportProperties = TransportPropertiesFile(self.solver)
        self._turbulenceProperties = TurbulencePropertiesFile(self.solver)
        self._physicalProperties = PhysicalPropertiesFile(self.solver)
        self._gravity = GravityFile(self.solver)
        self._pRef = PRefFile(self.solver)

        # Radiation files
        self._radiation: Optional[RadiationPropertiesFile] = None
        self._fvmodels: Optional[FvModelsFile] = None

        if with_radiation:
            self.enable_radiation()

    # Properties
    @property
    def transportProperties(self):
        return self._transportProperties

    @property
    def turbulenceProperties(self):
        return self._turbulenceProperties

    @property
    def physicalProperties(self):
        return self._physicalProperties

    @property
    def gravity(self):
        return self._gravity

    @property
    def pRef(self):
        return self._pRef

    @property
    def radiation(self):
        return self._radiation

    # Radiation management
    def enable_radiation(self, model: str = "P1", **kwargs):
        self.with_radiation = True
        self._radiation = RadiationPropertiesFile(self.solver, model=model, **kwargs)
        self._fvmodels = FvModelsFile(self.solver)
        logger.info(f"Radiation enabled with model: {model}")

    def disable_radiation(self):
        self.with_radiation = False
        self._radiation = None
        self._fvmodels = None
        logger.info("Radiation disabled")

    # Write files
    def write(self):
        constant_path = Path(self.solver.case_path) / "constant"
        constant_path.mkdir(parents=True, exist_ok=True)

        # Always write turbulence
        self._turbulenceProperties.write(constant_path / "turbulenceProperties")

        # Transport / Physical
        if getattr(self.solver, "compressible", False):
            self._physicalProperties.write(constant_path / "physicalProperties")
            self._pRef.write(constant_path / "pRef")
        else:
            self._transportProperties.write(constant_path / "transportProperties")

        # Gravity
        if getattr(self.solver, "with_gravity", False):
            self._gravity.write(constant_path / "g")
            # Update p → p_rgh if necessary
            if hasattr(self.solver.fields_manager.fields, "p") and "p_rgh" not in self.solver.fields_manager.fields:
                self.solver.fields_manager.fields["p_rgh"] = self.solver.fields_manager.fields.pop("p")

        # Radiation
        if self.with_radiation:
            if self._radiation is None:
                self.enable_radiation()
            self._radiation.write(constant_path / "radiationProperties")
            self._fvmodels.write(constant_path / "fvModels")

        logger.info(f"Constant directory written to {constant_path}")
        return self
