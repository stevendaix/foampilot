from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Dict, Union, List, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from foampilot.solver import Solver

# Configuration du logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Import des fichiers de configuration
from foampilot.constant.transportPropertiesFile import TransportPropertiesFile
from foampilot.constant.turbulencePropertiesFile import TurbulencePropertiesFile
from foampilot.constant.physicalProperties import PhysicalPropertiesFile
from foampilot.constant.gravityFile import GravityFile
from foampilot.constant.pRefFile import PRefFile
from foampilot.constant.radiationPropertiesFile import RadiationPropertiesFile, FvModelsFile

class ConstantDirectory:
    """
    Manager for the 'constant' directory in an OpenFOAM case.
    Handles all constant directory files and maintains backward compatibility.

    Features:
    - Automatic p → p_rgh conversion when gravity is enabled
    - Robust file writing with multiple fallback methods
    - Radiation support with dynamic configuration
    - Backward compatibility with solver.constant.write()
    """

    def __init__(self, parent: Any, *, with_radiation: bool = False):
        """
        Initialize the constant directory manager.

        Args:
            parent: Parent solver instance
            with_radiation: Enable radiation by default
        """
        self.parent = parent
        self.with_radiation = with_radiation

        # Initialize default file instances with parent reference
        self._transportProperties = TransportPropertiesFile(parent=self.parent)
        self._turbulenceProperties = TurbulencePropertiesFile(parent=self.parent)
        self._physicalProperties = PhysicalPropertiesFile(parent=self.parent)
        self._gravity = GravityFile(parent=self.parent)
        self._pRef = PRefFile(parent=self.parent)

        # Radiation components (optional)
        self._radiation: Optional[RadiationPropertiesFile] = None
        self._fvmodels: Optional[FvModelsFile] = None

        if with_radiation:
            self.enable_radiation()

    # ======================
    # Property Accessors
    # ======================
    @property
    def transportProperties(self) -> TransportPropertiesFile:
        return self._transportProperties

    @transportProperties.setter
    def transportProperties(self, value: TransportPropertiesFile) -> None:
        self._transportProperties = value

    @property
    def turbulenceProperties(self) -> TurbulencePropertiesFile:
        return self._turbulenceProperties

    @turbulenceProperties.setter
    def turbulenceProperties(self, value: TurbulencePropertiesFile) -> None:
        self._turbulenceProperties = value

    @property
    def physicalProperties(self) -> PhysicalPropertiesFile:
        return self._physicalProperties

    @physicalProperties.setter
    def physicalProperties(self, value: PhysicalPropertiesFile) -> None:
        self._physicalProperties = value

    @property
    def gravity(self) -> GravityFile:
        return self._gravity

    @gravity.setter
    def gravity(self, value: GravityFile) -> None:
        self._gravity = value
        if hasattr(self.parent, "with_gravity"):
            self.parent.with_gravity = True

    @property
    def pRef(self) -> PRefFile:
        return self._pRef

    @pRef.setter
    def pRef(self, value: PRefFile) -> None:
        self._pRef = value

    @property
    def radiation(self) -> Optional[RadiationPropertiesFile]:
        return self._radiation

    @radiation.setter
    def radiation(self, value: Optional[RadiationPropertiesFile]) -> None:
        self._radiation = value
        self.with_radiation = value is not None

    # ======================
    # Radiation Management
    # ======================
    def enable_radiation(self, model: str = "P1", **kwargs) -> None:
        """
        Enable radiation modeling with specified parameters.

        Args:
            model: Radiation model ("P1" or "fvDOM")
            **kwargs: Additional model parameters
        """
        self.with_radiation = True
        self._radiation = RadiationPropertiesFile(parent=self.parent, model=model, **kwargs)
        self._fvmodels = FvModelsFile(parent=self.parent)
        logger.info(f"Radiation enabled with model: {model}")

    def disable_radiation(self) -> None:
        """Disable radiation modeling."""
        self.with_radiation = False
        self._radiation = None
        self._fvmodels = None
        logger.info("Radiation disabled")

    # ======================
    # File Writing Utilities
    # ======================
    def _safe_write(self, obj: Any, target_path: Union[str, Path]) -> None:
        """
        Safely write a file using multiple fallback methods.

        Args:
            obj: File object to write
            target_path: Destination path
        """
        target = Path(target_path)
        logger.debug(f"Writing {getattr(obj, 'object_name', type(obj))} to {target}")

        try:
            # Try direct write with path
            if hasattr(obj, 'write') and callable(obj.write):
                obj.write(target)
                return

            # Try write with parent directory
            if hasattr(obj, 'write') and callable(obj.write):
                obj.write(target.parent)
                return

            # Fallback to write_file if available
            if hasattr(obj, 'write_file') and callable(obj.write_file):
                obj.write_file(target)
                return

            logger.error(f"Cannot write {obj}: no compatible write method found")
            raise RuntimeError(f"Cannot write {obj}: no compatible write method")

        except Exception as e:
            logger.error(f"Failed to write {obj}: {str(e)}")
            raise

    def _update_pressure_field(self) -> None:
        """
        Update pressure field from p to p_rgh when gravity is enabled.
        """
        if hasattr(self.parent, "fields_manager"):
            fields = self.parent.fields_manager
            if hasattr(fields, "fields") and "p" in fields.fields and "p_rgh" not in fields.fields:
                fields.fields["p_rgh"] = fields.fields.pop("p")
                logger.info("Updated pressure field: p → p_rgh (gravity enabled)")

    # ======================
    # Main Write Method
    # ======================
    def write(self) -> "ConstantDirectory":
        """
        Write all constant directory files and return self for chaining.
        Maintains backward compatibility with solver.constant.write().
        """
        base_path = Path(self.parent.case_path)
        constant_path = base_path / "constant"
        constant_path.mkdir(parents=True, exist_ok=True)

        # 1. Always write turbulence properties
        self._safe_write(self.turbulenceProperties, constant_path / "turbulenceProperties")

        # 2. Write transport/physical properties based on simulation type
        if getattr(self.parent, "compressible", False):
            self._safe_write(self.physicalProperties, constant_path / "physicalProperties")
            self._safe_write(self.pRef, constant_path / "pRef")
        else:
            self._safe_write(self.transportProperties, constant_path / "transportProperties")

        # 3. Write gravity file if enabled
        if getattr(self.parent, "with_gravity", False):
            self._safe_write(self.gravity, constant_path / "g")
            self._update_pressure_field()  # Update p → p_rgh

        # 4. Write radiation files if enabled
        if self.with_radiation:
            if self._radiation is None:
                self.enable_radiation()  # Create default instances
            self._safe_write(self._radiation, constant_path / "radiationProperties")
            self._safe_write(self._fvmodels, constant_path / "fvModels")

        logger.info(f"Successfully wrote constant directory files to {constant_path}")
        return self  # Enable method chaining

    # ======================
    # Backward Compatibility
    # ======================
    __call__ = write  # Allow solver.constant() syntax
