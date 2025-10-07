from pathlib import Path
from foampilot.constant.transportPropertiesFile import TransportPropertiesFile
from foampilot.constant.turbulencePropertiesFile import TurbulencePropertiesFile
from foampilot.constant.physicalProperties import PhysicalPropertiesFile
from foampilot.constant.gravityFile import GravityFile
from foampilot.constant.pRefFile import PRefFile
from foampilot.constant.radiationPropertiesFile import RadiationPropertiesFile, FvModelsFile


class ConstantDirectory:
    """
    Manager for the 'constant' directory in an OpenFOAM case.
    Each file is a separate class handling its own content.
    """

    def __init__(self, parent):
        self.parent = parent

        # Default components
        self._transportProperties = TransportPropertiesFile()
        self._turbulenceProperties = TurbulencePropertiesFile()
        self._physicalProperties = PhysicalPropertiesFile()
        self._gravity = GravityFile()
        self._pRef = PRefFile()

        # Radiation (optional)
        self.with_radiation = False
        self._radiation = None
        self._fvmodels = None

    # -------------------
    # Properties
    # -------------------
    @property
    def radiation(self):
        return self._radiation

    @radiation.setter
    def radiation(self, value):
        self._radiation = value

    # -------------------
    # Enable radiation
    # -------------------
    def enable_radiation(self, model: str = "P1", absorptivity=0.5, emissivity=0.5):
        """
        Enable radiation model in the constant directory.

        Args:
            model (str): 'P1' or 'fvDOM'
            absorptivity, emissivity (float): parameters for radiation model
        """
        self.with_radiation = True
        self._radiation = RadiationPropertiesFile(
            model=model,
            absorptivity=absorptivity,
            emissivity=emissivity,
        )
        self._fvmodels = FvModelsFile()

    # -------------------
    # Write method
    # -------------------
    def write(self):
        """
        Writes required files in the 'constant' directory.
        Only writes files that are relevant for the current case.
        """
        base_path = Path(self.parent.case_path)
        constant_path = base_path / "constant"
        polyMesh_path = constant_path / "polyMesh"
        polyMesh_path.mkdir(parents=True, exist_ok=True)

        # Always write turbulence properties
        self.turbulenceProperties.write(constant_path / "turbulenceProperties")

        # Compressible vs incompressible
        if getattr(self.parent, "compressible", False):
            self.physicalProperties.write(constant_path / "physicalProperties")
            self.pRef.write(constant_path / "pRef")
        else:
            self.transportProperties.write(constant_path / "transportProperties")

        # Gravity if enabled
        if getattr(self.parent, "with_gravity", True):
            self.gravity.write(constant_path / "g")

        # Radiation if enabled
        if self.with_radiation and self._radiation is not None:
            self._radiation.write(constant_path)
            if self._fvmodels is not None:
                self._fvmodels.write(constant_path)