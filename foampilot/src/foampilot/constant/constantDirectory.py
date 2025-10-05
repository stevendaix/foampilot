# constant/constantDirectory.py

from pathlib import Path
from foampilot.constant.transportPropertiesFile import TransportPropertiesFile
from foampilot.constant.turbulencePropertiesFile import TurbulencePropertiesFile
from foampilot.constant.physicalProperties import PhysicalPropertiesFile
from foampilot.constant.gravityFile import GravityFile
from foampilot.constant.pRefFile import PRefFile


class ConstantDirectory:
    """
    Manager for the 'constant' directory in an OpenFOAM case.
    Each file is a separate class handling its own content.
    This class only decides:
      - Which files are required
      - Where to write them
    """

    def __init__(self, parent):
        self.parent = parent

        # Instantiate default files
        self._transportProperties = TransportPropertiesFile()
        self._turbulenceProperties = TurbulencePropertiesFile()
        self._physicalProperties = PhysicalPropertiesFile()
        self._gravity = GravityFile()
        self._pRef = PRefFile()  # Instancié mais n'est écrit que si compressible

    # -------------------
    # Properties
    # -------------------
    @property
    def transportProperties(self):
        return self._transportProperties

    @transportProperties.setter
    def transportProperties(self, value):
        self._transportProperties = value

    @property
    def turbulenceProperties(self):
        return self._turbulenceProperties

    @turbulenceProperties.setter
    def turbulenceProperties(self, value):
        self._turbulenceProperties = value

    @property
    def physicalProperties(self):
        return self._physicalProperties

    @physicalProperties.setter
    def physicalProperties(self, value):
        self._physicalProperties = value

    @property
    def gravity(self):
        return self._gravity

    @gravity.setter
    def gravity(self, value):
        self._gravity = value  # La classe GravityFile gère elle-même le format et les unités

    @property
    def pRef(self):
        return self._pRef

    @pRef.setter
    def pRef(self, value):
        self._pRef = value  # La classe PRefFile gère elle-même float ou Quantity

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

        # Create directories if needed
        polyMesh_path.mkdir(parents=True, exist_ok=True)

        # Always write turbulence properties
        self.turbulenceProperties.write(constant_path / "turbulenceProperties")

        # Compressible vs incompressible
        if getattr(self.parent, "compressible", False):
            self.physicalProperties.write(constant_path / "physicalProperties")
            self.pRef.write(constant_path / "pRef")  # uniquement compressible
        else:
            self.transportProperties.write(constant_path / "transportProperties")

        # Gravity if enabled
        if getattr(self.parent, "with_gravity", True):
            self.gravity.write(constant_path / "g")