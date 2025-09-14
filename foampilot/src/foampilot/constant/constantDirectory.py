# constant/constantDirectory.py

from pathlib import Path
from foampilot.constant.transportPropertiesFile import TransportPropertiesFile
from foampilot.constant.turbulencePropertiesFile import TurbulencePropertiesFile
from foampilot.constant.physicalPropertiesFile import PhysicalPropertiesFile
from foampilot.constant.gravityFile import GravityFile


class ConstantDirectory:
    """
    A class representing the 'constant' directory in an OpenFOAM case.
    """

    def __init__(self, parent):
        """
        Initializes the ConstantDirectory with default instances of files.
        """
        self.parent = parent 

        # Initialize default files
        self._transportProperties = TransportPropertiesFile()
        self._turbulenceProperties = TurbulencePropertiesFile()
        self._physicalProperties = PhysicalPropertiesFile()
        self._gravity = GravityFile()

    # -------------------
    # Transport Properties
    # -------------------
    @property
    def transportProperties(self):
        return self._transportProperties

    @transportProperties.setter
    def transportProperties(self, value):
        self._transportProperties = value

    # -------------------
    # Turbulence Properties
    # -------------------
    @property
    def turbulenceProperties(self):
        return self._turbulenceProperties

    @turbulenceProperties.setter
    def turbulenceProperties(self, value):
        self._turbulenceProperties = value

    # -------------------
    # Physical Properties
    # -------------------
    @property
    def physicalProperties(self):
        return self._physicalProperties

    @physicalProperties.setter
    def physicalProperties(self, value):
        self._physicalProperties = value

    # -------------------
    # Gravity
    # -------------------
    @property
    def gravity(self):
        return self._gravity

    @gravity.setter
    def gravity(self, value):
        self._gravity = value

    # -------------------
    # Write method
    # -------------------
    def write(self):
        """
        Writes the files to their respective locations within the 'constant' directory.
        Uses parent configuration to decide which files are required.
        """
        base_path = Path(self.parent.case_path) 
        constant_path = base_path / "constant"
        polyMesh_path = constant_path / "polyMesh"

        # Create directories if they don't exist
        polyMesh_path.mkdir(parents=True, exist_ok=True)

        # Always write turbulence properties
        self.turbulenceProperties.write(constant_path / "turbulenceProperties")

        # Decide between incompressible and compressible setups
        if getattr(self.parent, "compressible", False):
            # Compressible → physicalProperties
            self.physicalProperties.write(constant_path / "physicalProperties")
        else:
            # Incompressible → transportProperties
            self.transportProperties.write(constant_path / "transportProperties")

        # If gravity is enabled
        if getattr(self.parent, "with_gravity", True):
            self.gravity.write(constant_path / "g")