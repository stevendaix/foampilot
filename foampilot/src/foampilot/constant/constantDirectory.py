from pathlib import Path
from foampilot.constant.transportPropertiesFile import TransportPropertiesFile
from foampilot.constant.turbulencePropertiesFile import TurbulencePropertiesFile
from foampilot.base.openFOAMFile import OpenFOAMFile


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

    @property
    def transportProperties(self):
        """Get the transport properties object"""
        return self._transportProperties

    @transportProperties.setter
    def transportProperties(self, value):
        """Set the transport properties object"""
        self._transportProperties = value

    @property
    def turbulenceProperties(self):
        """Get the turbulence properties object"""
        return self._turbulenceProperties

    @turbulenceProperties.setter
    def turbulenceProperties(self, value):
        """Set the turbulence properties object"""
        self._turbulenceProperties = value

    def write(self):
        """
        Writes the files to their respective locations within the 'constant' directory.
        """
        base_path = Path(self.parent.case_path) 
        constant_path = base_path / 'constant'
        polyMesh_path = constant_path / 'polyMesh'

        # Create directories if they don't exist
        polyMesh_path.mkdir(parents=True, exist_ok=True)

        # Write files to the appropriate paths
        self.transportProperties.write(constant_path / 'transportProperties')
        self.turbulenceProperties.write(constant_path / 'turbulenceProperties')
