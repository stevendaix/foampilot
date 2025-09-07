from pathlib import Path
from foampilot.constant.transportPropertiesFile import TransportPropertiesFile
from foampilot.constant.turbulencePropertiesFile import TurbulencePropertiesFile
from foampilot.base.openFOAMFile import OpenFOAMFile


class ConstantDirectory:
    """
    A class representing the 'constant' directory in an OpenFOAM case, which contains files
    such as 'transportProperties', 'turbulenceProperties'.
    """

    def __init__(self,parent):
        """
        Initializes the ConstantDirectory with default instances of files.
        """
        self.parent = parent 
        # Initialize default files
        self.transportProperties = TransportPropertiesFile()
        self.turbulenceProperties = TurbulencePropertiesFile()


    def write(self):
        """
        Writes the files to their respective locations within the 'constant' directory.

        Args:
            base_path (str or Path): The base path to the OpenFOAM case directory.
        """
        base_path = Path(self.parent.case_path) 
        constant_path = base_path / 'constant'
        polyMesh_path = constant_path / 'polyMesh'

        # Create directories if they don't exist
        polyMesh_path.mkdir(parents=True, exist_ok=True)

        # Write files to the appropriate paths
        self.transportProperties.write(constant_path / 'transportProperties')
        self.turbulenceProperties.write(constant_path / 'turbulenceProperties')
