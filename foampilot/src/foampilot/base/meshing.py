from foampilot.base.openFOAMFile import OpenFOAMFile
from pathlib import Path
from foampilot.mesh.BlockMeshFile import BlockMesher
from foampilot.mesh.gmsh_mesher import GmshMesher
from foampilot.mesh.snappymesh import SnappyMesher

import json
import subprocess
from typing import Union, Dict, Any


class Meshing:
    """A high-level manager for the meshing process in an OpenFOAM case.

    This class acts as a factory and orchestrator for different meshing strategies 
    (blockMesh, Gmsh, or snappyHexMesh). It handles the initialization of specific 
    meshing backends and manages supplementary configuration files required in 
    the OpenFOAM `system/` directory.

    Attributes:
        case_path (Path): The root directory of the OpenFOAM case.
        mesher_name (str): The name of the selected meshing strategy.
        mesher (Union[BlockMesher, GmshMesher, SnappyMesher]): The specific mesher 
            instance handling the mesh generation logic.
        additional_files (Dict[str, OpenFOAMFile]): A registry of additional 
            OpenFOAM configuration files to be written to the `system/` folder.
    """

    def __init__(self, case_path: Union[str, Path], mesher: str = "blockMesh"):
        """Initializes the Meshing manager with a specific backend.

        Args:
            case_path: The filesystem path to the OpenFOAM case directory.
            mesher: The meshing tool to use. Options are "blockMesh", 
                "gmsh", or "snappy". Defaults to "blockMesh".

        Raises:
            ValueError: If the provided `mesher` string does not match a 
                supported meshing backend.
        """
        self.case_path = Path(case_path)
        self.mesher_name = mesher

        if mesher == "blockMesh":
            self.mesher = BlockMesher(self)
        elif mesher == "gmsh":
            self.mesher = GmshMesher(self)
        elif mesher == "snappy":
            self.mesher = SnappyMesher(self, stl_file='placeholder.stl')
        else:
            raise ValueError(f"Unknown mesher: {mesher}")

        self.additional_files = {}

    def add_file(self, file_name: str, file_content: Dict[str, Any]):
        """Adds a supplementary OpenFOAM configuration file to the system directory.

        This is useful for adding files like `surfaceFeatureExtractDict` or 
        `meshQualityDict` that are required by specific meshing workflows.

        Args:
            file_name: The name of the file (e.g., 'surfaceFeatureExtractDict').
            file_content: A dictionary containing the parameters and settings 
                for the OpenFOAM file.
        """
        self.additional_files[file_name] = OpenFOAMFile(
            object_name=file_name,
            **file_content
        )

    def write(self):
        """Writes all meshing-related configuration files to the disk.

        This method triggers the `write` methods of the primary mesher 
        (e.g., generating `blockMeshDict`) and then iterates through any 
        `additional_files` to write them into the `system/` directory of the case.

        Note:
            Ensure that `self.case_path` is writable before calling this method.
        """
        # Write primary mesher files (blockMesh, snappy, gmsh)
        # Note: In the original snippet, self.blockmesh and self.snappy 
        # were called directly. Assuming these are handled by self.mesher:
        if hasattr(self.mesher, 'write'):
            self.mesher.write()
        
        # Write extra files to the system directory
        system_path = self.case_path / "system"
        system_path.mkdir(exist_ok=True)
        
        for fname, fcontent in self.additional_files.items():
            fcontent.write(system_path / fname)
