from foampilot.base.openFOAMFile import OpenFOAMFile
from pathlib import Path
from foampilot.mesh.BlockMeshFile import BlockMesher
from foampilot.mesh.gmsh_mesher import GmshMesher
from foampilot.mesh.snappymesh import SnappyMesher

import json

import subprocess



class Meshing:
    """
    A class representing the meshing process in an OpenFOAM case.

    Attributes:
        blockMeshDict (BlockMeshDictFile): The blockMeshDict file.
        additional_files (dict): Additional files to include in the system directory.
    """
    def __init__(self, case_path, mesher="blockMesh"):
        self.case_path = Path(case_path)
        self.mesher_name = mesher

        if mesher == "blockMesh":
            self.mesher = BlockMesher(self)
        elif mesher == "gmsh":
            self.mesher = GmshMesher(self)
        elif mesher == "snappy":
            self.mesher = SnappyMesher(self)
        else:
            raise ValueError(f"Unknown mesher: {mesher}")
        
        self.additional_files = {}

    def add_file(self, file_name, file_content):
        self.additional_files[file_name] = OpenFOAMFile(
            object_name=file_name,
            **file_content
        )

    def write(self):
        # Write blockMesh, snappy, gmsh files
        self.blockmesh.write()
        self.snappy.write_dict()

        # Write extra files
        system_path = self.case_path / "system"
        system_path.mkdir(exist_ok=True)
        for fname, fcontent in self.additional_files.items():
            fcontent.write(system_path / fname)
    