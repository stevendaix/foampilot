from foampilot.base.openFOAMFile import OpenFOAMFile
from pathlib import Path
from foampilot.mesh.BlockMeshFile import BlockMesh
from foampilot.mesh.gmsher import GmshMesher
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
    def __init__(self, path_case):
        self.case_path = Path(path_case)

        self.blockmesh = BlockMesher(self)
        self.gmsher = GmshMesher(self)
        self.snappy = SnappyMesher(self)

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
    