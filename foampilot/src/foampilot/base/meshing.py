from foampilot.base.openFOAMFile import OpenFOAMFile
from pathlib import Path
from foampilot.mesh.BlockMeshFile import BlockMeshFile
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
        """
        Initializes the Meshing with default files.
        """
        # Initialize blockMeshDict with default parameters
        self.blockMeshDict = BlockMeshFile(self)
        self.additional_files = {}
        self.case_path = Path(path_case)

    def add_file(self, file_name, file_content):
        """
        Adds an additional file to the meshing process.

        Args:
            file_name (str): The name of the file to add.
            file_content (dict): The content of the file.
        """
        self.additional_files[file_name] = OpenFOAMFile(object_name=file_name, **file_content)

    def load_from_json(self, json_path):
        """
        Loads the mesh configuration from a JSON file and updates the blockMeshDict.

        Args:
            json_path (str): The path to the JSON file.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Initialize BlockMeshFile with the data from the JSON
        self.blockMeshDict = BlockMeshFile(
            scale=data.get("scale", 1.0),
            vertices=data.get("vertices", []),
            blocks=data.get("blocks", []),
            edges=data.get("edges", []),
            defaultPatch = data.get("defaultPatch", []),
            boundary=data.get("boundary", {}),
            mergePatchPairs=data.get("mergePatchPairs", [])
        )
    def write(self):
        """
        Writes the files to their respective directories within the system directory.

        Args:
            base_path (str): The base path to the OpenFOAM case directory.
        """
        system_path = Path(self.case_path) / 'system'
        system_path.mkdir(parents=True, exist_ok=True)

        # Writing blockMeshDict file
        self.blockMeshDict.write(system_path / 'blockMeshDict')

        # Writing additional files
        for file_name, file_content in self.additional_files.items():
            file_content.write(system_path / file_name)



    def run_blockMesh(self):
        """
        Executes the blockMesh command in the specified case path and logs the output.

        Raises:
            FileNotFoundError: If the case path does not exist.
            RuntimeError: If the blockMesh command fails.
        """
        base_path = self.case_path
        log_file = base_path / "log.blockMesh"

        if not base_path.exists():
            raise FileNotFoundError(f"The case path '{base_path}' does not exist.")

        if not base_path.is_dir():
            raise NotADirectoryError(f"The case path '{base_path}' is not a directory.")

        try:
            # Run blockMesh
            with log_file.open("w") as f:
                f.write(f"Running 'blockMesh' in: {base_path}\n")
                result = subprocess.run(
                    ["blockMesh"],
                    cwd=base_path,
                    text=True,
                    capture_output=True,
                    check=True
                )
                f.write("blockMesh executed successfully.\n")
                f.write(result.stdout + "\n")
                f.write(result.stderr + "\n")  # capture warnings/errors if any

        except subprocess.CalledProcessError as e:
            with log_file.open("a") as f:
                f.write(f"Error executing blockMesh:\n{e.stderr}\n")
            raise RuntimeError(f"blockMesh failed with error: {e.stderr}")

        except Exception as e:
            with log_file.open("a") as f:
                f.write(f"Unexpected error: {str(e)}\n")
            raise
