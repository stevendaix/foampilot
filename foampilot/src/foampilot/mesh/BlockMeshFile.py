from foampilot.base.openFOAMFile import OpenFOAMFile
import json
import os
from pathlib import Path

class BlockMesher(OpenFOAMFile):
    """
    Represents the `blockMeshDict` file in OpenFOAM.

    This class allows you to build, modify, and export
    the `system/blockMeshDict` file, which defines the mesh
    topology for OpenFOAM simulations.

    Attributes
    ----------
    scale : float
        Scaling factor applied to the mesh.
    vertices : list of list
        List of vertex coordinates (x, y, z).
    blocks : list
        List of block definitions.
    edges : list
        List of edges definitions.
    defaultPatch : dict
        Default patch definition.
    boundary : dict
        Dictionary of boundary patches and their conditions.
    mergePatchPairs : list of tuple
        List of merge patch pairs.
    """

    def __init__(self,parent, scale: float = 1, vertices=None, blocks=None, edges=None,
                 defaultPatch=None, boundary=None, mergePatchPairs=None):
        """
        Initialize the blockMeshDict file handler.

        Parameters
        ----------
        scale : float, optional
            Mesh scaling factor (default is 1).
        vertices : list of list, optional
            List of vertices, each as `[x, y, z]` (default is empty list).
        blocks : list, optional
            List of block definitions (default is empty list).
        edges : list, optional
            List of edges (default is empty list).
        defaultPatch : dict, optional
            Default patch definition (default is empty dict).
        boundary : dict, optional
            Boundary definitions, e.g. `{"inlet": {"type": "patch", "faces": [...]}}`.
        mergePatchPairs : list of tuple, optional
            List of merge patch pairs (default is empty list).
        """
        self.parent = parent                       
        self.case_path = parent.case_path 
        self.scale = scale
        self.vertices = vertices if vertices is not None else []
        self.blocks = blocks if blocks is not None else []
        self.edges = edges if edges is not None else []
        self.defaultPatch = defaultPatch if defaultPatch is not None else {}
        self.boundary = boundary if boundary is not None else {}
        self.mergePatchPairs = mergePatchPairs if mergePatchPairs is not None else []

        super().__init__(object_name="blockMeshDict", scale=self.scale,
                         vertices=self.vertices, blocks=self.blocks,
                         edges=self.edges, defaultPatch=self.defaultPatch,
                         boundary=self.boundary, mergePatchPairs=self.mergePatchPairs)

    def load_from_json(self, json_path: str):
        """
        Load a blockMeshDict configuration from a JSON file.

        Parameters
        ----------
        json_path : str
            Path to the JSON configuration file.

        Raises
        ------
        FileNotFoundError
            If the JSON file does not exist.
        KeyError
            If the JSON file structure is invalid or missing required keys.
        """
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"The file {json_path} does not exist.")

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Update attributes
        self.scale = data.get("scale", 1.0)
        self.vertices = data.get("vertices", [])
        self.blocks = data.get("blocks", [])
        self.edges = data.get("edges", [])
        self.defaultPatch = data.get("defaultPatch", {})
        self.boundary = data.get("boundary", {})
        self.mergePatchPairs = data.get("mergePatchPairs", [])

    def write(self, file_path: Path):
        """
        Write the blockMeshDict content to a file.

        Parameters
        ----------
        file_path : Path
            Destination path of the `blockMeshDict` file.

        Notes
        -----
        This method generates the OpenFOAM dictionary syntax
        directly from the instance attributes (`scale`, `vertices`,
        `blocks`, `boundary`, etc.).
        """
        with open(file_path, 'w') as f:
            # Header
            f.write("FoamFile\n{\n")
            for key, value in self.header.items():
                f.write(f"    {key}     {value};\n")
            f.write("}\n\n")

            f.write(f"scale {self.scale};\n\n")

            # Vertices
            f.write("vertices\n(\n")
            for vertex in self.vertices:
                f.write(f"    ({' '.join(map(str, vertex))})\n")
            f.write(");\n\n")

            # Blocks
            f.write("blocks\n(\n")
            for block in self.blocks:
                f.write(f"    {block}\n")
            f.write(");\n\n")

            # Edges
            f.write("edges\n(\n")
            for edge in self.edges:
                f.write(f"    {edge}\n")
            f.write(");\n\n")

            # Default patch
            f.write("defaultPatch\n{\n")
            for name, conditions in self.defaultPatch.items():
                f.write(f"    {name}    {conditions};\n")
            f.write("}\n\n")

            # Boundary
            f.write("boundary\n(\n")
            for name, conditions in self.boundary.items():
                f.write(f"    {name}\n    {{\n")
                f.write(f"        type {conditions['type']};\n")
                if 'faces' in conditions:
                    f.write("        faces\n        (\n")
                    for face in conditions['faces']:
                        f.write(f"            ({' '.join(map(str, face))})\n")
                    f.write("        );\n")
                f.write("    }\n")
            f.write(")\n\n")

            # Merge patch pairs
            f.write("mergePatchPairs\n(\n")
            for pair in self.mergePatchPairs:
                f.write(f"    ({pair[0]} {pair[1]});\n")
            f.write(");\n")

    def run(self):
        """
        Executes the blockMesh command in the specified case path and logs the output.

        Raises:
            FileNotFoundError: If the case path does not exist.
            RuntimeError: If the blockMesh command fails.
        """
        base_path = self.parent.case_path
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