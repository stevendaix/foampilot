from foampilot.base.openFOAMFile import OpenFOAMFile
import json
import os
import gzip
from pathlib import Path
import subprocess

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
                 defaultPatch=None, boundary=None, mergePatchPairs=None, definitions=None,
                 geometry=None):
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
        mergePatchPairs : list of tuple
            List of merge patch pairs (default is empty list).
        definitions : list of str
            Raw OpenFOAM dictionary definitions emitted before ``blocks``;
            useful for reusable grading variables and other declarative entries.
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
        self.definitions = definitions if definitions is not None else []
        self.geometry = geometry if geometry is not None else []

        super().__init__(object_name="blockMeshDict")


    def load_from_json(self, json_path: str):
        if not os.path.isfile(json_path):
            raise FileNotFoundError(json_path)

        with open(json_path) as f:
            data = json.load(f)

        self.scale = data.get("scale", 1.0)
        self.vertices = data.get("vertices", [])
        self.blocks = data.get("blocks", [])
        self.edges = data.get("edges", [])
        self.defaultPatch = data.get("defaultPatch", {})
        self.boundary = data.get("boundary", {})
        self.mergePatchPairs = data.get("mergePatchPairs", [])

        # DEBUG OBLIGATOIRE
        print("JSON loaded:")
        print("vertices:", len(self.vertices))
        print("blocks:", len(self.blocks))


    def import_reference_dict(self, source_path: str | Path, destination: str | Path | None = None) -> Path:
        """Import a complete OpenFOAM ``blockMeshDict`` without lossy parsing.

        This preserves advanced entries such as ``arc`` and ``project`` edges
        that are not representable by the JSON convenience format.
        """
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        target = Path(destination) if destination is not None else self.case_path / "system" / "blockMeshDict"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        return target

    def import_reference_asset(self, source_path: str | Path, destination: str | Path) -> Path:
        """Import a mesh asset, transparently decompressing a ``.gz`` source."""
        source = Path(source_path)
        target = Path(destination)
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".gz":
            target.write_bytes(gzip.decompress(source.read_bytes()))
        else:
            target.write_bytes(source.read_bytes())
        return target

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
            f.write("FoamFile\n{\n")
            for key, value in self.header.items():
                f.write(f"    {key}     {value};\n")
            f.write("}\n\n")
            f.write(f"scale {self.scale};\n\n")

            if self.definitions:
                for definition in self.definitions:
                    f.write(f"{definition}\n")
                f.write("\n")

            if self.geometry:
                f.write("geometry\n{\n")
                for item in self.geometry:
                    if isinstance(item, str):
                        f.write(f"    {item}\n")
                    else:
                        name, content = item
                        f.write(f"    {name}\n    {{\n{content}\n    }}\n")
                f.write("}\n\n")

            f.write("vertices\n(\n")
            for vertex in self.vertices:
                if isinstance(vertex, str):
                    f.write(f"    {vertex}\n")
                else:
                    f.write(f"    ({' '.join(map(str, vertex))})\n")
            f.write(");\n\n")

            f.write("blocks\n(\n")
            for block in self.blocks:
                f.write(f"    {block}\n")
            f.write(");\n\n")

            f.write("edges\n(\n")
            for edge in self.edges:
                f.write(f"    {edge}\n")
            f.write(");\n\n")

            if self.defaultPatch:
                f.write("defaultPatch\n{\n")
                for key, val in self.defaultPatch.items():
                    f.write(f"    {key if key in ('type', 'name') else 'type'} {val};\n")
                f.write("}\n\n")

            f.write("boundary\n(\n")
            for name, conditions in self.boundary.items():
                f.write(f"    {name}\n    {{\n")
                f.write(f"        type {conditions['type']};\n")
                if 'faces' in conditions:
                    f.write("        faces\n        (\n")
                    for face in conditions['faces']:
                        if isinstance(face, str):
                            f.write(f"            {face}\n")
                        else:
                            f.write(f"            ({' '.join(map(str, face))})\n")
                    f.write("        );\n")
                f.write("    }\n")
            f.write(");\n\n")

            f.write("mergePatchPairs\n(\n")
            for pair in self.mergePatchPairs:
                f.write(f"    ({pair[0]} {pair[1]})\n")
            f.write(");\n")
            
    def copy_mesh(self, source_mesh: str, destination: str = "constant") -> Path:
        """Copy a named mesh ``polyMesh`` into the case constant mesh."""
        import shutil
        source = self.case_path / "constant" / "meshes" / source_mesh / "polyMesh"
        target = self.case_path / destination / "polyMesh"
        if not source.is_dir():
            raise FileNotFoundError(source)
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        return target

    def write_mesh_times(self, times, destination: str = "constant/meshTimes") -> Path:
        """Write the ordered list of temporal mesh names."""
        target = self.case_path / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(str(t) for t in times) + "\n")
        return target

    def create_non_conformal_couples(self) -> None:
        """Create non-conformal couples required by OF13 multi-patch meshes."""
        log_path = self.case_path / "log.createNonConformalCouples"
        try:
            result = subprocess.run(
                ["createNonConformalCouples", "-case", str(self.case_path)],
                cwd=self.case_path,
                text=True,
                capture_output=True,
                check=True,
            )
            log_path.write_text(result.stdout + "\n" + result.stderr)
        except subprocess.CalledProcessError as exc:
            log_path.write_text((exc.stdout or "") + "\n" + (exc.stderr or ""))
            raise RuntimeError(f"createNonConformalCouples failed: {exc.stderr}") from exc

    def run(self):
        """
        Executes blockMesh for the case, logging output to a file.

        Raises:
            FileNotFoundError: If the case path does not exist.
            RuntimeError: If the command fails.
        """
        base_path = self.case_path
        if not base_path.exists():
            raise FileNotFoundError(f"The case path '{base_path}' does not exist.")
        if not base_path.is_dir():
            raise NotADirectoryError(f"The case path '{base_path}' is not a directory.")

        bm_log = base_path / "log.blockMesh"
        try:
            with bm_log.open("w") as f:
                f.write(f"Running 'blockMesh' in: {base_path}\n")
                result = subprocess.run(
                    ["blockMesh", "-case", str(base_path)],
                    cwd=base_path,
                    text=True,
                    capture_output=True,
                    check=True
                )
                f.write("blockMesh executed successfully.\n")
                f.write(result.stdout + "\n")
                f.write(result.stderr + "\n")
        except subprocess.CalledProcessError as e:
            with bm_log.open("a") as f:
                f.write(f"Error executing blockMesh:\n{e.stderr}\n")
            raise RuntimeError(f"blockMesh failed with error: {e.stderr}")
        except Exception as e:
            with bm_log.open("a") as f:
                f.write(f"Unexpected error: {str(e)}\n")
            raise
