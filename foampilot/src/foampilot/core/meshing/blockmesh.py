from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import os
import gzip
import subprocess

from foampilot.core.meshing.base import MeshGenerator


class BlockMesher(MeshGenerator):
    """BlockMesh dictionary generator for OpenFOAM cases."""

    def __init__(self, case_path: Path, scale: float = 1, vertices=None, blocks=None,
                 edges=None, defaultPatch=None, boundary=None, mergePatchPairs=None, **options):
        if hasattr(case_path, "case_path"):
            case_path = Path(case_path.case_path)
        super().__init__(case_path, **options)
        self.scale = scale
        self.vertices = vertices if vertices is not None else []
        self.blocks = blocks if blocks is not None else []
        self.edges = edges if edges is not None else []
        self.defaultPatch = defaultPatch if defaultPatch is not None else {}
        self.boundary = boundary if boundary is not None else {}
        self.mergePatchPairs = mergePatchPairs if mergePatchPairs is not None else []

    def generate(self) -> None:
        self.scale = self.options.get("scale", self.scale)

    def write(self, destination: Optional[Path] = None) -> Path:
        file_path = Path(destination) if destination else self.case_path / "system" / "blockMeshDict"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            f.write("FoamFile\n{\n")
            f.write("    version     2.0;\n")
            f.write("    format     ascii;\n")
            f.write("    class      dictionary;\n")
            f.write("    object     blockMeshDict;\n")
            f.write("}\n\n")

            f.write(f"scale {self.scale};\n\n")

            f.write("vertices\n(\n")
            for vertex in self.vertices:
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
                    if key in ("type", "name"):
                        f.write(f"    {key} {val};\n")
                    else:
                        f.write(f"    type {val};\n")
                f.write("}\n\n")

            f.write("boundary\n(\n")
            for name, conditions in self.boundary.items():
                f.write(f"    {name}\n    {{\n")
                f.write(f"        type {conditions['type']};\n")
                if "faces" in conditions:
                    f.write("        faces\n        (\n")
                    for face in conditions["faces"]:
                        f.write(f"            ({' '.join(map(str, face))})\n")
                    f.write("        );\n")
                f.write("    }\n")
            f.write(")\n\n")

            f.write("mergePatchPairs\n(\n")
            for pair in self.mergePatchPairs:
                f.write(f"    ({pair[0]} {pair[1]});\n")
            f.write(");\n")

        return file_path

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

        print("JSON loaded:")
        print("vertices:", len(self.vertices))
        print("blocks:", len(self.blocks))

    def import_reference_dict(self, source_path: str | Path, destination: str | Path | None = None) -> Path:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        target = Path(destination) if destination is not None else self.case_path / "system" / "blockMeshDict"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        return target

    def import_reference_asset(self, source_path: str | Path, destination: str | Path) -> Path:
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

    def copy_mesh(self, source_mesh: str, destination: str = "constant") -> Path:
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
        target = self.case_path / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(str(t) for t in times) + "\n")
        return target

    def create_non_conformal_couples(self) -> None:
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

    def run(self) -> None:
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
                    check=True,
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
