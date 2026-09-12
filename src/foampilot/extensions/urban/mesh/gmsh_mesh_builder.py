from pathlib import Path
from typing import Optional

from foampilot.extensions.urban.mesh.sizing import MeshConfig


class GmshMeshBuilder:
    def __init__(self, case_path: Path, config: Optional[MeshConfig] = None):
        self.case_path = case_path
        self.config = config or MeshConfig()

    def build(self) -> None:
        pass
