import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import vtk

from .pypes import vmtkBaseScript

logger = logging.getLogger(__name__)


class vmtkMeshWriter(vmtkBaseScript):
    def __init__(self):
        super().__init__()
        self.Mesh: Optional[vtk.vtkUnstructuredGrid] = None
        self.OutputFileName: str = ""
        self.EntityIdsArrayName: str = "CellEntityIds"
        self.WriteAllFiles: bool = True
        self.OfVersion: str = ""

    def Execute(self):
        if not self.OutputFileName:
            self.PrintError("Error: OutputFileName not set")
            return
        if self.Mesh is None:
            self.PrintError("Error: No input mesh")
            return
        path = Path(self.OutputFileName)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".vtu":
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(self.OutputFileName)
            writer.SetInputData(self.Mesh)
            writer.Write()
            self.PrintLog(f"Mesh written to {self.OutputFileName}")
        else:
            self.PrintError("Only .vtu export supported in local implementation")
