import logging
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from .vmtk_local.vmtkcenterlines import vmtkCenterlines, _trimesh_to_vtk_polydata

logger = logging.getLogger(__name__)


class CenterlineExtractor:
    def __init__(self, resampling_step_mm: float = 1.0):
        self.resampling_step_mm = resampling_step_mm

    @staticmethod
    def _detect_inlet_outlet(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
        verts = mesh.vertices
        mean = verts.mean(axis=0)
        centered = verts - mean
        cov = centered.T @ centered / (len(verts) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        principal_axis = eigvecs[:, order[0]]
        projections = centered @ principal_axis
        proj_min = projections.min()
        proj_max = projections.max()
        length = proj_max - proj_min
        thresh = length * 0.01
        inlet_vidx = np.where(projections <= proj_min + thresh)[0]
        outlet_vidx = np.where(projections >= proj_max - thresh)[0]
        inlet_center = verts[inlet_vidx].mean(axis=0)
        outlet_center = verts[outlet_vidx].mean(axis=0)
        logger.info(
            f"Inlet/outlet auto-detected: inlet={inlet_center}, outlet={outlet_center}"
        )
        return inlet_center, outlet_center

    def extract(self, stl_path: Path, source_point: Optional[np.ndarray] = None, target_point: Optional[np.ndarray] = None) -> np.ndarray:
        mesh = trimesh.load(stl_path)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        mesh = mesh.process(True)

        if source_point is None or target_point is None:
            inlet_center, outlet_center = self._detect_inlet_outlet(mesh)
        if source_point is None:
            source_point = inlet_center
        if target_point is None:
            target_point = outlet_center

        source_ids = [int(np.argmin(np.linalg.norm(mesh.vertices - source_point, axis=1)))]
        target_ids = [int(np.argmin(np.linalg.norm(mesh.vertices - target_point, axis=1)))]

        centerliner = vmtkCenterlines()
        centerliner.Surface = _trimesh_to_vtk_polydata(mesh)
        centerliner.SeedSelectorName = "idlist"
        centerliner.SourceIds = source_ids
        centerliner.TargetIds = target_ids
        centerliner.ResamplingStepLength = self.resampling_step_mm
        centerliner.Execute()

        if centerliner.Centerlines is None:
            raise RuntimeError("Centerline extraction failed")

        pts = np.array([centerliner.Centerlines.GetPoint(i) for i in range(centerliner.Centerlines.GetNumberOfPoints())])
        return pts
