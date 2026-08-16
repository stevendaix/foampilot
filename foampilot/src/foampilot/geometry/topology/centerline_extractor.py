import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh

from .open_profile import BoundaryRole, OpenProfile

logger = logging.getLogger(__name__)

from .vmtk.vmtkcenterlines import vmtkCenterlines, _trimesh_to_vtk_polydata


class TopologyCenterlineExtractor:
    def __init__(self, resampling_step_mm: float = 1.0) -> None:
        self.resampling_step_mm = resampling_step_mm

    def extract_axis(self, mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        centerline = self._compute_centerline(mesh)
        return self._axis_from_centerline(centerline)

    def _compute_centerline(self, mesh: trimesh.Trimesh) -> np.ndarray:
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        mesh = mesh.process(True)

        inlet_center, outlet_center = self._detect_inlet_outlet(mesh)
        source_ids = [int(np.argmin(np.linalg.norm(mesh.vertices - inlet_center, axis=1)))]
        target_ids = [int(np.argmin(np.linalg.norm(mesh.vertices - outlet_center, axis=1)))]

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

    @staticmethod
    def _detect_inlet_outlet(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        verts = mesh.vertices
        mean = verts.mean(axis=0)
        centered = verts - mean
        cov = centered.T @ centered / max(len(verts) - 1, 1)
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
        inlet_center = verts[inlet_vidx].mean(axis=0) if len(inlet_vidx) > 0 else verts[projections.argmin()]
        outlet_center = verts[outlet_vidx].mean(axis=0) if len(outlet_vidx) > 0 else verts[projections.argmax()]
        logger.info("Inlet/outlet auto-detected: inlet=%s, outlet=%s", inlet_center, outlet_center)
        return inlet_center, outlet_center

    @staticmethod
    def _axis_from_centerline(centerline: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(centerline) < 2:
            raise ValueError("Centerline has too few points")
        origin = centerline[0]
        end = centerline[-1]
        axis = end - origin
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise ValueError("Degenerate centerline axis")
        axis = axis / norm
        return axis, origin

    def classify_profiles(
        self, profiles: List[OpenProfile], mesh: Optional[trimesh.Trimesh] = None
    ) -> List[OpenProfile]:
        if not profiles:
            return profiles
        if len(profiles) == 1:
            profiles[0].role = BoundaryRole.INLET
            profiles[0].confidence = 0.6
            return profiles

        axis, origin = self.extract_axis(mesh)

        centroids = np.array([p.centroid for p in profiles], dtype=float)
        projections = (centroids - origin) @ axis
        min_idx = int(np.argmin(projections))
        max_idx = int(np.argmax(projections))
        for idx, p in enumerate(profiles):
            if idx == min_idx:
                p.role = BoundaryRole.INLET
                p.confidence = 0.7
            elif idx == max_idx:
                p.role = BoundaryRole.OUTLET
                p.confidence = 0.7
            else:
                p.role = BoundaryRole.UNKNOWN
                p.confidence = 0.3
            p.metadata.setdefault("axis_projection", float(projections[idx]))
            p.metadata.setdefault("classification_method", "centerline")
        return profiles
