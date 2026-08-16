import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


class Section:
    def __init__(self, center: np.ndarray, direction: np.ndarray, points: np.ndarray, orientation: float = 1.0):
        self.center = np.asarray(center, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        self.points = np.asarray(points, dtype=float)
        self.orientation = float(orientation)
        if self.direction.ndim == 1:
            self.direction = self.direction / np.linalg.norm(self.direction)

    def local_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = self.direction
        arbitrary = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(z, arbitrary)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return x, y, z

    def to_2d(self) -> np.ndarray:
        x, y, _ = self.local_frame()
        pts = self.points - self.center
        return np.column_stack((pts @ x, pts @ y))

    def signed_area(self) -> float:
        pts = self.to_2d()
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _ensure_consistent_orientation(sections: List[Section]) -> List[Section]:
    if len(sections) < 2:
        return sections
    target_sign = np.sign(sections[0].signed_area())
    if target_sign == 0:
        target_sign = 1.0
    result = []
    for section in sections:
        area = section.signed_area()
        if np.sign(area) != target_sign:
            pts = section.points[::-1].copy()
            center = pts.mean(axis=0)
            result.append(Section(center=center, direction=section.direction, points=pts, orientation=-section.orientation))
        else:
            result.append(section)
    return result


class SectionExtractor:
    def __init__(self, spacing_mm: float = 2.0):
        self.spacing_mm = spacing_mm

    def extract(self, mesh: trimesh.Trimesh, centerline: np.ndarray, enforce_orientation: bool = True) -> List[Section]:
        sections: List[Section] = []
        if len(centerline) < 2:
            return sections
        for i in range(len(centerline) - 1):
            center = centerline[i]
            direction = centerline[i + 1] - centerline[i]
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            direction = direction / norm
            try:
                section = mesh.section(plane_origin=center, plane_normal=direction)
            except Exception:
                continue
            if section is None or len(section.discrete) == 0:
                continue
            points = np.asarray(section.discrete[0])
            if points.ndim != 2 or points.shape[1] != 3:
                continue
            center = points.mean(axis=0)
            sections.append(Section(center=center, direction=direction, points=points))
        if not sections:
            raise RuntimeError("No section could be extracted along the centerline")
        if enforce_orientation:
            sections = _ensure_consistent_orientation(sections)
        logger.info("Sections extracted: %d", len(sections))
        return sections
