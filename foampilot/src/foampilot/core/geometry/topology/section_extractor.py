import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh

from .open_profile import OpenProfile

logger = logging.getLogger(__name__)


class Section:
    def __init__(self, center: np.ndarray, direction: np.ndarray, points: np.ndarray, orientation: float = 1.0) -> None:
        self.center = np.asarray(center, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        self.points = np.asarray(points, dtype=float)
        self.orientation = float(orientation)
        if self.direction.ndim == 1:
            self.direction = self.direction / (np.linalg.norm(self.direction) + 1e-12)

    def local_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = self.direction
        arbitrary = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(z, arbitrary)
        x /= np.linalg.norm(x) + 1e-12
        y = np.cross(z, x)
        return x, y, z

    def to_2d(self) -> np.ndarray:
        x, y, _ = self.local_frame()
        pts = self.points - self.center
        return np.column_stack((pts @ x, pts @ y))

    def signed_area(self) -> float:
        pts = self.to_2d()
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    @property
    def radius(self) -> float:
        area = abs(self.signed_area())
        return float(np.sqrt(area / np.pi)) if area > 0.0 else 0.0

    @property
    def area(self) -> float:
        return float(abs(self.signed_area()))


def _compute_polygon_area(points_2d: np.ndarray) -> float:
    x, y = points_2d[:, 0], points_2d[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _build_local_frame(plane_normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(plane_normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(plane_normal, arbitrary)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(plane_normal, u)
    return u, v


def _select_best_polyline(
    polylines: List[np.ndarray],
    plane_normal: np.ndarray,
    reference_point: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if not polylines:
        return None
    if len(polylines) == 1:
        return polylines[0]

    u, v = _build_local_frame(plane_normal)

    best_polyline = None
    best_score = -np.inf

    for poly in polylines:
        if len(poly) < 3:
            continue
        is_closed = np.allclose(poly[0], poly[-1], atol=1e-8)
        centroid = poly.mean(axis=0)
        pts_2d = np.column_stack(((poly - centroid) @ u, (poly - centroid) @ v))
        area = abs(_compute_polygon_area(pts_2d))
        score = area * (2.0 if is_closed else 1.0)
        if reference_point is not None:
            dist = np.linalg.norm(centroid - reference_point)
            score /= (1.0 + dist)
        if score > best_score:
            best_score = score
            best_polyline = poly

    return best_polyline


def _resample_contour(points: np.ndarray, n_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points
    if not np.allclose(points[0], points[-1], atol=1e-8):
        points = np.vstack([points, points[0:1]])
    edges = np.diff(points, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    cumulative = np.cumsum(lengths)
    total = cumulative[-1]
    if total < 1e-12:
        return np.tile(points[0:1], (n_points, 1))
    target_d = np.linspace(0, total, n_points, endpoint=False)
    resampled = []
    for d in target_d:
        idx = np.searchsorted(cumulative, d)
        if idx == 0:
            resampled.append(points[0].copy())
        else:
            seg_start = cumulative[idx - 1]
            seg_end = cumulative[idx]
            t = (d - seg_start) / (seg_end - seg_start + 1e-12)
            pt = points[idx - 1] + t * edges[idx - 1]
            resampled.append(pt)
    return np.array(resampled)


def _process_section_polylines(
    section_result,
    plane_normal: np.ndarray,
    reference_point: np.ndarray,
    n_resample: int = 64,
    min_area: float = 1e-10,
) -> Optional[np.ndarray]:
    if section_result is None or len(section_result.discrete) == 0:
        return None
    polylines = [np.asarray(p, dtype=float) for p in section_result.discrete]
    polyline = _select_best_polyline(polylines, plane_normal, reference_point)
    if polyline is None or len(polyline) < 3:
        return None
    diffs = np.diff(polyline, axis=0)
    keep = np.concatenate([[True], np.linalg.norm(diffs, axis=1) > 1e-12])
    polyline = polyline[keep]
    if len(polyline) < 3:
        return None
    if not np.allclose(polyline[0], polyline[-1], atol=1e-8):
        polyline = np.vstack([polyline, polyline[0:1]])
    polyline = _resample_contour(polyline, n_resample)
    centroid = polyline.mean(axis=0)
    u, v = _build_local_frame(plane_normal)
    pts_2d = np.column_stack(((polyline - centroid) @ u, (polyline - centroid) @ v))
    area = abs(_compute_polygon_area(pts_2d))
    if area < min_area:
        return None
    return polyline


class TopologySectionExtractor:
    def __init__(self, spacing_mm: float = 2.0) -> None:
        self.spacing_mm = spacing_mm

    def extract_at_profiles(
        self, mesh: trimesh.Trimesh, profiles: List[OpenProfile]
    ) -> List[OpenProfile]:
        if not profiles:
            return profiles
        for profile in profiles:
            center = profile.centroid
            normal = profile.normal
            try:
                section = mesh.section(plane_origin=center, plane_normal=normal)
            except Exception as exc:
                logger.warning("Section extraction failed for profile %d: %s", profile.id, exc)
                continue
            points = _process_section_polylines(section, normal, center, n_resample=64)
            if points is None:
                logger.warning("No section found for profile %d", profile.id)
                continue
            sec = Section(center=points.mean(axis=0), direction=normal, points=points)
            profile.area = float(abs(sec.signed_area()))
            profile.perimeter = float(np.sum(np.linalg.norm(np.diff(np.vstack([points, points[0]]), axis=0), axis=1)))
            profile.equivalent_radius = float(np.sqrt(profile.area / np.pi)) if profile.area > 0.0 else 0.0
            profile.metadata.setdefault("section_points", points.shape[0])
            profile.metadata.setdefault("classification_method", profile.metadata.get("classification_method", "geometric") + "_section")
        return profiles

    def extract_along_axis(
        self, mesh: trimesh.Trimesh, axis: np.ndarray, origin: np.ndarray, n_steps: int = 10
    ) -> List[Section]:
        axis = np.asarray(axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise ValueError("Degenerate axis")
        axis = axis / norm
        sections: List[Section] = []
        for i in range(n_steps):
            t = i / max(n_steps - 1, 1)
            center = origin + t * axis * self.spacing_mm * (n_steps - 1)
            try:
                section = mesh.section(plane_origin=center, plane_normal=axis)
            except Exception:
                continue
            points = _process_section_polylines(section, axis, center, n_resample=64)
            if points is None:
                continue
            sections.append(Section(center=points.mean(axis=0), direction=axis, points=points))
        logger.info("Extracted %d sections along axis", len(sections))
        return sections

    def extract_sections_from_centerline(
        self,
        mesh: trimesh.Trimesh,
        centerline: np.ndarray,
        n_sections: int = 90,
        n_resample: int = 64,
        min_area: float = 1e-10,
    ) -> List[Section]:
        centerline = np.asarray(centerline, dtype=float)
        if centerline.ndim != 2 or centerline.shape[1] != 3:
            raise ValueError("Centerline must be (N, 3)")
        if len(centerline) < 2:
            raise ValueError("Centerline needs at least 2 points")

        diffs = np.diff(centerline, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_length = cumulative[-1]

        if total_length < 1e-12:
            raise ValueError("Degenerate centerline")

        target_d = np.linspace(0, total_length, n_sections, endpoint=False)

        sample_points = []
        for d in target_d:
            idx = np.searchsorted(cumulative, d)
            idx = min(idx, len(centerline) - 1)
            if idx == 0:
                sample_points.append(centerline[0])
            else:
                seg_start = cumulative[idx - 1]
                seg_end = cumulative[idx]
                t = (d - seg_start) / (seg_end - seg_start + 1e-12)
                pt = centerline[idx - 1] + t * diffs[idx - 1]
                sample_points.append(pt)

        sample_points = np.array(sample_points)

        raw_tangents = np.zeros_like(sample_points)
        raw_tangents[0] = sample_points[1] - sample_points[0]
        raw_tangents[-1] = sample_points[-1] - sample_points[-2]
        for i in range(1, len(sample_points) - 1):
            raw_tangents[i] = sample_points[i + 1] - sample_points[i - 1]

        norms = np.linalg.norm(raw_tangents, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        tangents = raw_tangents / norms

        smoothed = np.zeros_like(tangents)
        for i in range(len(tangents)):
            start = max(0, i - 1)
            end = min(len(tangents), i + 2)
            smoothed[i] = tangents[start:end].mean(axis=0)
        norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        tangents = smoothed / norms

        sections: List[Section] = []
        for i in range(n_sections):
            center = sample_points[i]
            direction = tangents[i]
            try:
                section = mesh.section(plane_origin=center, plane_normal=direction)
            except Exception as exc:
                logger.warning("Section extraction failed at point %d: %s", i, exc)
                continue
            points = _process_section_polylines(section, direction, center, n_resample, min_area)
            if points is None:
                logger.debug("No valid section at point %d", i)
                continue
            sections.append(Section(center=center, direction=direction, points=points))

        logger.info("Extracted %d sections from centerline", len(sections))
        return sections
