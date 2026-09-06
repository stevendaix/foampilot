from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SectionFilterConfig:
    max_radius_ratio: float = 1.8
    max_area_ratio: float = 2.5
    min_shape: float = 0.35
    junction_angle_deg: float = 25.0
    junction_radius_factor: float = 2.5
    center_tolerance_factor: float = 0.5
    ambiguity_margin: float = 0.15


@dataclass
class ContourCandidate:
    points: np.ndarray
    closed: bool
    area: float
    perimeter: float
    radius_min: float
    radius_median: float
    radius_max: float
    shape: float
    centroid: np.ndarray
    score: float = float("inf")
    status: str = "UNCLASSIFIED"
    reason: str = ""


def plane_basis(tangent: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(tangent, dtype=float)
    t /= max(float(np.linalg.norm(t)), 1.0e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, t) * t
    u /= max(float(np.linalg.norm(u)), 1.0e-12)
    v = np.cross(t, u)
    v /= max(float(np.linalg.norm(v)), 1.0e-12)
    return u, v, t


def contour_metrics(points: np.ndarray, center: Sequence[float], tangent: Sequence[float]) -> ContourCandidate:
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3:
        raise ValueError("A contour requires at least three 3D points")
    c = np.asarray(center, dtype=float)
    u, v, t = plane_basis(tangent)
    q = p - c
    q = q - (q @ t)[:, None] * t[None, :]
    xy = np.column_stack((q @ u, q @ v))
    closed = bool(np.linalg.norm(p[0] - p[-1]) <= 1.0e-5)
    ring = xy[:-1] if closed and len(xy) > 3 else xy
    x, y = ring[:, 0], ring[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) if closed else 0.0
    perimeter = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    radial = np.linalg.norm(xy, axis=1)
    return ContourCandidate(
        points=p,
        closed=closed,
        area=area,
        perimeter=perimeter,
        radius_min=float(np.quantile(radial, 0.05)),
        radius_median=float(np.median(radial)),
        radius_max=float(np.quantile(radial, 0.95)),
        shape=float(np.quantile(radial, 0.05) / max(np.quantile(radial, 0.95), 1.0e-12)),
        centroid=np.mean(p, axis=0),
    )


def select_branch_contour(
    candidates: Iterable[ContourCandidate],
    center: Sequence[float],
    tangent: Sequence[float],
    expected_radius: float,
    previous_centroid: Sequence[float] | None = None,
    config: SectionFilterConfig = SectionFilterConfig(),
) -> tuple[ContourCandidate | None, str]:
    c = np.asarray(center, dtype=float)
    t = np.asarray(tangent, dtype=float)
    t /= max(float(np.linalg.norm(t)), 1.0e-12)
    valid = []
    for candidate in candidates:
        if not candidate.closed:
            candidate.status, candidate.reason = "REJECTED", "OPEN_CONTOUR"
            continue
        center_distance = float(np.linalg.norm(candidate.centroid - c))
        radius_error = abs(candidate.radius_median - expected_radius) / max(abs(expected_radius), 1.0e-12)
        continuity = 0.0 if previous_centroid is None else float(np.linalg.norm(candidate.centroid - np.asarray(previous_centroid)))
        candidate.score = 3.0 * center_distance + 8.0 * radius_error + 0.5 * continuity
        valid.append(candidate)
    if not valid:
        return None, "NO_CLOSED_CONTOUR"
    valid.sort(key=lambda item: item.score)
    best = valid[0]
    if len(valid) > 1 and valid[1].score <= best.score * (1.0 + config.ambiguity_margin) + 1.0e-9:
        for candidate in valid:
            candidate.status, candidate.reason = "JUNCTION", "AMBIGUOUS_MULTIPLE_CLOSED_CONTOURS"
        return None, "AMBIGUOUS_MULTIPLE_CLOSED_CONTOURS"
    if best.shape < config.min_shape:
        best.status, best.reason = "REJECTED", "BAD_SHAPE"
        return None, "BAD_SHAPE"
    best.status, best.reason = "VALID", "SELECTED"
    return best, "SELECTED"


def continuity_rejection(
    current: ContourCandidate,
    previous: ContourCandidate | None,
    next_: ContourCandidate | None,
    config: SectionFilterConfig = SectionFilterConfig(),
) -> tuple[bool, str]:
    if not current.closed:
        return True, "OPEN_CONTOUR"
    if current.shape < config.min_shape:
        return True, "BAD_SHAPE"
    neighbors = [item for item in (previous, next_) if item is not None and item.closed]
    if not neighbors:
        return False, "NO_NEIGHBOR_REFERENCE"
    radius_ref = float(np.median([item.radius_median for item in neighbors]))
    area_values = [item.area for item in neighbors if item.area > 0.0]
    area_ref = float(np.median(area_values)) if area_values else 0.0
    if current.radius_median / max(radius_ref, 1.0e-12) > config.max_radius_ratio:
        return True, "RADIUS_SPIKE"
    if area_ref > 0.0 and current.area / area_ref > config.max_area_ratio:
        return True, "AREA_SPIKE"
    return False, "OK"


def classify_station(
    candidates: Sequence[ContourCandidate],
    center: Sequence[float],
    tangent: Sequence[float],
    expected_radius: float,
    previous: ContourCandidate | None = None,
    next_: ContourCandidate | None = None,
    config: SectionFilterConfig = SectionFilterConfig(),
) -> tuple[ContourCandidate | None, str]:
    selected, selection_reason = select_branch_contour(
        candidates, center, tangent, expected_radius,
        previous.centroid if previous is not None else None, config,
    )
    if selected is None:
        return None, selection_reason
    rejected, reason = continuity_rejection(selected, previous, next_, config)
    if rejected:
        selected.status, selected.reason = "REJECTED", reason
        return None, reason
    return selected, "VALID"


__all__ = [
    "ContourCandidate",
    "SectionFilterConfig",
    "classify_station",
    "continuity_rejection",
    "contour_metrics",
    "plane_basis",
    "select_branch_contour",
]
