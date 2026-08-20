from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Protocol

import numpy as np

from .models import BoundaryCondition, ReconstructionSpec

logger = logging.getLogger(__name__)


class ReconstructionBackend(Protocol):
    def build(self, sections: Any, spec: ReconstructionSpec) -> Any:
        ...


@dataclass
class SectionLoftInput:
    """Normalized section representation consumed by a reconstruction backend."""

    center: np.ndarray
    points: np.ndarray
    tangent: np.ndarray
    radius: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_sections(sections: Iterable[Any]) -> list[SectionLoftInput]:
    normalized = []
    for section in sections:
        if isinstance(section, dict):
            get = section.get
            center_value = get("center")
            points_value = get("phase_locked_points", get("points"))
            tangent_value = get("direction", get("tangent"))
            radius_value = get("radius", get("equivalent_radius", 0.0))
            metadata_value = dict(get("metadata", {}))
            if "branch_id" in section: metadata_value.setdefault("branch_id", section["branch_id"])
        else:
            center_value = getattr(section, "center")
            points_value = getattr(section, "phase_locked_points", getattr(section, "points"))
            tangent_value = getattr(section, "direction", getattr(section, "tangent", np.zeros(3)))
            radius_value = getattr(section, "radius", getattr(section, "equivalent_radius", 0.0))
            metadata_value = dict(getattr(section, "metadata", {}))
        center = np.asarray(center_value, dtype=float)
        points = np.asarray(points_value, dtype=float)
        tangent = np.asarray(tangent_value, dtype=float)
        radius = float(radius_value)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
            raise ValueError("Each section must contain at least three 3D points")
        # OCC rejects zero-length edges. Section extraction can legitimately
        # return repeated vertices at an intersection or at the closed seam.
        cleaned = [points[0]]
        for point in points[1:]:
            if float(np.linalg.norm(point - cleaned[-1])) > 1.0e-8:
                cleaned.append(point)
        if len(cleaned) > 1 and float(np.linalg.norm(cleaned[0] - cleaned[-1])) <= 1.0e-8:
            cleaned.pop()
        points = np.asarray(cleaned, dtype=float)
        if len(points) < 3:
            raise ValueError("Section degenerates after removal of duplicate points")
        normalized.append(SectionLoftInput(center=center, points=points, tangent=tangent, radius=radius, metadata=metadata_value))
    if len(normalized) < 2:
        raise ValueError("At least two sections are required for a loft")
    return normalized


class Build123dReconstruction:
    """Optional Build123d adapter; imports Build123d only when invoked."""

    def build(self, sections: Any, spec: ReconstructionSpec) -> Any:
        normalized = normalize_sections(sections)
        try:
            import build123d as bd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Build123d is required for reconstruction") from exc

        max_points = int(spec.metadata.get("max_section_points", 32)) if spec.metadata else 32
        groups: Dict[int, list[SectionLoftInput]] = {}
        for section in normalized:
            branch_id = int(section.metadata.get("branch_id", 0))
            groups.setdefault(branch_id, []).append(section)

        lofts = []
        for branch_id, branch_sections in sorted(groups.items()):
            if len(branch_sections) < 2:
                logger.warning("Skipping branch %d: fewer than two valid sections", branch_id)
                continue
            wires = []
            for section in branch_sections:
                points = section.points.copy()
                tangent = np.asarray(section.tangent, dtype=float)
                tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
                # OCC loft profiles must be planar. Centerline cutters can leave
                # sub-micron out-of-plane noise, so project each contour onto its
                # measured normal plane before constructing the Wire.
                if spec.metadata.get("project_profiles", True) if spec.metadata else True:
                    points = points - ((points - section.center) @ tangent)[:, None] * tangent[None, :]
                if max_points >= 3 and len(points) > max_points:
                    take = np.rint(np.linspace(0, len(points) - 1, max_points, endpoint=False)).astype(int)
                    points = points[take]
                vectors = [bd.Vector(float(p[0]), float(p[1]), float(p[2])) for p in points]
                edges = [bd.Edge.make_line(vectors[i], vectors[(i + 1) % len(vectors)]) for i in range(len(vectors))]
                wires.append(bd.Wire(edges))
            try:
                branch_loft = bd.Solid.make_loft(wires, ruled=False)
            except Exception as exc:
                raise RuntimeError(f"Build123d loft failed for branch {branch_id} with {len(wires)} sections and {max_points} contour points") from exc
            if spec.wall_thickness is not None and spec.wall_thickness > 0:
                try:
                    branch_loft = branch_loft.offset_3d(-float(spec.wall_thickness))
                except Exception as exc:
                    logger.warning("Wall offset failed on branch %d; returning lumen loft: %s", branch_id, exc)
            lofts.append(branch_loft)
        if not lofts:
            raise RuntimeError("No branch produced a valid Build123d loft")
        return lofts[0] if len(lofts) == 1 else bd.Compound.make_compound(lofts)
