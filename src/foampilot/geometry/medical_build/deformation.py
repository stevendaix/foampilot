"""Optional local geometric deformations for medical_build.

The reference analysis is never mutated.  A deformation returns a deep-copied
GeometryAnalysisData object and is disabled when ``spec`` is None.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import exp, pi
from typing import Any, Iterable, Mapping

import numpy as np

from .analysis_data import GeometryAnalysisData, SectionRecord


@dataclass(frozen=True)
class LocalDeformationSpec:
    """Parameters for a local radial deformation on selected branches.

    ``center_abscissa`` and ``sigma`` are measured in the branch abscissa
    coordinate.  ``amplitude=0`` is an exact no-op.  The deformation is
    smoothly suppressed inside ``junction_protection`` of either branch end.
    """

    branch_ids: tuple[int, ...] = field(default_factory=tuple)
    center_abscissa: float = 0.0
    sigma: float = 1.0
    amplitude: float = 0.0
    profile: str = "gaussian"
    junction_protection: float = 0.0
    max_scale: float = 3.0

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.profile != "gaussian":
            raise ValueError("Only the gaussian profile is currently supported")
        if self.junction_protection < 0:
            raise ValueError("junction_protection must be non-negative")
        if self.max_scale <= 0:
            raise ValueError("max_scale must be positive")
        if 1.0 + self.amplitude <= 0:
            raise ValueError("amplitude would invert or collapse a section")
        object.__setattr__(self, "branch_ids", tuple(int(v) for v in self.branch_ids))

    def as_dict(self) -> dict[str, Any]:
        return {
            "branch_ids": list(self.branch_ids),
            "center_abscissa": self.center_abscissa,
            "sigma": self.sigma,
            "amplitude": self.amplitude,
            "profile": self.profile,
            "junction_protection": self.junction_protection,
            "max_scale": self.max_scale,
        }


def _smooth_junction_factor(s: float, s_min: float, s_max: float, protection: float) -> float:
    if protection <= 0:
        return 1.0
    if s_max <= s_min or 2.0 * protection >= s_max - s_min:
        return 0.0
    distance = min(s - s_min, s_max - s)
    if distance <= 0:
        return 0.0
    if distance >= protection:
        return 1.0
    x = distance / protection
    return x * x * (3.0 - 2.0 * x)


def _section_scale(section: SectionRecord, spec: LocalDeformationSpec, s_min: float, s_max: float) -> float:
    gaussian = exp(-0.5 * ((section.abscissa - spec.center_abscissa) / spec.sigma) ** 2)
    support = _smooth_junction_factor(section.abscissa, s_min, s_max, spec.junction_protection)
    scale = 1.0 + spec.amplitude * gaussian * support
    return float(np.clip(scale, 1.0 / spec.max_scale, spec.max_scale))


def _polygon_area(points: np.ndarray, center: np.ndarray, normal: np.ndarray, binormal: np.ndarray) -> float:
    x = np.dot(points - center, normal)
    y = np.dot(points - center, binormal)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _perimeter(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum())


def _deform_section(section: SectionRecord, scale: float) -> SectionRecord:
    points = section.center + scale * (section.points - section.center)
    phase_points = section.center + scale * (section.phase_locked_points - section.center)
    area = _polygon_area(points, section.center, section.normal, section.binormal)
    perimeter = _perimeter(points)
    radius = float(np.sqrt(area / pi))
    metadata = dict(section.metadata)
    metadata["local_deformation_scale"] = scale
    return SectionRecord(
        branch_id=section.branch_id,
        station_id=section.station_id,
        abscissa=section.abscissa,
        center=section.center.copy(),
        tangent=section.tangent.copy(),
        normal=section.normal.copy(),
        binormal=section.binormal.copy(),
        points=points,
        phase_locked_points=phase_points,
        area=area,
        perimeter=perimeter,
        equivalent_radius=radius,
        valid=section.valid,
        metadata=metadata,
    )


def apply_local_deformation(
    analysis: GeometryAnalysisData,
    spec: LocalDeformationSpec | None,
) -> GeometryAnalysisData:
    """Return a deformed copy of ``analysis`` without mutating the input."""
    result = deepcopy(analysis)
    if spec is None or spec.amplitude == 0.0 or not spec.branch_ids:
        return result

    selected = set(spec.branch_ids)
    changed = []
    for branch in result.branches:
        if branch.branch_id not in selected or not branch.sections:
            continue
        s_values = np.asarray([section.abscissa for section in branch.sections], dtype=float)
        s_min, s_max = float(s_values.min()), float(s_values.max())
        new_sections = []
        for section in branch.sections:
            scale = _section_scale(section, spec, s_min, s_max)
            new_sections.append(_deform_section(section, scale))
            if abs(scale - 1.0) > 1e-14:
                changed.append((branch.branch_id, section.station_id, scale))
        branch.sections = new_sections
        branch.diagnostics = dict(branch.diagnostics)
        branch.diagnostics["local_deformation"] = spec.as_dict()

    result.metadata = dict(result.metadata)
    result.metadata["local_deformation"] = spec.as_dict()
    result.diagnostics = dict(result.diagnostics)
    result.diagnostics["local_deformation_changed_sections"] = len(changed)
    result.validate()
    return result


def deformation_report(analysis: GeometryAnalysisData) -> dict[str, Any]:
    """Build a JSON-serializable summary of deformation scales."""
    values = []
    for branch in analysis.branches:
        for section in branch.sections:
            scale = section.metadata.get("local_deformation_scale", 1.0)
            values.append(float(scale))
    return {
        "enabled": "local_deformation" in analysis.metadata,
        "parameters": analysis.metadata.get("local_deformation"),
        "sections": len(values),
        "min_scale": min(values) if values else 1.0,
        "max_scale": max(values) if values else 1.0,
        "mean_scale": float(np.mean(values)) if values else 1.0,
    }
