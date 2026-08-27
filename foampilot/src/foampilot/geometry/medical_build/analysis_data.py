from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import numpy as np


def _array(value: Any, shape: Optional[tuple[int, ...]] = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if shape is not None and array.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Analysis data contains non-finite values")
    return array


@dataclass
class SectionRecord:
    branch_id: int
    station_id: int
    abscissa: float
    center: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    binormal: np.ndarray
    points: np.ndarray
    phase_locked_points: np.ndarray
    area: float
    perimeter: float
    equivalent_radius: float
    valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    closed: bool = True
    min_size: float = 0.0
    max_size: float = 0.0
    shape: float = 1.0
    status: str = "VALID"
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        self.center = _array(self.center, (3,))
        self.tangent = _array(self.tangent, (3,))
        self.normal = _array(self.normal, (3,))
        self.binormal = _array(self.binormal, (3,))
        self.points = _array(self.points)
        self.phase_locked_points = _array(self.phase_locked_points)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or len(self.points) < 3:
            raise ValueError("A section requires at least three 3D points")
        if self.phase_locked_points.ndim != 2 or self.phase_locked_points.shape[1] != 3:
            raise ValueError("phase_locked_points must be an N x 3 array")
        if self.area < 0 or self.perimeter < 0 or self.equivalent_radius < 0:
            raise ValueError("Section measures must be non-negative")
        if self.min_size < 0 or self.max_size < 0 or not (0.0 <= self.shape <= 1.0):
            raise ValueError("Section shape measures are invalid")
        if self.status not in {"VALID", "JUNCTION", "REJECTED"}:
            raise ValueError(f"Unknown section status: {self.status}")

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in ("center", "tangent", "normal", "binormal", "points", "phase_locked_points"):
            data[key] = getattr(self, key).tolist()
        return data


@dataclass
class BranchRecord:
    branch_id: int
    source_cap_id: int
    target_cap_id: int
    points: np.ndarray
    abscissas: np.ndarray
    tangents: np.ndarray
    length: float
    sections: List[SectionRecord] = field(default_factory=list)
    parent_branch_id: Optional[int] = None
    children_branch_ids: List[int] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = _array(self.points)
        self.abscissas = _array(self.abscissas)
        self.tangents = _array(self.tangents)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("Branch points must be an N x 3 array")
        if len(self.abscissas) != len(self.points):
            raise ValueError("Branch abscissas must match branch points")
        if self.tangents.shape != self.points.shape:
            raise ValueError("Branch tangents must match branch points")

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in ("points", "abscissas", "tangents"):
            data[key] = getattr(self, key).tolist()
        data["sections"] = [section.as_dict() for section in self.sections]
        return data


@dataclass
class GeometryAnalysisData:
    coordinate_system: str = "input"
    source_cap_id: Optional[int] = None
    cap_records: List[Dict[str, Any]] = field(default_factory=list)
    branches: List[BranchRecord] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    phase_timings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.branches:
            raise ValueError("Analysis data contains no branches")
        for branch in self.branches:
            if not branch.sections:
                raise ValueError(f"Branch {branch.branch_id} contains no sections")
            stations = [section.abscissa for section in branch.sections]
            if stations != sorted(stations):
                raise ValueError(f"Sections of branch {branch.branch_id} are not ordered")
            for section in branch.sections:
                if not section.valid:
                    continue
                if len(section.points) < 3:
                    raise ValueError(f"Invalid section {section.station_id}")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "coordinate_system": self.coordinate_system,
            "source_cap_id": self.source_cap_id,
            "cap_records": self.cap_records,
            "branches": [branch.as_dict() for branch in self.branches],
            "diagnostics": self.diagnostics,
            "quality_metrics": self.quality_metrics,
            "phase_timings": self.phase_timings,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.validate()
        destination.write_text(json.dumps(self.as_dict(), indent=2, default=str))
        return destination
