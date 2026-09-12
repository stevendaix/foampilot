"""Small, dependency-light contracts shared by medical_build backends."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class BoundaryCondition:
    name: str
    kind: str = "patch"
    values: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReconstructionSpec:
    method: str = "smooth"
    wall_thickness: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MedicalBuildConfig:
    project_name: str = "medical_build"
    coordinate_system: str = "input"
    reconstruction: ReconstructionSpec = field(default_factory=ReconstructionSpec)

__all__ = ["BoundaryCondition", "ReconstructionSpec", "MedicalBuildConfig"]
