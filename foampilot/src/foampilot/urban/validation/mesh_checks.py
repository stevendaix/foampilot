from dataclasses import dataclass
from typing import List, Optional

from foampilot.utilities.manageunits import ValueWithUnit


@dataclass
class MeshValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: dict = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class MeshValidator:
    def __init__(self, options: Optional[dict] = None):
        self.options = options or {}

    def validate(self, mesh) -> MeshValidationResult:
        return MeshValidationResult(valid=True, errors=[], warnings=[])
