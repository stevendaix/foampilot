from dataclasses import dataclass, field
from typing import Optional

from foampilot.urban.bc.patch_types import PatchTypes, FieldBoundaryConditions


@dataclass
class BoundaryConditionConfig:
    patch_types: PatchTypes = field(default_factory=PatchTypes)
    fields: FieldBoundaryConditions = field(default_factory=FieldBoundaryConditions)

    def get_patch_type(self, patch_name: str) -> str:
        mapping = {
            "inlet": self.patch_types.inlet,
            "outlet": self.patch_types.outlet,
            "top": self.patch_types.top,
            "side_left": self.patch_types.side_left,
            "side_right": self.patch_types.side_right,
            "ground": self.patch_types.ground,
            "buildings": self.patch_types.buildings,
        }
        return mapping.get(patch_name.lower(), "patch")
