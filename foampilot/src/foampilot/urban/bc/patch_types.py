from dataclasses import dataclass, field
from typing import Dict

from foampilot.utilities.manageunits import ValueWithUnit


@dataclass
class PatchTypes:
    inlet: str = "patch"
    outlet: str = "patch"
    top: str = "symmetryPlane"
    side_left: str = "symmetryPlane"
    side_right: str = "symmetryPlane"
    ground: str = "wall"
    buildings: str = "wall"

    def to_dict(self) -> Dict[str, str]:
        return {
            "inlet": self.inlet,
            "outlet": self.outlet,
            "top": self.top,
            "side_left": self.side_left,
            "side_right": self.side_right,
            "ground": self.ground,
            "buildings": self.buildings,
        }


@dataclass
class FieldBoundaryConditions:
    U_inlet: str = "fixedValue"
    U_outlet: str = "pressureInletOutletVelocity"
    p_inlet: str = "zeroGradient"
    p_outlet: str = "fixedValue"
    k_inlet: str = "fixedValue"
    k_outlet: str = "inletOutlet"
    omega_inlet: str = "fixedValue"
    omega_outlet: str = "inletOutlet"
    nut_ground: str = "nutkRoughWallFunction"
    nut_buildings: str = "nutkRoughWallFunction"

    def to_dict(self) -> Dict[str, str]:
        return {
            "U_inlet": self.U_inlet,
            "U_outlet": self.U_outlet,
            "p_inlet": self.p_inlet,
            "p_outlet": self.p_outlet,
            "k_inlet": self.k_inlet,
            "k_outlet": self.k_outlet,
            "omega_inlet": self.omega_inlet,
            "omega_outlet": self.omega_outlet,
            "nut_ground": self.nut_ground,
            "nut_buildings": self.nut_buildings,
        }


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
