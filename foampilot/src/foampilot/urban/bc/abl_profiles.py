import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path

from foampilot.utilities.manageunits import ValueWithUnit


@dataclass
class ABLProfile:
    reference_height: ValueWithUnit = ValueWithUnit(10.0, "m")
    reference_velocity: ValueWithUnit = ValueWithUnit(10.0, "m/s")
    roughness_length: ValueWithUnit = ValueWithUnit(0.3, "m")
    kappa: float = 0.41
    model: str = "log"

    def friction_velocity(self) -> float:
        z_ref = self.reference_height.get_in("m")
        u_ref = self.reference_velocity.get_in("m/s")
        z0 = self.roughness_length.get_in("m")
        return u_ref * self.kappa / math.log(z_ref / z0)

    def velocity_at_height(self, z: ValueWithUnit) -> ValueWithUnit:
        z_m = z.get_in("m")
        z0 = self.roughness_length.get_in("m")
        u_star = self.friction_velocity()
        u = (u_star / self.kappa) * math.log(max(z_m, z0 + 1e-6) / z0)
        return ValueWithUnit(u, "m/s")
