import math
from dataclasses import dataclass
from typing import Tuple

from foampilot.utilities.manageunits import ValueWithUnit


@dataclass
class LocalTransform:
    origin_world: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_deg: float = 0.0

    def world_to_local(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        dx = x - self.origin_world[0]
        dy = y - self.origin_world[1]
        dz = z - self.origin_world[2]

        theta = math.radians(self.rotation_deg)
        c = math.cos(theta)
        s = math.sin(theta)

        xl = dx * c + dy * s
        yl = -dx * s + dy * c
        zl = dz

        return xl, yl, zl

    def local_to_world(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        theta = math.radians(self.rotation_deg)
        c = math.cos(theta)
        s = math.sin(theta)

        xw = x * c - y * s + self.origin_world[0]
        yw = x * s + y * c + self.origin_world[1]
        zw = z + self.origin_world[2]

        return xw, yw, zw
