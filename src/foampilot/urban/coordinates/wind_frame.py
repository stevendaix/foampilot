import math
from dataclasses import dataclass
from typing import Tuple

from foampilot.urban.coordinates.transforms import LocalTransform


@dataclass
class WindFrame:
    """
    Convention:
        direction_deg = 0 -> flow along world +X
        direction_deg = 90 -> flow along world +Y
        +Z local = vertical, identical to world
    """
    direction_deg: float
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_local(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        dx = x - self.origin[0]
        dy = y - self.origin[1]
        dz = z - self.origin[2]

        theta = math.radians(self.direction_deg)
        c = math.cos(theta)
        s = math.sin(theta)

        xl = dx * c + dy * s
        yl = -dx * s + dy * c
        zl = dz

        return xl, yl, zl

    def to_world(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        theta = math.radians(self.direction_deg)
        c = math.cos(theta)
        s = math.sin(theta)

        xw = x * c - y * s + self.origin[0]
        yw = x * s + y * c + self.origin[1]
        zw = z + self.origin[2]

        return xw, yw, zw

    def to_transform(self) -> LocalTransform:
        return LocalTransform(
            origin_world=self.origin,
            rotation_deg=self.direction_deg,
        )
