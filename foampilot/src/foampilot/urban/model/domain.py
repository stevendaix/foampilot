import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from foampilot.utilities.manageunits import ValueWithUnit


ReferenceHeightMethod = Literal["Hmax", "Hmean", "H90", "H95", "custom"]
ExtentUnits = Literal["href", "meters"]


@dataclass
class CFDDomain:
    upstream: float = 8.0
    downstream: float = 15.0
    lateral: float = 4.0
    top: float = 2.5
    extent_units: ExtentUnits = "href"
    reference_height_method: ReferenceHeightMethod = "Hmax"
    custom_reference_height: Optional[ValueWithUnit] = None

    def compute_reference_height(self, urban) -> float:
        if self.reference_height_method == "custom":
            if self.custom_reference_height is None:
                raise ValueError("custom_reference_height must be provided")
            return self.custom_reference_height.get_in("m")

        buildings = urban.buildings()
        if not buildings:
            return 0.0

        heights = [b.height for b in buildings]

        if self.reference_height_method == "Hmax":
            return max(heights)

        if self.reference_height_method == "Hmean":
            return sum(heights) / len(heights)

        # H90 / H95 : percentile pondéré par surface au sol
        areas = [b.area for b in buildings]
        total_area = sum(areas)
        if total_area == 0:
            return max(heights) if heights else 0.0

        # Tri par hauteur croissante
        sorted_pairs = sorted(zip(heights, areas), key=lambda x: x[0])
        cumsum = 0.0
        for h, a in sorted_pairs:
            cumsum += a
            if cumsum / total_area >= 0.90 and self.reference_height_method == "H90":
                return h
            if cumsum / total_area >= 0.95 and self.reference_height_method == "H95":
                return h

        return max(heights)

    def compute_box(
        self,
        urban,
        wind_frame=None,
        terrain=None,
    ) -> Tuple[float, float, float, float, float, float]:
        href = self.compute_reference_height(urban)

        upstream = self.upstream * href if self.extent_units == "href" else self.upstream
        downstream = self.downstream * href if self.extent_units == "href" else self.downstream
        lateral = self.lateral * href if self.extent_units == "href" else self.lateral
        top = self.top * href if self.extent_units == "href" else self.top

        xmin, ymin, zmin_buildings, xmax, ymax, zmax_buildings = urban.bbox()

        zmin_terrain = 0.0
        zmax_terrain = 0.0
        if terrain is not None and terrain.points:
            xmin_terrain, ymin_terrain, zmin_terrain, xmax_terrain, ymax_terrain, zmax_terrain = terrain.get_bounds()
            xmin = min(xmin, xmin_terrain)
            ymin = min(ymin, ymin_terrain)
            xmax = max(xmax, xmax_terrain)
            ymax = max(ymax, ymax_terrain)

        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0

        # Si WindFrame fourni, calculer la bbox dans le repère local
        if wind_frame is not None:
            zmin_world = min(zmin_buildings, zmin_terrain)
            zmax_world = max(zmax_buildings, zmax_terrain)
            corners = [
                wind_frame.to_local(xmin, ymin, zmin_world),
                wind_frame.to_local(xmax, ymin, zmin_world),
                wind_frame.to_local(xmin, ymax, zmin_world),
                wind_frame.to_local(xmax, ymax, zmin_world),
                wind_frame.to_local(xmin, ymin, zmax_world),
                wind_frame.to_local(xmax, ymin, zmax_world),
                wind_frame.to_local(xmin, ymax, zmax_world),
                wind_frame.to_local(xmax, ymax, zmax_world),
            ]
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            zs = [c[2] for c in corners]
            local_xmin = min(xs) - upstream
            local_xmax = max(xs) + downstream
            local_ymin = min(ys) - lateral
            local_ymax = max(ys) + lateral
            local_zmin = min(zs)
            local_zmax = max(zs) + top
            return (local_xmin, local_ymin, local_zmin, local_xmax, local_ymax, local_zmax)

        # Repère monde : vent le long de X
        xmin_local = xmin - upstream
        xmax_local = xmax + downstream
        ymin_local = cy - lateral
        ymax_local = cy + lateral
        zmin_local = min(zmin_buildings, zmin_terrain)
        zmax_local = max(zmax_buildings, zmax_terrain) + top

        return (xmin_local, ymin_local, zmin_local, xmax_local, ymax_local, zmax_local)


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
