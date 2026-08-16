from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import json

from foampilot.utilities.manageunits import ValueWithUnit
from foampilot.urban.model.domain import CFDDomain
from foampilot.urban.model.terrain import CFDTerrain


@dataclass
class SimplificationOptions:
    simplify_tolerance: Optional[ValueWithUnit] = None  # auto si None
    min_building_area: ValueWithUnit = ValueWithUnit(1.0, "m^2")
    min_building_height: ValueWithUnit = ValueWithUnit(0.5, "m")
    min_gap: ValueWithUnit = ValueWithUnit(0.5, "m")
    merge_overlapping_buildings: bool = True
    remove_small_holes: bool = True
    hole_area_threshold: ValueWithUnit = ValueWithUnit(0.5, "m^2")
    snap_tolerance: Optional[ValueWithUnit] = None


@dataclass
class CFDBuilding:
    id: str
    footprint_local: Any  # shapely.geometry.Polygon
    ground_z_local: float
    roof_z_local: float
    height: float
    source_building_id: str
    attributes: dict = field(default_factory=dict)


@dataclass
class CFDTerrain:
    pass


@dataclass
class CFDGeometry:
    buildings: List[CFDBuilding]
    terrain: Optional[CFDTerrain]
    domain_box: Tuple[float, float, float, float, float, float]
    lod: str
    metadata: dict = field(default_factory=dict)


class CFDSimplifier:
    def __init__(
        self,
        urban,
        lod: str = "lod1",
        options: Optional[SimplificationOptions] = None,
    ):
        from foampilot.urban.model.urban_model import CFDLOD
        self.urban = urban
        self.lod = CFDLOD(lod) if isinstance(lod, str) else lod
        self.options = options or SimplificationOptions()

    def simplify(
        self,
        wind_frame=None,
        domain: Optional[CFDDomain] = None,
        terrain: Optional[CFDTerrain] = None,
    ) -> CFDGeometry:
        buildings = []
        for b in self.urban.buildings():
            footprint = b.footprint
            ground_z = b.ground_z
            roof_z = b.roof_z

            if wind_frame is not None:
                coords = list(footprint.exterior.coords)
                local_coords = [wind_frame.to_local(x, y, ground_z) for x, y in coords]
                from shapely.geometry import Polygon as ShapelyPolygon
                footprint = ShapelyPolygon(local_coords)
                ground_z_local = local_coords[0][2]
                roof_z_local = roof_z
            else:
                ground_z_local = ground_z
                roof_z_local = roof_z

            buildings.append(CFDBuilding(
                id=b.id,
                footprint_local=footprint,
                ground_z_local=ground_z_local,
                roof_z_local=roof_z_local,
                height=b.height,
                source_building_id=b.id,
                attributes=b.attributes,
            ))

        if domain is not None:
            domain_box = domain.compute_box(self.urban, wind_frame=wind_frame, terrain=terrain)
        else:
            xmin, ymin, zmin, xmax, ymax, zmax = self.urban.bbox()
            if wind_frame is not None:
                corners = [
                    wind_frame.to_local(xmin, ymin, zmin),
                    wind_frame.to_local(xmax, ymin, zmin),
                    wind_frame.to_local(xmin, ymax, zmin),
                    wind_frame.to_local(xmax, ymax, zmin),
                    wind_frame.to_local(xmin, ymin, zmax),
                    wind_frame.to_local(xmax, ymin, zmax),
                    wind_frame.to_local(xmin, ymax, zmax),
                    wind_frame.to_local(xmax, ymax, zmax),
                ]
                xs = [c[0] for c in corners]
                ys = [c[1] for c in corners]
                zs = [c[2] for c in corners]
                domain_box = (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
            else:
                domain_box = (xmin, ymin, zmin, xmax, ymax, zmax)

        return CFDGeometry(
            buildings=buildings,
            terrain=terrain,
            domain_box=domain_box,
            lod=self.lod.value,
            metadata={"wind_frame": wind_frame},
        )
