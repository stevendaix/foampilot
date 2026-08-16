from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
from shapely.geometry import Polygon


class RoofType(str, Enum):
    FLAT = "flat"
    GABLE = "gable"
    HIP = "hip"
    PYRAMID = "pyramid"
    UNKNOWN = "unknown"


class CFDLOD(str, Enum):
    LOD0 = "lod0"
    LOD1 = "lod1"
    LOD2 = "lod2"
    LOD3 = "lod3"
    LOD4 = "lod4"


@dataclass
class Building:
    id: str
    footprint: Polygon
    ground_z: float
    roof_z: float
    roof_type: RoofType = RoofType.FLAT
    lod: CFDLOD = CFDLOD.LOD1
    source: str = "manual"
    confidence: float = 1.0
    attributes: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.footprint.is_valid:
            raise ValueError(f"Building {self.id}: footprint is not valid")

        if self.roof_z <= self.ground_z:
            raise ValueError(
                f"Building {self.id}: roof_z must be greater than ground_z"
            )

    @property
    def height(self) -> float:
        return self.roof_z - self.ground_z

    @property
    def area(self) -> float:
        return self.footprint.area

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "footprint": list(self.footprint.exterior.coords),
            "ground_z": self.ground_z,
            "roof_z": self.roof_z,
            "roof_type": self.roof_type.value,
            "lod": self.lod.value,
            "source": self.source,
            "confidence": self.confidence,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Building":
        return cls(
            id=data["id"],
            footprint=Polygon(data["footprint"]),
            ground_z=float(data["ground_z"]),
            roof_z=float(data["roof_z"]),
            roof_type=RoofType(data.get("roof_type", "flat")),
            lod=CFDLOD(data.get("lod", "lod1")),
            source=data.get("source", "manual"),
            confidence=float(data.get("confidence", 1.0)),
            attributes=data.get("attributes", {}),
        )


@dataclass
class Terrain:
    pass


@dataclass
class Road:
    pass


@dataclass
class UrbanModelMetadata:
    crs: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[str] = None
    description: Optional[str] = None
    units: str = "meters"


class UrbanModel:
    def __init__(self, crs: Optional[str] = None, metadata: Optional[UrbanModelMetadata] = None):
        self.crs = crs
        self.metadata = metadata or UrbanModelMetadata(crs=crs)
        self._buildings: Dict[str, Building] = {}
        self._terrain: Optional[Terrain] = None
        self._roads: Dict[str, Road] = {}

    def add_building(self, building: Building) -> None:
        if building.id in self._buildings:
            raise ValueError(f"Building {building.id} already exists")
        self._buildings[building.id] = building

    def add_terrain(self, terrain: Terrain) -> None:
        self._terrain = terrain

    def add_road(self, road: Road) -> None:
        pass

    def buildings(self) -> List[Building]:
        return list(self._buildings.values())

    def building_count(self) -> int:
        return len(self._buildings)

    def bbox(self) -> Tuple[float, float, float, float, float, float]:
        if not self._buildings:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        xs = []
        ys = []
        zs_min = []
        zs_max = []

        for b in self._buildings.values():
            minx, miny, maxx, maxy = b.footprint.bounds
            xs.extend([minx, maxx])
            ys.extend([miny, maxy])
            zs_min.append(b.ground_z)
            zs_max.append(b.roof_z)

        return (
            min(xs), min(ys), min(zs_min),
            max(xs), max(ys), max(zs_max),
        )

    def center_xy(self) -> Tuple[float, float, float]:
        xmin, ymin, zmin, xmax, ymax, zmax = self.bbox()
        return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)

    def to_geojson(self, path: Path) -> None:
        import json
        features = []
        for b in self._buildings.values():
            features.append({
                "type": "Feature",
                "properties": {
                    "id": b.id,
                    "ground_z": b.ground_z,
                    "roof_z": b.roof_z,
                    "height": b.height,
                    "roof_type": b.roof_type.value,
                    "lod": b.lod.value,
                    "source": b.source,
                    "confidence": b.confidence,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(b.footprint.exterior.coords)],
                },
            })
        geojson = {
            "type": "FeatureCollection",
            "properties": {
                "crs": self.crs,
                "units": self.metadata.units,
            },
            "features": features,
        }
        path.write_text(json.dumps(geojson, indent=2))

    @classmethod
    def from_geojson(cls, path: Path) -> "UrbanModel":
        import json
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "UrbanModel":
        crs = data.get("crs")
        metadata = UrbanModelMetadata(
            crs=crs,
            source=data.get("source"),
            description=data.get("description"),
        )
        model = cls(crs=crs, metadata=metadata)
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            coords = feature.get("geometry", {}).get("coordinates", [[]])[0]
            if coords:
                building = Building(
                    id=props.get("id", "unknown"),
                    footprint=Polygon(coords),
                    ground_z=float(props.get("ground_z", 0.0)),
                    roof_z=float(props.get("roof_z", 0.0)),
                    roof_type=RoofType(props.get("roof_type", "flat")),
                    lod=CFDLOD(props.get("lod", "lod1")),
                    source=props.get("source", "manual"),
                    confidence=float(props.get("confidence", 1.0)),
                )
                model.add_building(building)
        return model
